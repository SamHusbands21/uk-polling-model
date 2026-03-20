"""
aggregator.py
-------------
Multi-party poll aggregator using a Kalman filter with EM-estimated pollster
house effects. Each party's true support is modelled as a latent random walk;
each poll observation is true_support + house_effect[pollster] + noise.

The EM loop alternates between:
  E-step: run Kalman smoother with current house-effect estimates
  M-step: re-estimate house effects as mean residual per pollster

After convergence the state is renormalised so party shares sum to 1.

State is persisted to data/aggregator_state.pkl so incremental runs only
process new polls rather than refitting from scratch.

Run as a module:  python -m src.aggregator
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.scraper import PARTIES, load as load_polls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_PATH = DATA_DIR / "aggregator_state.pkl"
PROCESSED_DIR = DATA_DIR / "processed"
SMOOTHED_PATH = PROCESSED_DIR / "smoothed_shares.parquet"

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# Process noise: how much the true support can move per day (std dev in pp)
PROCESS_NOISE_STD = 0.15  # percentage points per day — used for major parties

# Party-specific process noise overrides for regional/minor parties whose
# true support changes more slowly (and whose polls are sparse + rounded)
PROCESS_NOISE_BY_PARTY: dict[str, float] = {
    "snp":    0.05,   # SNP polling is more stable; sparse Scottish sub-samples
    "pc":     0.02,   # Plaid Cymru is very stable; very sparse Welsh sub-samples
    "green":  0.10,   # Greens can surge but not as volatile as Reform
}

# Observation noise: poll sampling error (std dev in pp)
OBS_NOISE_STD = 2.0  # percentage points

# EM convergence
EM_MAX_ITER = 30
EM_TOL = 1e-4  # max change in house-effect estimates across iterations

# Confidence interval z-score (90% → 1.645)
CI_Z = 1.645


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PartyState:
    """Kalman filter state for a single party."""
    mean: float          # current latent support estimate (pp)
    variance: float      # current state variance


@dataclass
class AggregatorState:
    """Persisted state for incremental runs."""
    # Last date included in the filter (ISO string)
    last_date: str = "2024-07-04"
    # Kalman state per party at last_date
    party_states: dict[str, PartyState] = field(default_factory=dict)
    # House effects: pollster → party → bias (pp)
    house_effects: dict[str, dict[str, float]] = field(default_factory=dict)
    # Daily smoothed series (list of dicts, one per day)
    history: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kalman filter (univariate, run independently per party)
# ---------------------------------------------------------------------------

def _kalman_smooth(
    observations: np.ndarray,       # shape (T,) — NaN where no poll on that day
    obs_weights: np.ndarray,        # shape (T,) — effective number of polls (weight)
    init_mean: float,
    init_var: float,
    q: float,                       # process noise variance (per day)
    r: float,                       # base observation noise variance
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a fixed-interval Kalman smoother on a daily time series.

    Returns (smoothed_means, smoothed_vars), both shape (T,).
    Days with no observation use the predictive distribution.
    Multiple polls on the same day are combined into a weighted mean before
    entering the filter.
    """
    T = len(observations)
    filtered_means = np.zeros(T)
    filtered_vars = np.zeros(T)

    m = init_mean
    v = init_var

    for t in range(T):
        # Predict
        m_pred = m
        v_pred = v + q

        # Update (only if we have observation(s) on this day)
        if obs_weights[t] > 0 and not np.isnan(observations[t]):
            # Scale observation noise by effective sample count
            r_t = r / max(obs_weights[t], 1.0)
            k = v_pred / (v_pred + r_t)
            m = m_pred + k * (observations[t] - m_pred)
            v = (1 - k) * v_pred
        else:
            m = m_pred
            v = v_pred

        filtered_means[t] = m
        filtered_vars[t] = v

    # Rauch-Tung-Striebel smoother (backward pass)
    smoothed_means = filtered_means.copy()
    smoothed_vars = filtered_vars.copy()

    for t in range(T - 2, -1, -1):
        g = filtered_vars[t] / (filtered_vars[t] + q)
        smoothed_means[t] = (
            filtered_means[t]
            + g * (smoothed_means[t + 1] - filtered_means[t])
        )
        smoothed_vars[t] = (
            filtered_vars[t]
            + g ** 2 * (smoothed_vars[t + 1] - filtered_vars[t] - q)
        )

    return smoothed_means, smoothed_vars


# ---------------------------------------------------------------------------
# EM + full aggregator
# ---------------------------------------------------------------------------

def _build_daily_grid(
    polls: pd.DataFrame,
    house_effects: dict[str, dict[str, float]],
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Map polls to a daily grid, subtracting house effects.

    Returns (dates, obs_dict, weight_dict) where obs_dict[party] is an array
    of house-effect-corrected poll shares (NaN where no poll) and weight_dict
    counts the effective number of polls per day.
    """
    min_date = polls["fieldwork_end"].min()
    max_date = polls["fieldwork_end"].max()
    dates = pd.date_range(min_date, max_date, freq="D")
    T = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    obs: dict[str, list[list[float]]] = {p: [[] for _ in range(T)] for p in PARTIES}

    for _, row in polls.iterrows():
        d = row["fieldwork_end"]
        if d not in date_idx:
            continue
        t = date_idx[d]
        pollster = row["pollster"]
        he = house_effects.get(pollster, {})
        for p in PARTIES:
            val = row.get(p)
            if pd.isna(val) or val is None:
                continue
            corrected = float(val) - he.get(p, 0.0)
            obs[p][t].append(corrected)

    obs_arr: dict[str, np.ndarray] = {}
    wt_arr: dict[str, np.ndarray] = {}

    for p in PARTIES:
        o = np.full(T, np.nan)
        w = np.zeros(T)
        for t in range(T):
            vals = obs[p][t]
            if vals:
                o[t] = float(np.mean(vals))
                w[t] = float(len(vals))
        obs_arr[p] = o
        wt_arr[p] = w

    return dates, obs_arr, wt_arr


def _estimate_house_effects(
    polls: pd.DataFrame,
    smoothed: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> dict[str, dict[str, float]]:
    """
    M-step: estimate each pollster's house effect as the mean residual
    between their polls and the smoothed state on those poll dates.
    """
    date_idx = {d: i for i, d in enumerate(dates)}
    residuals: dict[str, dict[str, list[float]]] = {}

    for _, row in polls.iterrows():
        d = row["fieldwork_end"]
        if d not in date_idx:
            continue
        t = date_idx[d]
        pollster = row["pollster"]
        if pollster not in residuals:
            residuals[pollster] = {p: [] for p in PARTIES}
        for p in PARTIES:
            val = row.get(p)
            if pd.isna(val) or val is None:
                continue
            residuals[pollster][p].append(float(val) - smoothed[p][t])

    house_effects: dict[str, dict[str, float]] = {}
    for pollster, party_res in residuals.items():
        house_effects[pollster] = {}
        for p in PARTIES:
            vals = party_res[p]
            house_effects[pollster][p] = float(np.mean(vals)) if vals else 0.0

    return house_effects


def _renormalise(smoothed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Renormalise so shares sum to 100 at each time step.
    Parties with NaN are excluded from the sum and left as NaN.
    """
    parties_with_data = [p for p in PARTIES if not np.all(np.isnan(smoothed[p]))]
    T = len(next(iter(smoothed.values())))
    totals = np.zeros(T)
    for p in parties_with_data:
        arr = smoothed[p].copy()
        arr[np.isnan(arr)] = 0.0
        totals += arr

    result = {}
    for p in PARTIES:
        arr = smoothed[p].copy()
        mask = ~np.isnan(arr)
        arr[mask] = arr[mask] / np.maximum(totals[mask], 1.0) * 100.0
        result[p] = arr
    return result


def run_em(
    polls: pd.DataFrame,
    init_state: AggregatorState | None = None,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """
    Run the full EM loop and return:
      (dates, smoothed_means, smoothed_vars, house_effects)
    """
    q = PROCESS_NOISE_STD ** 2
    r = OBS_NOISE_STD ** 2

    # Initialise house effects
    if init_state and init_state.house_effects:
        house_effects = init_state.house_effects
    else:
        house_effects = {}

    # Rough initial means from the first week of polls
    early = polls[polls["fieldwork_end"] <= polls["fieldwork_end"].min() + pd.Timedelta(days=14)]
    init_means: dict[str, float] = {}
    for p in PARTIES:
        vals = early[p].dropna()
        init_means[p] = float(vals.mean()) if len(vals) > 0 else 25.0

    init_var = 25.0  # wide initial uncertainty (5pp std dev)

    prev_he_flat = np.array([
        v for he in house_effects.values() for v in he.values()
    ]) if house_effects else np.zeros(1)

    smoothed_means: dict[str, np.ndarray] = {}
    smoothed_vars: dict[str, np.ndarray] = {}
    dates: pd.DatetimeIndex = pd.DatetimeIndex([])

    for iteration in range(EM_MAX_ITER):
        dates, obs_arr, wt_arr = _build_daily_grid(polls, house_effects)

        # E-step: Kalman smoother per party (use party-specific process noise if set)
        smoothed_means = {}
        smoothed_vars = {}
        for p in PARTIES:
            q_p = PROCESS_NOISE_BY_PARTY.get(p, PROCESS_NOISE_STD) ** 2
            sm, sv = _kalman_smooth(
                obs_arr[p], wt_arr[p],
                init_mean=init_means[p],
                init_var=init_var,
                q=q_p, r=r,
            )
            smoothed_means[p] = sm
            smoothed_vars[p] = sv

        # Renormalise after E-step
        smoothed_means = _renormalise(smoothed_means)

        # M-step: update house effects
        house_effects = _estimate_house_effects(polls, smoothed_means, dates)

        # Check convergence
        new_he_flat = np.array([
            v for he in house_effects.values() for v in he.values()
        ])
        if len(new_he_flat) == len(prev_he_flat):
            delta = np.max(np.abs(new_he_flat - prev_he_flat))
            logger.info("EM iteration %d: max house-effect change = %.5f pp", iteration + 1, delta)
            if delta < EM_TOL:
                logger.info("EM converged after %d iterations", iteration + 1)
                break
        prev_he_flat = new_he_flat

    return dates, smoothed_means, smoothed_vars, house_effects


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

def _estimate_covariance(
    polls: pd.DataFrame,
    smoothed: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> np.ndarray:
    """
    Estimate the current-day covariance matrix across parties from the
    cross-poll residuals. Used for correlated Monte Carlo sampling.
    Returns a (n_parties × n_parties) matrix in percentage-point units.
    """
    date_idx = {d: i for i, d in enumerate(dates)}
    residual_vectors: list[np.ndarray] = []

    for _, row in polls.iterrows():
        d = row["fieldwork_end"]
        if d not in date_idx:
            continue
        t = date_idx[d]
        vec = []
        for p in PARTIES:
            val = row.get(p)
            if pd.isna(val) or val is None:
                vec.append(np.nan)
            else:
                vec.append(float(val) - smoothed[p][t])
        if not all(np.isnan(v) for v in vec):
            residual_vectors.append(np.array(vec))

    if len(residual_vectors) < 5:
        return np.eye(len(PARTIES)) * OBS_NOISE_STD ** 2

    mat = np.array(residual_vectors)
    # Replace NaN columns with 0 for covariance computation
    col_means = np.nanmean(mat, axis=0)
    for j in range(mat.shape[1]):
        mask = np.isnan(mat[:, j])
        mat[mask, j] = col_means[j]

    cov = np.cov(mat.T)
    return cov


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def aggregate(force_full: bool = False) -> AggregatorState:
    """
    Run the poll aggregator. Loads polls from processed parquet, runs EM,
    persists state to aggregator_state.pkl, saves smoothed series to parquet.

    force_full: if True, refit from scratch ignoring any cached state.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    polls = load_polls()
    logger.info("Loaded %d polls", len(polls))

    # Load existing state for incremental runs
    prev_state: AggregatorState | None = None
    if STATE_PATH.exists() and not force_full:
        prev_state = load_state()
        logger.info("Loaded aggregator state (last date: %s)", prev_state.last_date)

        # Check if there are any new polls since last run
        last_dt = pd.Timestamp(prev_state.last_date)
        new_polls = polls[polls["fieldwork_end"] > last_dt]
        if new_polls.empty:
            logger.info("No new polls since %s — skipping refit", prev_state.last_date)
            return prev_state

    # Run EM on all polls (full refit ensures house effects are well-estimated)
    logger.info("Running EM …")
    dates, smoothed_means, smoothed_vars, house_effects = run_em(polls, init_state=prev_state)

    # Estimate cross-party covariance
    cov_matrix = _estimate_covariance(polls, smoothed_means, dates)

    # Build daily history list
    history: list[dict] = []
    for i, d in enumerate(dates):
        row: dict = {"date": d.strftime("%Y-%m-%d")}
        for p in PARTIES:
            mean = smoothed_means[p][i]
            std = float(np.sqrt(max(smoothed_vars[p][i], 0.0)))
            row[p] = round(float(mean) / 100.0, 4) if not np.isnan(mean) else None
            row[f"{p}_lo90"] = round((float(mean) - CI_Z * std) / 100.0, 4) if not np.isnan(mean) else None
            row[f"{p}_hi90"] = round((float(mean) + CI_Z * std) / 100.0, 4) if not np.isnan(mean) else None
        history.append(row)

    # Current estimates (last day)
    last_means: dict[str, float] = {}
    last_stds: dict[str, float] = {}
    for p in PARTIES:
        last_means[p] = float(smoothed_means[p][-1])
        last_stds[p] = float(np.sqrt(max(smoothed_vars[p][-1], 0.0)))

    # Build current party states
    party_states = {
        p: PartyState(mean=last_means[p], variance=float(smoothed_vars[p][-1]))
        for p in PARTIES
    }

    state = AggregatorState(
        last_date=dates[-1].strftime("%Y-%m-%d"),
        party_states=party_states,
        house_effects=house_effects,
        history=history,
    )
    # Store the covariance matrix on the state object for the seat projector
    state.cov_matrix = cov_matrix  # type: ignore[attr-defined]
    state.parties = PARTIES         # type: ignore[attr-defined]

    # Persist state as a plain dict so it loads correctly regardless of
    # which module is __main__ (avoids AggregatorState pickle class mismatch).
    with open(STATE_PATH, "wb") as f:
        pickle.dump(_state_to_dict(state), f)
    logger.info("Saved aggregator state to %s", STATE_PATH)

    # Save smoothed series to parquet
    smoothed_df = pd.DataFrame(history)
    smoothed_df.to_parquet(SMOOTHED_PATH, index=False)
    logger.info("Saved smoothed series (%d days) to %s", len(smoothed_df), SMOOTHED_PATH)

    return state


def _state_to_dict(state: AggregatorState) -> dict:
    """Serialise an AggregatorState to a plain dict (no class references)."""
    return {
        "last_date": state.last_date,
        "party_states": {
            p: {"mean": float(ps.mean), "variance": float(ps.variance)}
            for p, ps in state.party_states.items()
        },
        "house_effects": {
            pollster: {p: float(v) for p, v in effects.items()}
            for pollster, effects in state.house_effects.items()
        },
        "history": state.history,
        "cov_matrix": getattr(state, "cov_matrix", None),
        "parties": getattr(state, "parties", PARTIES),
    }


def _dict_to_state(d: dict) -> AggregatorState:
    """Reconstruct an AggregatorState from a plain dict."""
    party_states = {
        p: PartyState(mean=v["mean"], variance=v["variance"])
        for p, v in d["party_states"].items()
    }
    state = AggregatorState(
        last_date=d["last_date"],
        party_states=party_states,
        house_effects=d["house_effects"],
        history=d["history"],
    )
    state.cov_matrix = d.get("cov_matrix")  # type: ignore[attr-defined]
    state.parties = d.get("parties", PARTIES)  # type: ignore[attr-defined]
    return state


def load_state() -> AggregatorState:
    """Load the persisted aggregator state."""
    if not STATE_PATH.exists():
        raise FileNotFoundError(
            f"{STATE_PATH} not found. Run aggregate() or python -m src.aggregator first."
        )
    with open(STATE_PATH, "rb") as f:
        raw = pickle.load(f)
    # Support both new dict format and old pickled-dataclass format
    if isinstance(raw, dict):
        return _dict_to_state(raw)
    return raw  # backwards compat for any old .pkl files


def current_estimates(state: AggregatorState) -> dict[str, dict[str, float]]:
    """
    Return the current national vote-share estimates as a dict:
    {party: {mean, lo90, hi90, std}}
    Values are in [0, 1] (fractions, not percentages).
    """
    result = {}
    for p, ps in state.party_states.items():
        std = float(np.sqrt(max(ps.variance, 0.0)))
        result[p] = {
            "mean": round(ps.mean / 100.0, 4),
            "lo90": round((ps.mean - CI_Z * std) / 100.0, 4),
            "hi90": round((ps.mean + CI_Z * std) / 100.0, 4),
            "std": round(std / 100.0, 4),
        }
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    state = aggregate(force_full=True)
    estimates = current_estimates(state)
    print("\nCurrent national estimates:")
    for party, est in sorted(estimates.items(), key=lambda x: -x[1]["mean"]):
        print(
            f"  {party.upper():8s}  {est['mean']*100:.1f}%  "
            f"[{est['lo90']*100:.1f}%–{est['hi90']*100:.1f}%]"
        )
    print(f"\nHouse effects ({len(state.house_effects)} pollsters):")
    for pollster, effects in sorted(state.house_effects.items()):
        he_str = "  ".join(f"{p}:{v:+.1f}" for p, v in effects.items() if abs(v) > 0.2)
        if he_str:
            print(f"  {pollster}: {he_str}")
