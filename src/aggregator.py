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

from src.pollster_reputation import reputation_weight, tier_summary
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

# Party-specific process noise overrides for parties whose true support
# genuinely changes more slowly than GB-wide vote intent, OR whose polls
# are sparse enough that the filter needs regularisation.
#
# SNP and Plaid both stay lower because their samples come from small
# Scottish / Welsh sub-samples with heavy sampling noise — the low process
# noise stops the filter chasing sub-sample jitter.
#
# Green used to be overridden to 0.10 with the comment "Greens can surge
# but not as volatile as Reform". That assumption is no longer safe — once
# Greens are a 15%+ party they can move as fast as the majors, and the
# dampened override was actively preventing the filter from tracking the
# 2026 surge through a disagreeing pollster market. Reverted to the default.
PROCESS_NOISE_BY_PARTY: dict[str, float] = {
    "snp":    0.05,   # SNP polling is more stable; sparse Scottish sub-samples
    "pc":     0.02,   # Plaid Cymru is very stable; very sparse Welsh sub-samples
}

# Observation noise: poll sampling error (std dev in pp)
OBS_NOISE_STD = 2.0  # percentage points

# EM convergence
EM_MAX_ITER = 30
EM_TOL = 1e-4  # max change in house-effect estimates across iterations

# Confidence interval z-score (90% → 1.645)
CI_Z = 1.645

# Half-life (in days) for recency-weighting residuals when estimating
# house effects. Polls from `HE_HALFLIFE_DAYS` ago contribute half as much
# to a pollster's estimated bias as today's polls. Shorter = more
# responsive to current-era pollster methodology; longer = more stable.
# 30 days is aggressive — pollsters that have caught up with a shifting
# market (e.g. More in Common moving from 8% to 13% on Green) see their
# HE collapse toward zero within a couple of fresh polls, instead of
# being dragged by a year of historical residuals. Makes the aggregator
# genuinely tracking current-era pollster behaviour rather than
# long-run averages.
HE_HALFLIFE_DAYS = 30.0

# Fraction of pollsters to trim from each end (by house-effect value)
# before computing the zero-mean anchor. 0.10 means the most extreme
# ~10% of pollsters on each side are dropped from the anchor calculation
# for each party, which stops a single extreme-HE pollster (e.g. a
# one-off poll from a Tier 1 house that happens to be a big outlier)
# dragging the "market centre" for that party. The trim is applied on
# top of reputation weighting — the trimmed-down pollsters still
# contribute observations in the E-step, they just don't vote on where
# zero sits. Falls back to untrimmed if there are fewer than
# HE_ANCHOR_TRIM_MIN_N pollsters classified for the party.
HE_ANCHOR_TRIM = 0.10
HE_ANCHOR_TRIM_MIN_N = 5

# Half-life (in days) for down-weighting the actual poll observations fed
# to the Kalman filter. Distinct from HE_HALFLIFE_DAYS: that one controls
# the M-step house-effect estimation; this one controls how much each poll
# pulls the latent in the E-step. A longer half-life than HE is correct —
# observations are the filter's signal so we don't want to throw history
# away, just lean slightly toward recent evidence. 90 days gives fresh
# evidence of a shifting market (e.g. the post-Reform Green surge)
# meaningful pull while still keeping enough history for stable filter
# variance estimates.
OBS_HALFLIFE_DAYS = 90.0

# Soft cap on effective number of polls contributed by any single
# pollster. Each of their polls is multiplied by min(1, CAP / n_polls),
# where n_polls is that pollster's total contribution to the dataset.
# Rationale: one weekly pollster (Techne, ~40 polls in a year) should
# not be able to dominate the filter's E-step simply by out-publishing
# the BPC core. With a cap of 12 a weekly pollster over a year counts
# as ~12 fortnightly polls, which is roughly the right volume to match
# the less-frequent Tier 1 houses that poll monthly. Set to a large
# number to effectively disable.
POLLSTER_VOLUME_CAP = 12.0

# Sample-size weighting bounds. Each poll's weight is clip(n / 1000, LO, HI),
# where n is the reported sample size. The lower bound stops tiny polls
# being ignored entirely; the upper bound stops a single 17k-sample YouGov
# MRP from drowning the day's market entirely (it still gets ~3x a standard
# 1k-sample poll, which is about right).
SAMPLE_SIZE_WEIGHT_LO = 0.5
SAMPLE_SIZE_WEIGHT_HI = 3.0
SAMPLE_SIZE_WEIGHT_REF = 1000.0  # reference sample size = weight 1.0


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

def _poll_weight(sample_size) -> float:
    """
    Convert a poll's sample size into a Kalman observation weight.

    Kalman-optimal weighting is proportional to 1 / sampling-variance ~ n, so
    we take n / REF and clip to [LO, HI] to stop a single MRP dominating and
    to give small sub-sample polls some say. Missing or malformed sample
    sizes fall back to 1.0 (treated as a standard 1k-sample poll).
    """
    if sample_size is None:
        return 1.0
    try:
        n = float(sample_size)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(n) or n <= 0:
        return 1.0
    raw = n / SAMPLE_SIZE_WEIGHT_REF
    return float(np.clip(raw, SAMPLE_SIZE_WEIGHT_LO, SAMPLE_SIZE_WEIGHT_HI))


def _build_daily_grid(
    polls: pd.DataFrame,
    house_effects: dict[str, dict[str, float]],
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Map polls to a daily grid, subtracting house effects.

    Returns (dates, obs_dict, weight_dict) where obs_dict[party] is the
    sample-size- and recency-weighted mean of house-effect-corrected poll
    shares on each day (NaN where no poll), and weight_dict is the total
    effective weight on that day (sum of per-poll weights).

    Per-poll weight is
        sample(n) * recency(days_old) * reputation(pollster) * volume_cap(pollster),
    where:
      - sample(n) = clip(n / 1000, 0.5, 3.0); polls without a recorded
        sample size fall back to 1.0.
      - recency(d) = 0.5 ** (days_old / OBS_HALFLIFE_DAYS);
        polls from ~3 months ago contribute half as much as today's.
      - reputation(pollster) is the hand-curated tier weight from
        src/pollster_reputation.py (1.00 / 0.60 / 0.30 for tiers 1/2/3).
      - volume_cap(pollster) = min(1, CAP / n_polls) where n_polls is the
        pollster's total count in the dataset. Caps any single pollster's
        effective contribution to ~CAP polls so a weekly publisher like
        Techne can't drown out the less-frequent Tier 1 houses just by
        out-publishing them. Reputation and volume are complementary:
        reputation adjusts for quality, volume adjusts for dominance.

    Recency weighting here (on top of the filter's implicit forgetting via
    process noise) is what actually lets the aggregator respond to fresh
    evidence from a shifting market — especially for volatile parties like
    Green — instead of being anchored by two years of stable historical
    polls.
    """
    min_date = polls["fieldwork_end"].min()
    max_date = polls["fieldwork_end"].max()
    dates = pd.date_range(min_date, max_date, freq="D")
    T = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    # Pre-compute per-pollster volume-cap scale factor. A pollster with
    # more than POLLSTER_VOLUME_CAP polls in the dataset has each of
    # their polls multiplied by CAP / n_polls, so their total effective
    # contribution is at most CAP polls regardless of publication cadence.
    pollster_counts = polls["pollster"].value_counts().to_dict()
    volume_scale = {
        ps: min(1.0, POLLSTER_VOLUME_CAP / n) if n > 0 else 1.0
        for ps, n in pollster_counts.items()
    }

    # Per day, per party: list of (weight, corrected_share) tuples
    obs: dict[str, list[list[tuple[float, float]]]] = {
        p: [[] for _ in range(T)] for p in PARTIES
    }

    for _, row in polls.iterrows():
        d = row["fieldwork_end"]
        if d not in date_idx:
            continue
        t = date_idx[d]
        pollster = row["pollster"]
        he = house_effects.get(pollster, {})
        w_sample = _poll_weight(row.get("sample_size"))
        days_old = max((max_date - d).days, 0)
        w_recency = 0.5 ** (days_old / OBS_HALFLIFE_DAYS)
        w_rep = reputation_weight(pollster)
        w_vol = volume_scale.get(pollster, 1.0)
        w_poll = w_sample * w_recency * w_rep * w_vol
        for p in PARTIES:
            val = row.get(p)
            if pd.isna(val) or val is None:
                continue
            corrected = float(val) - he.get(p, 0.0)
            obs[p][t].append((w_poll, corrected))

    obs_arr: dict[str, np.ndarray] = {}
    wt_arr: dict[str, np.ndarray] = {}

    for p in PARTIES:
        o = np.full(T, np.nan)
        w = np.zeros(T)
        for t in range(T):
            entries = obs[p][t]
            if not entries:
                continue
            total_w = sum(wi for wi, _ in entries)
            if total_w <= 0:
                continue
            o[t] = sum(wi * v for wi, v in entries) / total_w
            w[t] = total_w
        obs_arr[p] = o
        wt_arr[p] = w

    return dates, obs_arr, wt_arr


def _trimmed_weighted_mean(
    pairs: list[tuple[float, float]],
    trim_frac: float = HE_ANCHOR_TRIM,
    min_n_for_trim: int = HE_ANCHOR_TRIM_MIN_N,
) -> float | None:
    """
    Weighted mean of `pairs` = [(value, weight), ...] with the most
    extreme `trim_frac` fraction of pollsters dropped from each end
    (after sorting by value).

    The number of pollsters dropped per end is ``max(1, int(trim_frac * n))``
    when ``n >= min_n_for_trim``; otherwise the function falls back to
    the plain weighted mean so the anchor still works on sparse party
    subsets (e.g. SNP / Plaid).
    """
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return None
    n = len(pairs)
    if n < min_n_for_trim:
        return sum(v * w for v, w in pairs) / total_w
    k = max(1, int(trim_frac * n))
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    middle = sorted_pairs[k:n - k]
    mw = sum(w for _, w in middle)
    if mw <= 0:
        return sum(v * w for v, w in pairs) / total_w
    return sum(v * w for v, w in middle) / mw


def _estimate_house_effects(
    polls: pd.DataFrame,
    smoothed: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
) -> dict[str, dict[str, float]]:
    """
    M-step: estimate each pollster's house effect as the RECENCY-WEIGHTED
    mean residual between their polls and the smoothed state on those poll
    dates, then ANCHOR house effects so they have zero mean across
    pollsters per party.

    Two mechanics layered on top of the raw mean residual:

    1. Recency decay (HE_HALFLIFE_DAYS, default 60 days). Older residuals
       are exponentially down-weighted so a pollster's bias reflects their
       CURRENT behaviour, not their average over the full history. Stops
       YouGov's 2024 near-zero residuals flattening their 2026 +Green
       residuals.

    2. Zero-mean anchoring (reputation-weighted, trimmed). Without any
       constraint the joint {latent state, house effects} posterior is
       underdetermined — the filter can add +3 pp to every pollster's HE
       and subtract 3 pp from the latent with no penalty. Forcing the
       mean HE per party to zero across pollsters removes that degree of
       freedom and anchors the latent to the market centre. Two
       refinements on the raw weighted mean:

         a) Reputation weighting (1.00 / 0.60 / 0.30) so Tier 1 BPC
            pollsters have disproportionate say over Tier 3 novel /
            opaque ones.
         b) Trimming — the most extreme ``HE_ANCHOR_TRIM`` fraction of
            pollsters on each side (by HE value for this party) are
            dropped from the anchor calculation. This stops a single
            outlier pollster (e.g. a one-off poll with an unusually
            extreme residual) dragging the market centre for that
            party. Trimmed pollsters are NOT removed from the filter —
            they still contribute observations in the E-step; they just
            don't vote on where zero sits.

       If trimming is disabled (`HE_ANCHOR_TRIM = 0`) this reduces to a
       plain reputation-weighted mean, which itself reduces to a uniform
       mean if every pollster has weight 1.0. So the anchor is a nested
       set of refinements each safely disable-able.
    """
    date_idx = {d: i for i, d in enumerate(dates)}
    # residuals[pollster][party] = list of (weight, residual)
    residuals: dict[str, dict[str, list[tuple[float, float]]]] = {}

    last_date = dates.max() if len(dates) else pd.Timestamp.today().normalize()

    for _, row in polls.iterrows():
        d = row["fieldwork_end"]
        if d not in date_idx:
            continue
        t = date_idx[d]
        pollster = row["pollster"]
        if pollster not in residuals:
            residuals[pollster] = {p: [] for p in PARTIES}

        days_old = max((last_date - d).days, 0)
        recency_w = 0.5 ** (days_old / HE_HALFLIFE_DAYS)

        for p in PARTIES:
            val = row.get(p)
            if pd.isna(val) or val is None:
                continue
            residuals[pollster][p].append(
                (recency_w, float(val) - smoothed[p][t])
            )

    # First pass: recency-weighted mean residual per pollster/party
    house_effects: dict[str, dict[str, float]] = {}
    for pollster, party_res in residuals.items():
        house_effects[pollster] = {}
        for p in PARTIES:
            entries = party_res[p]
            if not entries:
                house_effects[pollster][p] = 0.0
                continue
            total_w = sum(w for w, _ in entries)
            if total_w <= 0:
                house_effects[pollster][p] = 0.0
                continue
            house_effects[pollster][p] = (
                sum(w * r for w, r in entries) / total_w
            )

    # Second pass: anchor house effects to a zero (reputation-weighted,
    # trimmed) mean per party across pollsters. See docstring above for
    # the full rationale; the key points are:
    #   - Reputation weighting (1.00 / 0.60 / 0.30) stops a high-volume
    #     low-reputation pollster drowning the anchor, AND stops
    #     uniform-per-pollster treating a novel non-BPC shop identically
    #     to YouGov.
    #   - Trimming the top/bottom HE_ANCHOR_TRIM fraction of pollsters
    #     by HE value removes single-poll outliers (e.g. Deltapoll on
    #     Green) from the anchor without removing them from the filter.
    # The offset we subtract gets absorbed into the next Kalman pass's
    # latent state, which is the correct place for it.
    pollster_weights = {
        ps: reputation_weight(ps) for ps in house_effects.keys()
    }
    for p in PARTIES:
        pairs = [
            (he[p], pollster_weights[ps])
            for ps, he in house_effects.items()
            if p in he
        ]
        if not pairs:
            continue
        offset = _trimmed_weighted_mean(pairs)
        if offset is None:
            continue
        for ps in house_effects:
            if p in house_effects[ps]:
                house_effects[ps][p] -= offset

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

    pollsters = sorted(state.house_effects.keys())
    summary = tier_summary(pollsters)
    logger.info(
        "Reputation weighting: %d pollsters classified", len(pollsters)
    )
    logger.info(
        "  Tier 1 (weight 1.00): %d pollsters", summary["tier_1"]
    )
    logger.info(
        "  Tier 2 (weight 0.60): %d pollsters", summary["tier_2"]
    )
    logger.info(
        "  Tier 3 (weight 0.30): %d pollsters", summary["tier_3"]
    )
    logger.info(
        "  Unclassified (default 1.00): %d pollsters", summary["unclassified"]
    )

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
