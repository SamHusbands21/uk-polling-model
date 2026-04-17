"""
seat_projector.py
-----------------
Monte Carlo seat projections for all 650 Westminster constituencies.

For each of N_DRAWS draws:
  1. Sample national vote shares from the Kalman posterior using a multivariate
     normal parameterised by the aggregator's covariance matrix. This preserves
     cross-party correlations (if Lab goes up, Con tends to go down). Shares
     are renormalised to sum to 1 after each draw.
  2. Scale MRP constituency shares (from mrp.py) so their electorate-weighted
     national aggregate matches the drawn national shares.
  3. Scale ML constituency shares (from ml_model.py) using the same draw.
  4. Apply FPTP: argmax(party_share) per constituency → seat count per party.
  5. Collect distributions; compute mean + 10th / 90th percentile.

Also computes the top-N most competitive marginal seats.

Run as a module:  python -m src.seat_projector
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregator import PARTIES, current_estimates, load_state

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_DRAWS = 10_000
CI_LO = 10   # 10th percentile → lower bound of 80% CI
CI_HI = 90   # 90th percentile → upper bound of 80% CI
N_MARGINALS = 20

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results_2024.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# All parties in the model (same order as aggregator)
MODEL_PARTIES = PARTIES  # ["lab", "con", "ld", "reform", "green", "snp", "pc", "others"]


# ---------------------------------------------------------------------------
# Helper: load constituency shares
# ---------------------------------------------------------------------------

def _load_shares(path: Path, parties: list[str]) -> pd.DataFrame:
    """Load constituency shares from parquet, ensuring required columns exist."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the appropriate model (mrp.py or ml_model.py) first."
        )
    df = pd.read_parquet(path)
    # Fill missing party columns with 0
    for p in parties:
        if p not in df.columns:
            df[p] = 0.0
    return df


# ---------------------------------------------------------------------------
# Sampling national shares from Kalman posterior
# ---------------------------------------------------------------------------

def _sample_national_shares(
    state,
    n_draws: int,
    parties: list[str],
) -> np.ndarray:
    """
    Draw n_draws samples from the multivariate normal distribution defined by
    the Kalman filter's current estimates and covariance matrix.

    Returns array of shape (n_draws, n_parties) with rows normalised to sum to 1.
    """
    means = np.array([state.party_states[p].mean for p in parties])  # in pp
    cov = getattr(state, "cov_matrix", None)

    if cov is None or cov.shape != (len(parties), len(parties)):
        # Fallback: diagonal covariance from individual variances
        variances = np.array([max(state.party_states[p].variance, 0.01) for p in parties])
        cov = np.diag(variances)

    # Ensure positive semi-definite by adding small ridge
    cov = cov + np.eye(len(parties)) * 0.01

    try:
        samples = np.random.multivariate_normal(means, cov, size=n_draws)
    except np.linalg.LinAlgError:
        # Fallback: independent sampling
        logger.warning("Multivariate normal failed — falling back to independent sampling")
        stds = np.sqrt(np.diag(cov))
        samples = np.random.normal(means, stds, size=(n_draws, len(parties)))

    # Clip to [0, 100] pp and renormalise to sum to 100
    samples = np.clip(samples, 0, 100)
    row_sums = samples.sum(axis=1, keepdims=True)
    samples = samples / np.maximum(row_sums, 1.0) * 100

    # Convert to fractions [0, 1]
    return samples / 100.0


# ---------------------------------------------------------------------------
# Scaling constituency shares to match a national draw
# ---------------------------------------------------------------------------

def _scale_constituency_shares(
    shares_df: pd.DataFrame,
    national_draw: np.ndarray,  # shape (n_parties,) — fractions
    parties: list[str],
    electorate: np.ndarray | None,
) -> np.ndarray:
    """
    Scale constituency-level shares so their weighted national aggregate matches
    the given national_draw. Returns a (n_constituencies × n_parties) matrix.

    Renormalises rows after scaling so shares sum to 1 per constituency.
    """
    n_seats = len(shares_df)
    n_parties = len(parties)
    shares_matrix = np.zeros((n_seats, n_parties))

    for j, p in enumerate(parties):
        if p in shares_df.columns:
            shares_matrix[:, j] = shares_df[p].fillna(0).values
        # else remains 0

    # Current weighted aggregate
    if electorate is not None and len(electorate) == n_seats:
        w = electorate / electorate.sum()
    else:
        w = np.ones(n_seats) / n_seats

    current_national = (shares_matrix * w[:, None]).sum(axis=0)  # shape (n_parties,)

    # Scale each party column
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(current_national > 0, national_draw / current_national, 1.0)

    scaled = shares_matrix * scale[None, :]
    scaled = np.clip(scaled, 0, 1)

    # Renormalise rows
    row_sums = scaled.sum(axis=1, keepdims=True)
    scaled = scaled / np.maximum(row_sums, 0.01)

    return scaled


# ---------------------------------------------------------------------------
# FPTP seat count
# ---------------------------------------------------------------------------

def _fptp_seat_counts(shares_matrix: np.ndarray, parties: list[str]) -> dict[str, int]:
    """
    Apply first-past-the-post: the party with the highest share in each
    constituency wins that seat.

    Returns dict: party → number of seats won.
    """
    winners = np.argmax(shares_matrix, axis=1)  # shape (n_constituencies,)
    counts = {p: int(np.sum(winners == j)) for j, p in enumerate(parties)}
    return counts


# ---------------------------------------------------------------------------
# Marginals computation
# ---------------------------------------------------------------------------

def _compute_marginals_from_mc(
    mean_shares: np.ndarray,
    win_prob: np.ndarray,
    codes: list[str],
    names: list[str],
    parties: list[str],
    incumbents_2024: dict[str, str] | None = None,
    n_marginals: int = N_MARGINALS,
) -> list[dict]:
    """
    Identify the most competitive seats from Monte Carlo output. A seat's
    "margin" here is the gap between the top two parties' *mean* simulated
    shares — i.e. the forecast's view of tightness.

    Using the MC draws (same data source as the map's per-seat panel) also
    avoids a latent bug in the previous implementation, where
    ``float(row.get(p, 0) or 0)`` silently coerced legitimate 0.0 shares to
    the fallback via Python's ``or`` short-circuit, and where zero-row seats
    produced a stable-sort artefact (alphabetical margin=0 rows).

    Output semantics
    ----------------
    - ``leader`` / ``runner_up`` are the top-two FORECAST parties (from the MC
      mean). These drive the tightness-of-race margin.
    - ``incumbent_2024`` is the actual 2024 winner from the HoC results — NOT
      the forecast leader. When ``leader != incumbent_2024`` that's a predicted
      flip. This was a real footgun before: the old output called the forecast
      leader "incumbent" so seats like Altrincham and Sale West (Lab held in
      2024, Con nose-ahead in the MRP this week) displayed "Con incumbent",
      which misrepresents the actual 2024 result.

    Parameters
    ----------
    mean_shares     : (n_seats, n_parties) mean simulated vote shares.
    win_prob        : (n_seats, n_parties) array of P(seat winner = party).
    codes, names    : parallel lists aligned with ``mean_shares`` row order.
    parties         : party labels aligned with ``mean_shares`` column order.
    incumbents_2024 : optional mapping of constituency_code → party that won
                      the seat in 2024. Parties outside ``parties`` (e.g. NI
                      parties bucketed as ``others``) are accepted.

    Returns
    -------
    List of dicts sorted by margin (ascending), truncated to ``n_marginals``.
    """
    n_seats, n_parties = mean_shares.shape
    if n_seats == 0 or n_parties < 2:
        return []

    incumbents_2024 = incumbents_2024 or {}

    order = np.argsort(-mean_shares, axis=1)
    first_idx = order[:, 0]
    second_idx = order[:, 1]

    seat_range = np.arange(n_seats)
    first_share = mean_shares[seat_range, first_idx]
    second_share = mean_shares[seat_range, second_idx]
    margin = first_share - second_share

    p_leader = win_prob[seat_range, first_idx]

    rows = []
    for s in range(n_seats):
        # Skip zero-information seats (all-zero shares): these show up when
        # a constituency falls outside the modelled universe (e.g. NI) and
        # would otherwise dominate the "tightest" list with meaningless 0s.
        if first_share[s] <= 0:
            continue
        code = codes[s]
        leader = parties[int(first_idx[s])]
        runner_up = parties[int(second_idx[s])]
        rows.append({
            "constituency": names[s],
            "constituency_code": code,
            "leader": leader,
            "runner_up": runner_up,
            "leader_share": round(float(first_share[s]), 4),
            "runner_up_share": round(float(second_share[s]), 4),
            "margin": round(float(margin[s]), 4),
            "swing_needed": round(float(margin[s]) / 2, 4),
            "p_win_leader": round(float(p_leader[s]), 4),
            "incumbent_2024": incumbents_2024.get(code),
            # Back-compat aliases — any downstream code still reading these
            # gets the same semantics as before (forecast top-two) rather
            # than a silent schema break.
            "incumbent": leader,
            "challenger": runner_up,
            "incumbent_share": round(float(first_share[s]), 4),
            "challenger_share": round(float(second_share[s]), 4),
        })

    rows.sort(key=lambda x: x["margin"])
    return rows[:n_marginals]


def _load_incumbents_2024() -> dict[str, str]:
    """
    Return a mapping of constituency_code → 2024 winning party, derived from
    ``data/results_2024.csv`` (the single source of truth for 2024 results).

    Works across the 6 modelled parties (lab/con/ld/reform/green/snp/pc) and
    buckets everything else as ``"others"`` — this matches the NI treatment
    where DUP / Sinn Féin / Alliance votes are lumped into ``others_2024``.
    Returns an empty dict if the file is missing (older runs shouldn't fail).
    """
    if not RESULTS_PATH.exists():
        return {}
    df = pd.read_csv(RESULTS_PATH)
    share_cols = [c for c in df.columns
                  if c.endswith("_2024") and c not in ("electorate_2024", "valid_votes_2024")]
    if not share_cols:
        return {}
    parties = [c.removesuffix("_2024") for c in share_cols]
    shares = df[share_cols].to_numpy(copy=False)
    winners_idx = shares.argmax(axis=1)
    return {
        str(code): parties[int(idx)]
        for code, idx in zip(df["constituency_code"].astype(str), winners_idx)
    }


# ---------------------------------------------------------------------------
# Main projection
# ---------------------------------------------------------------------------

def project(
    mrp_available: bool = True,
    ml_available: bool = True,
    n_draws: int = N_DRAWS,
) -> dict:
    """
    Run Monte Carlo seat projections using MRP and/or ML constituency shares.

    Returns a dict matching the seat_projections schema in polling_predictions.json:
    {
      "mrp": {party: {"mean": ..., "lo90": ..., "hi90": ...}},
      "ml":  {party: {"mean": ..., "lo90": ..., "hi90": ...}},
    }

    Also returns "raw_distributions" and "marginals" for downstream use.
    """
    state = load_state()
    estimates = current_estimates(state)

    parties = [p for p in MODEL_PARTIES if p in state.party_states]

    # Load constituency shares
    mrp_shares_df = None
    ml_shares_df = None

    mrp_path = PROCESSED_DIR / "mrp_shares.parquet"
    ml_path = PROCESSED_DIR / "ml_shares.parquet"

    if mrp_available and mrp_path.exists():
        mrp_shares_df = _load_shares(mrp_path, parties)
        logger.info("Loaded MRP shares: %d constituencies", len(mrp_shares_df))
    else:
        mrp_available = False

    if ml_available and ml_path.exists():
        ml_shares_df = _load_shares(ml_path, parties)
        logger.info("Loaded ML shares: %d constituencies", len(ml_shares_df))
    else:
        ml_available = False

    if not mrp_available and not ml_available:
        raise RuntimeError("Neither MRP nor ML constituency shares are available. Run mrp.py and/or ml_model.py first.")

    # Load electorate sizes for weighted aggregation
    electorate = None
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        shares_ref = mrp_shares_df if mrp_available else ml_shares_df
        if "electorate_2024" in results.columns:
            merged = shares_ref[["constituency_code"]].merge(
                results[["constituency_code", "electorate_2024"]], on="constituency_code", how="left"
            )
            electorate = merged["electorate_2024"].fillna(50000).values

    # Sample national shares from Kalman posterior
    logger.info("Drawing %d Monte Carlo samples …", n_draws)
    national_samples = _sample_national_shares(state, n_draws, parties)

    mrp_counts = {p: [] for p in parties} if mrp_available else {}
    ml_counts = {p: [] for p in parties} if ml_available else {}

    n_seats_total = len(mrp_shares_df) if mrp_available else len(ml_shares_df)
    n_parties = len(parties)

    # Per-seat Monte Carlo draws (float32: ~200MB per model at 10k x 650 x 8)
    # kept so we can compute constituency-level share CIs and win probabilities
    # after the loop for the interactive map.
    mrp_draws = np.zeros((n_draws, n_seats_total, n_parties), dtype=np.float32) if mrp_available else None
    ml_draws = np.zeros((n_draws, n_seats_total, n_parties), dtype=np.float32) if ml_available else None
    mrp_win_counts = np.zeros((n_seats_total, n_parties), dtype=np.int32) if mrp_available else None
    ml_win_counts = np.zeros((n_seats_total, n_parties), dtype=np.int32) if ml_available else None

    seat_idx = np.arange(n_seats_total)

    for i in range(n_draws):
        draw = national_samples[i]

        if mrp_available:
            scaled = _scale_constituency_shares(mrp_shares_df, draw, parties, electorate)
            mrp_draws[i] = scaled.astype(np.float32, copy=False)
            winners = np.argmax(scaled, axis=1)
            mrp_win_counts[seat_idx, winners] += 1
            for j, p in enumerate(parties):
                mrp_counts[p].append(int(np.sum(winners == j)))

        if ml_available:
            scaled = _scale_constituency_shares(ml_shares_df, draw, parties, electorate)
            ml_draws[i] = scaled.astype(np.float32, copy=False)
            winners = np.argmax(scaled, axis=1)
            ml_win_counts[seat_idx, winners] += 1
            for j, p in enumerate(parties):
                ml_counts[p].append(int(np.sum(winners == j)))

        if (i + 1) % 2000 == 0:
            logger.info("  %d / %d draws complete", i + 1, n_draws)

    logger.info("Monte Carlo complete. Total seats per model: %d", n_seats_total)

    # Aggregate distributions
    def _summarise(counts_dict: dict[str, list]) -> dict:
        out = {}
        for p, counts in counts_dict.items():
            arr = np.array(counts)
            out[p] = {
                "mean": round(float(np.mean(arr)), 1),
                "lo90": int(np.percentile(arr, CI_LO)),
                "hi90": int(np.percentile(arr, CI_HI)),
                "median": int(np.median(arr)),
            }
        return out

    seat_projections = {}
    if mrp_available:
        seat_projections["mrp"] = _summarise(mrp_counts)
    if ml_available:
        seat_projections["ml"] = _summarise(ml_counts)

    # Log headline seat numbers
    for model_name, proj in seat_projections.items():
        logger.info("%s seat projections:", model_name.upper())
        for p, v in sorted(proj.items(), key=lambda x: -x[1]["mean"]):
            logger.info("  %-10s %d [%d–%d]", p.upper(), v["mean"], v["lo90"], v["hi90"])

    # Build per-constituency outputs for the interactive map.
    meta_ref = mrp_shares_df if mrp_available else ml_shares_df
    codes = meta_ref["constituency_code"].astype(str).tolist()
    names = (
        meta_ref["constituency_name"].astype(str).tolist()
        if "constituency_name" in meta_ref.columns
        else codes
    )
    regions = (
        meta_ref["region_label"].astype(str).tolist()
        if "region_label" in meta_ref.columns
        else [""] * n_seats_total
    )

    # Compute marginals from the Monte Carlo mean shares — same data source
    # as the map's per-seat panel. Prefer MRP when available (richer model),
    # fall back to ML. The 2024 incumbents come from results_2024.csv so the
    # marginals table can show the actual sitting party alongside the
    # forecast leader (which may or may not be the same).
    incumbents_2024 = _load_incumbents_2024()
    if mrp_available and mrp_draws is not None:
        marginals = _compute_marginals_from_mc(
            mrp_draws.mean(axis=0),
            mrp_win_counts / max(n_draws, 1),
            codes, names, parties,
            incumbents_2024=incumbents_2024,
            n_marginals=N_MARGINALS,
        )
    elif ml_available and ml_draws is not None:
        marginals = _compute_marginals_from_mc(
            ml_draws.mean(axis=0),
            ml_win_counts / max(n_draws, 1),
            codes, names, parties,
            incumbents_2024=incumbents_2024,
            n_marginals=N_MARGINALS,
        )
    else:
        marginals = []

    def _per_seat_payloads(draws: np.ndarray, win_counts: np.ndarray) -> tuple[dict, dict]:
        lo = np.percentile(draws, CI_LO, axis=0)
        hi = np.percentile(draws, CI_HI, axis=0)
        mean = draws.mean(axis=0)
        win_prob = win_counts / max(n_draws, 1)
        prob_by_seat: dict[str, dict] = {}
        shares_by_seat: dict[str, dict] = {}
        for s in range(n_seats_total):
            code = codes[s]
            prob_by_seat[code] = {
                "name": names[s],
                "region": regions[s],
                "probs": {
                    parties[j]: round(float(win_prob[s, j]), 4)
                    for j in range(n_parties)
                    if win_prob[s, j] > 0
                },
            }
            shares_by_seat[code] = {
                "name": names[s],
                "region": regions[s],
                "shares": {
                    parties[j]: {
                        "mean": round(float(mean[s, j]), 4),
                        "lo90": round(float(lo[s, j]), 4),
                        "hi90": round(float(hi[s, j]), 4),
                    }
                    for j in range(n_parties)
                    if mean[s, j] > 0.001
                },
            }
        return prob_by_seat, shares_by_seat

    seat_probabilities: dict[str, dict] = {}
    seat_shares: dict[str, dict] = {}
    if mrp_available:
        prob, shares = _per_seat_payloads(mrp_draws, mrp_win_counts)
        seat_probabilities["mrp"] = prob
        seat_shares["mrp"] = shares
    if ml_available:
        prob, shares = _per_seat_payloads(ml_draws, ml_win_counts)
        seat_probabilities["ml"] = prob
        seat_shares["ml"] = shares

    return {
        "seat_projections": seat_projections,
        "marginals": marginals,
        "seat_probabilities": seat_probabilities,
        "seat_shares": seat_shares,
        "n_draws": n_draws,
        "raw_mrp_counts": mrp_counts,
        "raw_ml_counts": ml_counts,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    result = project()
    print("\nSeat projections:")
    for model_name, proj in result["seat_projections"].items():
        print(f"\n  [{model_name.upper()}]")
        for p, v in sorted(proj.items(), key=lambda x: -x[1]["mean"]):
            print(f"    {p.upper():10s} {v['mean']:5.0f}  [{v['lo90']}–{v['hi90']}]")

    print(f"\nTop {N_MARGINALS} marginals (tightest seats):")
    for m in result["marginals"]:
        print(f"  {m['constituency']:40s} {m['incumbent'].upper()} +{m['margin']*100:.1f}pp")
