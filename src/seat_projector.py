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

def _compute_marginals(
    mrp_shares_df: pd.DataFrame,
    parties: list[str],
    n_marginals: int = N_MARGINALS,
) -> list[dict]:
    """
    Identify the most competitive seats: those where the margin between the
    first and second party is smallest.

    Returns a list of dicts sorted by margin (ascending).
    """
    party_cols = [p for p in parties if p in mrp_shares_df.columns]
    rows = []

    for _, row in mrp_shares_df.iterrows():
        shares = {p: float(row.get(p, 0) or 0) for p in party_cols}
        if not shares:
            continue

        sorted_parties = sorted(shares.items(), key=lambda x: -x[1])
        if len(sorted_parties) < 2:
            continue

        first_party, first_share = sorted_parties[0]
        second_party, second_share = sorted_parties[1]
        margin = first_share - second_share

        rows.append({
            "constituency": row.get("constituency_name", row.get("constituency_code", "Unknown")),
            "constituency_code": row.get("constituency_code", ""),
            "incumbent": first_party,
            "challenger": second_party,
            "margin": round(margin, 4),
            "swing_needed": round(margin / 2, 4),
            "incumbent_share": round(first_share, 4),
            "challenger_share": round(second_share, 4),
        })

    rows.sort(key=lambda x: x["margin"])
    return rows[:n_marginals]


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

    # Storage for seat count distributions
    mrp_counts = {p: [] for p in parties} if mrp_available else {}
    ml_counts = {p: [] for p in parties} if ml_available else {}

    n_seats_total = len(mrp_shares_df) if mrp_available else len(ml_shares_df)

    for i in range(n_draws):
        draw = national_samples[i]  # (n_parties,)

        if mrp_available:
            scaled = _scale_constituency_shares(mrp_shares_df, draw, parties, electorate)
            counts = _fptp_seat_counts(scaled, parties)
            for p in parties:
                mrp_counts[p].append(counts.get(p, 0))

        if ml_available:
            scaled = _scale_constituency_shares(ml_shares_df, draw, parties, electorate)
            counts = _fptp_seat_counts(scaled, parties)
            for p in parties:
                ml_counts[p].append(counts.get(p, 0))

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

    # Compute marginals from the mean MRP (or ML) shares
    shares_ref_df = mrp_shares_df if mrp_available else ml_shares_df
    marginals = _compute_marginals(shares_ref_df, parties, N_MARGINALS)

    # Log headline seat numbers
    for model_name, proj in seat_projections.items():
        logger.info("%s seat projections:", model_name.upper())
        for p, v in sorted(proj.items(), key=lambda x: -x[1]["mean"]):
            logger.info("  %-10s %d [%d–%d]", p.upper(), v["mean"], v["lo90"], v["hi90"])

    return {
        "seat_projections": seat_projections,
        "marginals": marginals,
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
