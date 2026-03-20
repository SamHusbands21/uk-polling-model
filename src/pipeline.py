"""
pipeline.py
-----------
Main pipeline orchestrator. Runs the full sequence:
  1. Scrape polls
  2. Run Kalman aggregator
  3. (If BES + census available) Run MRP poststratification
  4. (If results file available) Run ML swing model
  5. (If constituency shares available) Run Monte Carlo seat projections
  6. Generate all outputs

Designed to be idempotent — skips steps that have already run today,
respects cached states, and only reruns expensive steps when inputs change.

Run as a module:  python -m src.pipeline [--full]
  --full: force refit of all models from scratch (use monthly)
  default: incremental update only (use weekly / daily)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def run(full: bool = False) -> None:
    from src.scraper import scrape
    from src.aggregator import aggregate, current_estimates, load_state
    from src.generate_outputs import generate

    # ── Step 1: Scrape ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Scraping polls …")
    df = scrape(force_refresh=full)
    logger.info("  %d polls loaded.", len(df))

    # ── Step 2: Aggregator ───────────────────────────────────────────────────
    logger.info("Step 2: Running Kalman aggregator …")
    state = aggregate(force_full=full)
    estimates = current_estimates(state)

    # ── Step 3: MRP poststratification (optional — requires BES) ─────────────
    mrp_available = False
    bes_path = DATA_DIR / "bes_2024.csv"
    census_path = DATA_DIR / "census_2021.csv"
    coeff_path = DATA_DIR / "mrp_coefficients.pkl"
    mrp_shares_path = PROCESSED_DIR / "mrp_shares.parquet"

    if bes_path.exists() and census_path.exists():
        logger.info("Step 3: Running MRP poststratification …")
        try:
            from src.mrp import poststratify
            poststratify(estimates, force_refit=(full and coeff_path.exists()))
            mrp_available = True
        except Exception as exc:
            logger.warning("MRP failed: %s — continuing without seat projections.", exc)
    else:
        logger.info("Step 3: Skipped — BES data or census not yet available.")
        if mrp_shares_path.exists():
            mrp_available = True
            logger.info("  Using cached MRP shares from previous run.")

    # ── Step 4: ML swing model (optional — requires results_2024.csv) ────────
    ml_available = False
    results_path = DATA_DIR / "results_2024.csv"
    ml_model_path = DATA_DIR / "ml_model.joblib"
    ml_shares_path = PROCESSED_DIR / "ml_shares.parquet"

    if results_path.exists():
        logger.info("Step 4: Running ML swing model …")
        try:
            from src.ml_model import predict_constituency_shares
            predict_constituency_shares(estimates, force_retrain=(full and ml_model_path.exists()))
            ml_available = True
        except Exception as exc:
            logger.warning("ML model failed: %s — continuing without.", exc)
    else:
        logger.info("Step 4: Skipped — results_2024.csv not yet available.")
        if ml_shares_path.exists():
            ml_available = True
            logger.info("  Using cached ML shares from previous run.")

    # ── Step 5: Monte Carlo seat projections ─────────────────────────────────
    seat_projections = {}
    marginals: list[dict] = []
    seat_history_entry: dict | None = None

    if mrp_available or ml_available:
        logger.info("Step 5: Running Monte Carlo seat projections …")
        try:
            from src.seat_projector import project
            proj_result = project(mrp_available=mrp_available, ml_available=ml_available)
            seat_projections = proj_result["seat_projections"]
            marginals = proj_result["marginals"]

            # Build seat history entry (weekly, keyed by ISO week)
            iso_week = date.today().isocalendar()
            week_key = f"{iso_week.year}-W{iso_week.week:02d}"
            mrp_means = {p: v["mean"] for p, v in seat_projections.get("mrp", {}).items()}
            ml_means = {p: v["mean"] for p, v in seat_projections.get("ml", {}).items()}
            seat_history_entry = {
                "week": week_key,
                "mrp": mrp_means,
                "ml": ml_means,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.warning("Seat projector failed: %s — skipping projections.", exc)
    else:
        logger.info("Step 5: Skipped — no constituency-level shares available.")

    # ── Step 6: Generate outputs ──────────────────────────────────────────────
    logger.info("Step 6: Generating outputs …")
    payload = generate(
        seat_projections=seat_projections or None,
        marginals=marginals or None,
        seat_history_entry=seat_history_entry,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("  Polls used: %d", payload["polls_used"])
    logger.info("  Last poll: %s", payload["last_poll_date"])
    if payload["seat_projections"]:
        for model_name, proj in payload["seat_projections"].items():
            lab = proj.get("lab", {})
            con = proj.get("con", {})
            logger.info(
                "  %s: Lab %d [%d–%d], Con %d [%d–%d]",
                model_name.upper(),
                lab.get("mean", 0), lab.get("lo90", 0), lab.get("hi90", 0),
                con.get("mean", 0), con.get("lo90", 0), con.get("hi90", 0),
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="UK Polling Model pipeline")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full refit from scratch (use monthly; default is incremental update)",
    )
    args = parser.parse_args()

    run(full=args.full)
