"""
generate_outputs.py
-------------------
Writes all website artefacts to output/:
  - polling_predictions.json  (the single source of truth, includes history)
  - vote_share_trend.png      (smoothed poll tracker with CI ribbons)
  - seat_distribution.png     (Monte Carlo seat histograms — Phase 2)
  - model_comparison.png      (MRP vs ML bar chart — Phase 2)

The JSON file accumulates history across runs. Each run:
  - Appends to vote_share_history (daily, deduped by date)
  - Appends to seat_history (weekly, deduped by ISO week — Phase 2)

Run as a module:  python -m src.generate_outputs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.aggregator import PARTIES, current_estimates, load_state
from src.scraper import load as load_polls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data"
JSON_OUT = OUTPUT_DIR / "polling_predictions.json"

# ---------------------------------------------------------------------------
# Party display config
# ---------------------------------------------------------------------------

PARTY_CONFIG: dict[str, dict] = {
    "lab":    {"label": "Labour",    "color": "#E4003B"},
    "con":    {"label": "Con",       "color": "#0087DC"},
    "reform": {"label": "Reform",    "color": "#12B6CF"},
    "ld":     {"label": "Lib Dems",  "color": "#FAA61A"},
    "green":  {"label": "Greens",    "color": "#02A95B"},
    "snp":    {"label": "SNP",       "color": "#FDF38E"},
    "pc":     {"label": "Plaid",     "color": "#3F8428"},
    "others": {"label": "Others",    "color": "#9CA3AF"},
}

# Parties to include in the trend chart (enough data to be meaningful)
TREND_PARTIES = ["lab", "con", "reform", "ld", "green", "snp"]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _load_existing_json() -> dict:
    """Load the existing predictions JSON, or return a fresh skeleton."""
    if JSON_OUT.exists():
        try:
            with open(JSON_OUT, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse existing JSON; starting fresh.")
    return {
        "updated_at": "",
        "polls_used": 0,
        "last_poll_date": "",
        "national_shares": {},
        "seat_projections": {},
        "house_effects": {},
        "marginals": [],
        "vote_share_history": [],
        "seat_history": [],
    }


def _merge_vote_share_history(
    existing: list[dict],
    new_history: list[dict],
) -> list[dict]:
    """
    Merge new daily entries into the existing history, deduped by date.
    Existing entries for the same date are overwritten with the new values.
    """
    by_date = {e["date"]: e for e in existing}
    for entry in new_history:
        by_date[entry["date"]] = entry
    merged = sorted(by_date.values(), key=lambda x: x["date"])
    return merged


def _append_seat_history(
    existing: list[dict],
    new_entry: dict | None,
) -> list[dict]:
    """
    Append a seat history entry, deduped by ISO week.
    new_entry: {"week": "2026-W12", "mrp": {...}, "ml": {...}}
    """
    if new_entry is None:
        return existing
    by_week = {e["week"]: e for e in existing}
    by_week[new_entry["week"]] = new_entry
    return sorted(by_week.values(), key=lambda x: x["week"])


# ---------------------------------------------------------------------------
# Vote-share trend chart
# ---------------------------------------------------------------------------

def _plot_vote_share_trend(smoothed_df: pd.DataFrame, out_path: Path) -> None:
    """
    Render the smoothed vote-share time series with 90% CI ribbons.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")

    dates = pd.to_datetime(smoothed_df["date"])

    legend_handles = []
    for p in TREND_PARTIES:
        if p not in smoothed_df.columns:
            continue
        cfg = PARTY_CONFIG.get(p, {"label": p.upper(), "color": "#888"})
        color = cfg["color"]
        label = cfg["label"]

        means = smoothed_df[p].values * 100
        lo_col = f"{p}_lo90"
        hi_col = f"{p}_hi90"

        ax.plot(dates, means, color=color, linewidth=2, label=label)

        if lo_col in smoothed_df.columns and hi_col in smoothed_df.columns:
            lo = smoothed_df[lo_col].values * 100
            hi = smoothed_df[hi_col].values * 100
            ax.fill_between(dates, lo, hi, color=color, alpha=0.15)

        patch = mpatches.Patch(color=color, label=label)
        legend_handles.append(patch)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Vote share (%)", fontsize=10)
    ax.set_title(
        "UK National Vote Share — Poll Aggregator",
        fontsize=13, color="#e2e8f0", pad=14, fontweight="bold",
    )
    ax.grid(True, color="#334155", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_ylim(0, None)

    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        framealpha=0.3,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        fontsize=9,
    )

    # Add a subtle "election" reference line if we have data spanning July 2024
    election_date = pd.Timestamp("2024-07-04")
    if dates.min() <= election_date <= dates.max():
        ax.axvline(election_date, color="#475569", linestyle=":", linewidth=1)
        ax.text(
            election_date, ax.get_ylim()[1] * 0.97,
            " 2024 GE",
            color="#64748b", fontsize=8, va="top",
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved vote share trend chart → %s", out_path)


# ---------------------------------------------------------------------------
# Seat distribution chart (stub — populated in Phase 2)
# ---------------------------------------------------------------------------

def _plot_seat_distribution(seat_projections: dict, out_path: Path) -> None:
    """
    Render overlapping histograms of Monte Carlo seat counts.
    This function is a stub until seat_projector is built.
    """
    if not seat_projections:
        logger.info("No seat projections yet — skipping seat_distribution.png")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8")

    # Use MRP projections if available
    mrp = seat_projections.get("mrp", {})
    ml = seat_projections.get("ml", {})

    parties_to_plot = [p for p in ["lab", "con", "reform", "ld"] if p in mrp]
    x = np.arange(len(parties_to_plot))
    width = 0.35

    mrp_means = [mrp[p]["mean"] for p in parties_to_plot]
    ml_means = [ml.get(p, {}).get("mean", 0) for p in parties_to_plot]

    bars1 = ax.bar(x - width / 2, mrp_means, width, label="MRP",
                   color=[PARTY_CONFIG[p]["color"] for p in parties_to_plot], alpha=0.85)
    bars2 = ax.bar(x + width / 2, ml_means, width, label="ML",
                   color=[PARTY_CONFIG[p]["color"] for p in parties_to_plot], alpha=0.45)

    ax.axhline(326, color="#e2e8f0", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(parties_to_plot) - 0.1, 327, "Majority", color="#94a3b8", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([PARTY_CONFIG[p]["label"] for p in parties_to_plot], color="#e2e8f0")
    ax.set_ylabel("Seats", color="#94a3b8")
    ax.set_title("Seat Projections — MRP vs ML", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved seat distribution chart → %s", out_path)


def _plot_model_comparison(seat_projections: dict, out_path: Path) -> None:
    """
    Render MRP vs ML seat predictions with CI bars.
    Stub until seat projector is built.
    """
    if not seat_projections:
        logger.info("No seat projections yet — skipping model_comparison.png")
        return

    mrp = seat_projections.get("mrp", {})
    ml = seat_projections.get("ml", {})
    parties = [p for p in ["lab", "con", "reform", "ld", "green", "snp"] if p in mrp]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor("#0f172a")

    for ax, model_data, title in zip(axes, [mrp, ml], ["MRP Model", "ML Model"]):
        ax.set_facecolor("#1e293b")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94a3b8")

        means = [model_data.get(p, {}).get("mean", 0) for p in parties]
        lo90 = [model_data.get(p, {}).get("lo90", m) for p, m in zip(parties, means)]
        hi90 = [model_data.get(p, {}).get("hi90", m) for p, m in zip(parties, means)]
        colors = [PARTY_CONFIG[p]["color"] for p in parties]
        labels = [PARTY_CONFIG[p]["label"] for p in parties]

        yerr_lo = [max(0, m - l) for m, l in zip(means, lo90)]
        yerr_hi = [max(0, h - m) for h, m in zip(hi90, means)]

        ax.bar(labels, means, color=colors, alpha=0.85,
               yerr=[yerr_lo, yerr_hi], capsize=4,
               error_kw={"ecolor": "#94a3b8", "elinewidth": 1})
        ax.axhline(326, color="#e2e8f0", linestyle="--", linewidth=1, alpha=0.4)
        ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold")
        ax.set_ylabel("Seats", color="#94a3b8")

    fig.suptitle("Seat Projection Comparison", color="#e2e8f0", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved model comparison chart → %s", out_path)


# ---------------------------------------------------------------------------
# Main output generator
# ---------------------------------------------------------------------------

def generate(
    seat_projections: dict | None = None,
    marginals: list[dict] | None = None,
    seat_history_entry: dict | None = None,
) -> dict:
    """
    Generate all output artefacts. Returns the final JSON payload.

    seat_projections: output from seat_projector.project() — None in Phase 1
    marginals: list of top marginal seats — None in Phase 1
    seat_history_entry: {"week": "2026-W12", "mrp": {...}, "ml": {...}} — None in Phase 1
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load aggregator state and polls
    state = load_state()
    polls = load_polls()
    estimates = current_estimates(state)

    # Load smoothed daily series
    smoothed_path = DATA_DIR / "processed" / "smoothed_shares.parquet"
    smoothed_df = pd.read_parquet(smoothed_path) if smoothed_path.exists() else pd.DataFrame()

    # Build national_shares block for JSON
    national_shares = {}
    for p in PARTIES:
        est = estimates.get(p, {})
        cfg = PARTY_CONFIG.get(p, {})
        if est.get("mean", 0) > 0.001:  # skip parties with effectively 0 support
            national_shares[p] = {
                "mean": est["mean"],
                "lo90": max(0.0, est["lo90"]),
                "hi90": min(1.0, est["hi90"]),
                "label": cfg.get("label", p.upper()),
                "color": cfg.get("color", "#888"),
            }

    # Build vote_share_history from the smoothed daily series
    new_history: list[dict] = []
    if not smoothed_df.empty:
        for _, row in smoothed_df.iterrows():
            entry: dict = {"date": row["date"]}
            for p in PARTIES:
                if p in row and row[p] is not None and not pd.isna(row[p]):
                    entry[p] = round(float(row[p]), 4)
            new_history.append(entry)

    # Build house_effects block — top-biased pollsters only
    house_effects_clean: dict[str, dict[str, float]] = {}
    for pollster, effects in state.house_effects.items():
        significant = {
            p: round(v, 2)
            for p, v in effects.items()
            if abs(v) > 0.5  # only show effects > 0.5pp
        }
        if significant:
            house_effects_clean[pollster] = significant

    # Load and merge existing JSON to preserve history
    existing = _load_existing_json()
    merged_history = _merge_vote_share_history(
        existing.get("vote_share_history", []),
        new_history,
    )
    merged_seat_history = _append_seat_history(
        existing.get("seat_history", []),
        seat_history_entry,
    )

    # Assemble final payload
    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "polls_used": len(polls),
        "last_poll_date": state.last_date,
        "national_shares": national_shares,
        "seat_projections": seat_projections or {},
        "house_effects": house_effects_clean,
        "marginals": marginals or [],
        "vote_share_history": merged_history,
        "seat_history": merged_seat_history,
    }

    # Write JSON
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Wrote polling_predictions.json (%d history entries)", len(merged_history))

    # Generate charts
    if not smoothed_df.empty:
        _plot_vote_share_trend(smoothed_df, OUTPUT_DIR / "vote_share_trend.png")

    if seat_projections:
        _plot_seat_distribution(seat_projections, OUTPUT_DIR / "seat_distribution.png")
        _plot_model_comparison(seat_projections, OUTPUT_DIR / "model_comparison.png")

    return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    payload = generate()
    shares = payload["national_shares"]
    print("\nCurrent estimates written to output/polling_predictions.json")
    for p, v in sorted(shares.items(), key=lambda x: -x[1]["mean"]):
        print(f"  {v['label']:12s} {v['mean']*100:.1f}%  [{v['lo90']*100:.1f}–{v['hi90']*100:.1f}%]")
