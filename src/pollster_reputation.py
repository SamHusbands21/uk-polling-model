"""
pollster_reputation.py
----------------------
Hand-curated pollster reputation tiers. Used by the aggregator to weight
poll observations and house-effect anchoring. Tier placement reflects:
  1. British Polling Council (BPC) membership.
  2. Length of published track record in GB voting intention.
  3. Transparency of methodology (do they publish tables / weighting?).

Weights deliberately sit on an aggressive 0.3 - 1.0 scale so that
low-reputation pollsters still contribute some signal but do NOT dominate
the market centre. Tier 1 pollsters (long, transparent, BPC) count ~3x
more than Tier 3 pollsters (new / opaque / single-client shops) in both
the Kalman filter's observation step and the house-effect anchor.

Single source of truth — aggregator.py and generate_outputs.py import
from here. Adding a new pollster means extending the relevant set below
(and bumping BACKLOG.md if tier placement is contentious).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier assignments
# ---------------------------------------------------------------------------
#
# BPC membership verified against the British Polling Council members
# listing and cross-referenced with Wikipedia's BPC article.
#
# Tier 1 — long track record + BPC + transparent tables/weighting
# Tier 2 — BPC (or long-established) but shorter VI history or sparse
#          methodology disclosure
# Tier 3 — non-BPC, opaque, or very new / single-client pollsters.
#          NOTE: BPC membership alone does NOT promote out of Tier 3 —
#          new BPC entrants still need a track record before Tier 2.

POLLSTER_TIER_1: set[str] = {
    "YouGov",
    "Ipsos",
    "Opinium",
    "Savanta",
    "Survation",
    "More in Common",
    "BMG Research",
    "Deltapoll",
    "Verian",
}

POLLSTER_TIER_2: set[str] = {
    "Find Out Now",
    "WeThink",
    "Focaldata",
    "JL Partners",
}

POLLSTER_TIER_3: set[str] = {
    "Omnisis",
    "Merlin Strategy",
    "Freshwater Strategy",
    "Lord Ashcroft Polls",
    # Techne is BPC-accredited but demoted to Tier 3 because their weekly
    # turnaround + sharp, persistent downward divergence from the wider
    # market on Green (~8% mean vs ~15% BPC cohort) suggests a
    # methodology that under-counts low-salience parties. The high poll
    # volume (~40% of the dataset) means even a moderate bias dominates
    # the filter, so they need both the reputation down-weight AND the
    # volume cap in _build_daily_grid to stop overwhelming the signal.
    "Techne",
}

TIER_WEIGHTS: dict[int, float] = {1: 1.00, 2: 0.60, 3: 0.30}

# Fallback for pollsters that haven't been classified yet. Set to 1.0
# (conservative) with a one-time warning so new entrants don't silently
# slip into the feed with full Tier-1 weight and nobody notices.
DEFAULT_WEIGHT: float = 1.00

_warned_unknown: set[str] = set()


def reputation_tier(pollster: str) -> Optional[int]:
    """Return the tier (1/2/3) for a pollster, or None if unclassified."""
    if pollster in POLLSTER_TIER_1:
        return 1
    if pollster in POLLSTER_TIER_2:
        return 2
    if pollster in POLLSTER_TIER_3:
        return 3
    return None


def reputation_weight(pollster: str) -> float:
    """
    Return the reputation weight used by the aggregator.

    Unclassified pollsters fall back to DEFAULT_WEIGHT and log a one-time
    warning so we notice new entrants rather than silently over-weighting
    them.
    """
    tier = reputation_tier(pollster)
    if tier is None:
        if pollster not in _warned_unknown:
            logger.warning(
                "Unclassified pollster %r — falling back to weight %.2f. "
                "Add to src/pollster_reputation.py to silence this.",
                pollster, DEFAULT_WEIGHT,
            )
            _warned_unknown.add(pollster)
        return DEFAULT_WEIGHT
    return TIER_WEIGHTS[tier]


def all_tiers() -> dict[str, int]:
    """Flat {pollster: tier} mapping, useful for JSON export."""
    out: dict[str, int] = {}
    for p in POLLSTER_TIER_1:
        out[p] = 1
    for p in POLLSTER_TIER_2:
        out[p] = 2
    for p in POLLSTER_TIER_3:
        out[p] = 3
    return out


def tier_summary(pollsters: list[str]) -> dict[str, int]:
    """
    Count how many of the supplied pollsters fall into each tier.
    Returned keys: "tier_1", "tier_2", "tier_3", "unclassified".
    """
    summary = {"tier_1": 0, "tier_2": 0, "tier_3": 0, "unclassified": 0}
    for p in pollsters:
        tier = reputation_tier(p)
        if tier is None:
            summary["unclassified"] += 1
        else:
            summary[f"tier_{tier}"] += 1
    return summary
