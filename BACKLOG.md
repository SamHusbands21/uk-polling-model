# Backlog

Deferred work items for the UK polling model. Most recent first.

---

## Pollster reputation weighting

**Status:** DONE — shipped with hand-curated three-tier weighting
(1.00 / 0.60 / 0.30). Implemented as `src/pollster_reputation.py` with
tiers exported via `data/polling_predictions.json → pollster_reputation`
and rendered on the site under "How It Works". Applied in both
`_build_daily_grid` (per-poll weight) and `_estimate_house_effects`
(reputation-weighted anchoring). Smoke test moved Green national from
~13.1% to ~13.2% and preserved YouGov's correct +Green house effect.

**Why:** today's anchoring change made every pollster one vote in defining
the "market centre". That's a principled default but it treats a transparent
BPC-member pollster with a long, published track record (YouGov, Ipsos,
Opinium) identically to a pollster with a short or opaque history (Omnisis,
Merlin Strategy, Freshwater). We already know Omnisis is an outlier (Green
HE +20 in recent runs) and that their polls drag the anchor with equal
weight to YouGov's. Reputation weighting would correctly down-weight the
noisy/opaque end of the market.

**Scope sketch:**

1. Build a `POLLSTER_REPUTATION: dict[str, float]` table in
   [src/aggregator.py](src/aggregator.py) (or a small
   `src/pollster_reputation.py`), with weights on e.g. a 0.25-1.0 scale.
   Inputs to the score (pick a couple, not all):
   - BPC membership (yes / no).
   - Historical mean absolute error vs the 2024 GE result (use
     `data/results_2024.csv` as truth; can be computed one-off from the
     final pre-election polls).
   - Transparency: do they publish tables? methodology?
   - Total polls contributed to the dataset (light reward for frequency,
     with sqrt decay so weekly pollsters don't dominate again).
2. Use the weight in two places:
   - `_estimate_house_effects`: replace the uniform zero-mean anchor with
     a reputation-weighted anchor. Each pollster's contribution to the
     offset numerator and denominator gets multiplied by their reputation
     weight. This is the main lever.
   - `_build_daily_grid`: multiply each poll's existing
     `sample_weight * recency_weight` by the pollster's reputation
     weight. Secondary — lets the filter trust reputable pollsters more
     directly in the E-step.
3. Keep the default weight = 1.0 for any pollster not in the table
   (conservative: no change for unknown pollsters).
4. Log the weighted and unweighted Green latent side-by-side in the
   smoke test so we can see the effect.

**Expected impact:** Green latent moves up toward YouGov's signal (~17%)
when YouGov gets weight 1.0 and Omnisis gets 0.3. Reform / Lab / Con also
become marginally less sensitive to the single-pollster shocks that
currently move them by a few tenths of a point.

**Open questions to resolve before implementing:**

- Where does the reputation data come from? Start with a hand-curated dict
  based on BPC membership + 2024 accuracy. Can be upgraded later to a
  learned weight from cross-validation, but don't go there first.
- Should the weight be per-party (e.g. Opinium might be reliable for
  Con/Lab but weak for Green specifically) or per-pollster? Start
  per-pollster; per-party is more powerful but twice the calibration.
- Risk: reputation weighting is a judgement call and can look politically
  motivated. Write the methodology note on the website at the same time,
  and publish the table — no hidden weights.

**Files likely to change:**

- [src/aggregator.py](src/aggregator.py) — new `POLLSTER_REPUTATION` dict,
  update `_build_daily_grid` and `_estimate_house_effects`.
- [polling.html](../sam_website/polling.html) (in sam_website) —
  methodology note explaining the weights are used and where to see them.
