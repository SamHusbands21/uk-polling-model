"""
ml_model.py
-----------
XGBoost constituency-level swing model. Trained on 2017 and 2019 election
results (swing from prior election), tested on 2024 (held out).

In prediction mode, the model takes the current national swing from the
Kalman aggregator and produces constituency-level vote shares by predicting
the differential swing each constituency experiences relative to the national.

Expected columns in results_2024.csv (HoC Library format):
  constituency_code, constituency_name, region_label,
  electorate_2024,
  lab_2024, con_2024, ld_2024, reform_2024, green_2024, snp_2024, pc_2024, others_2024,
  lab_2019, con_2019, ld_2019, brexit_2019, green_2019, snp_2019, pc_2019, others_2019,
  lab_2017, con_2017, ld_2017, ukip_2017, green_2017, snp_2017, pc_2017, others_2017,
  pct_leave, pct_graduate, median_income, urban_rural

Shares should be in [0, 1] (fractions, not percentages).

Run as a module:  python -m src.ml_model
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results_2024.csv"
MODEL_PATH = DATA_DIR / "ml_model.joblib"
PROCESSED_DIR = DATA_DIR / "processed"
ML_SHARES_PATH = PROCESSED_DIR / "ml_shares.parquet"

# ---------------------------------------------------------------------------
# Party definitions
# ---------------------------------------------------------------------------

# Parties we model (map current party name → historical equivalents)
PARTIES_2024 = ["lab", "con", "ld", "reform", "green", "snp", "pc"]

# Historical column names per election
PARTY_COLS = {
    2024: {
        "lab": "lab_2024", "con": "con_2024", "ld": "ld_2024",
        "reform": "reform_2024", "green": "green_2024",
        "snp": "snp_2024", "pc": "pc_2024",
    },
    2019: {
        "lab": "lab_2019", "con": "con_2019", "ld": "ld_2019",
        # Brexit Party contested 2019 under Reform's spiritual predecessor
        "reform": "brexit_2019", "green": "green_2019",
        "snp": "snp_2019", "pc": "pc_2019",
    },
    2017: {
        "lab": "lab_2017", "con": "con_2017", "ld": "ld_2017",
        "reform": "ukip_2017", "green": "green_2017",
        "snp": "snp_2017", "pc": "pc_2017",
    },
}

# Geographic / demographic feature columns
GEO_FEATURES = [
    "pct_leave", "pct_graduate", "median_income", "urban_rural",
    "is_scotland", "is_wales",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _swing(after: pd.Series, before: pd.Series) -> pd.Series:
    """Compute vote share swing (after minus before), in percentage points."""
    return (after.fillna(0) - before.fillna(0)) * 100


def _build_training_rows(results: pd.DataFrame, election_year: int) -> pd.DataFrame:
    """
    Build training rows for one election:
      target = swing_{party} in this election vs prior election
      features = prior election shares + geographic variables
    """
    prior_year = {2019: 2017, 2024: 2019}[election_year]
    rows = results.copy()

    feature_cols = []

    # Prior election shares as features
    for party in PARTIES_2024:
        col = PARTY_COLS[prior_year].get(party)
        if col and col in rows.columns:
            feat_name = f"prior_{party}"
            rows[feat_name] = rows[col].fillna(0)
            feature_cols.append(feat_name)

    # Geographic features
    for col in GEO_FEATURES:
        if col in rows.columns:
            feature_cols.append(col)
        else:
            rows[col] = 0
            feature_cols.append(col)

    # Region dummies
    if "region_label" in rows.columns:
        rows["region_code"] = rows["region_label"].astype("category").cat.codes
        feature_cols.append("region_code")

    # Incumbent majority (seat vulnerability)
    if f"majority_{prior_year}" in rows.columns:
        rows["prior_majority"] = rows[f"majority_{prior_year}"]
        feature_cols.append("prior_majority")

    # National swing (same for all rows in this training set — will be
    # the key predictor in prediction mode)
    for party in PARTIES_2024:
        curr_col = PARTY_COLS[election_year].get(party)
        prior_col = PARTY_COLS[prior_year].get(party)
        if curr_col and prior_col and curr_col in rows.columns and prior_col in rows.columns:
            national_swing = _swing(rows[curr_col], rows[prior_col]).mean()
            rows[f"national_swing_{party}"] = national_swing
            feature_cols.append(f"national_swing_{party}")

    # Target: swing for each party (one model per party)
    for party in PARTIES_2024:
        curr_col = PARTY_COLS[election_year].get(party)
        prior_col = PARTY_COLS[prior_year].get(party)
        if curr_col and prior_col and curr_col in rows.columns and prior_col in rows.columns:
            rows[f"target_{party}"] = _swing(rows[curr_col], rows[prior_col])

    rows["election_year"] = election_year
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate, preserve order
    return rows, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(force: bool = False) -> dict:
    """
    Train one XGBoost model per party on 2017 and 2019 data.
    Evaluate on 2024 (held-out test). Save to data/ml_model.joblib.

    Returns: dict with 'models', 'feature_cols', 'evaluation'.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required. Install with: pip install xgboost")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and not force:
        logger.info("ML model already exists at %s — loading. Use force=True to retrain.", MODEL_PATH)
        return joblib.load(MODEL_PATH)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. "
            "Download the HoC Library constituency results file and save as data/results_2024.csv."
        )

    results = pd.read_csv(RESULTS_PATH)
    logger.info("Loaded results: %d constituencies", len(results))

    # Add is_scotland, is_wales flags
    results["is_scotland"] = results["constituency_code"].str.startswith("S").astype(int)
    results["is_wales"] = results["constituency_code"].str.startswith("W").astype(int)

    # Build training sets for 2019 and 2024 (but keep 2024 for test only)
    train_rows_2019, feat_cols_2019 = _build_training_rows(results, 2019)

    # 2024 is held-out test
    test_rows, feat_cols_test = _build_training_rows(results, 2024)

    # Union of feature columns
    all_feat_cols = list(dict.fromkeys(feat_cols_2019 + feat_cols_test))

    # Ensure all feature cols exist in both
    for col in all_feat_cols:
        for df in [train_rows_2019, test_rows]:
            if col not in df.columns:
                df[col] = 0.0

    train_df = pd.concat([train_rows_2019], ignore_index=True)

    models = {}
    evaluation = {}

    for party in PARTIES_2024:
        target_col = f"target_{party}"
        if target_col not in train_df.columns:
            logger.warning("No target column for %s — skipping", party)
            continue
        if target_col not in test_rows.columns:
            continue

        # Filter out rows where prior or current share is fully missing
        train_valid = train_df[train_df[target_col].notna()].copy()
        test_valid = test_rows[test_rows[target_col].notna()].copy()

        if len(train_valid) < 50:
            logger.warning("Party %s: only %d training rows — skipping", party, len(train_valid))
            continue

        X_train = train_valid[all_feat_cols].fillna(0)
        y_train = train_valid[target_col]
        X_test = test_valid[all_feat_cols].fillna(0)
        y_test = test_valid[target_col]

        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.5,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)

        # Evaluate on 2024 holdout
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_test)))
        logger.info("  %s: RMSE=%.2f pp, MAE=%.2f pp (n=%d test)", party, rmse, mae, len(test_valid))

        models[party] = model
        evaluation[party] = {
            "rmse_pp": round(rmse, 3),
            "mae_pp": round(mae, 3),
            "n_test": len(test_valid),
            "n_train": len(train_valid),
        }

    payload = {
        "models": models,
        "feature_cols": all_feat_cols,
        "evaluation": evaluation,
    }
    joblib.dump(payload, MODEL_PATH)
    logger.info("Saved ML model bundle to %s", MODEL_PATH)

    return payload


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_constituency_shares(
    national_estimates: dict[str, dict],
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Apply the trained XGBoost swing models to the current national estimates
    to produce constituency-level vote share predictions.

    national_estimates: {party: {mean, lo90, hi90}} — output from aggregator.current_estimates()

    Returns DataFrame with constituency_code + one column per party (fractions [0,1]).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    payload = train(force=force_retrain)
    models = payload["models"]
    feature_cols = payload["feature_cols"]

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found.")

    results = pd.read_csv(RESULTS_PATH)
    results["is_scotland"] = results["constituency_code"].str.startswith("S").astype(int)
    results["is_wales"] = results["constituency_code"].str.startswith("W").astype(int)

    if "region_label" not in results.columns:
        results["region_label"] = "england"
        results.loc[results["is_scotland"] == 1, "region_label"] = "scotland"
        results.loc[results["is_wales"] == 1, "region_label"] = "wales"
    results["region_code"] = results["region_label"].astype("category").cat.codes

    # Build prediction frame using 2024 shares as the baseline
    pred_frame = results[["constituency_code", "constituency_name", "region_label"]].copy()

    # Prior shares (2024 actuals as baseline)
    for party in PARTIES_2024:
        col = PARTY_COLS[2024].get(party)
        pred_frame[f"prior_{party}"] = results[col].fillna(0) if col and col in results.columns else 0.0

    # National swing from Kalman vs 2024 actuals
    # We compute 2024 national share as the mean across constituencies
    for party in PARTIES_2024:
        col_2024 = PARTY_COLS[2024].get(party)
        if col_2024 and col_2024 in results.columns:
            national_2024 = results[col_2024].fillna(0).mean() * 100  # to pp
        else:
            national_2024 = 0.0
        current_mean = national_estimates.get(party, {}).get("mean", national_2024 / 100) * 100
        national_swing = current_mean - national_2024
        pred_frame[f"national_swing_{party}"] = national_swing

    # Geographic features
    for col in GEO_FEATURES + ["region_code"]:
        if col in results.columns:
            pred_frame[col] = results[col]
        else:
            pred_frame[col] = 0

    # Ensure all feature columns present
    for col in feature_cols:
        if col not in pred_frame.columns:
            pred_frame[col] = 0.0

    # Predict swing for each party, add to 2024 baseline
    shares_df = pred_frame[["constituency_code", "constituency_name", "region_label"]].copy()
    for party in PARTIES_2024:
        model = models.get(party)
        col_2024 = PARTY_COLS[2024].get(party)
        baseline = results[col_2024].fillna(0).values * 100 if col_2024 and col_2024 in results.columns else np.zeros(len(results))

        if model is not None:
            X = pred_frame[feature_cols].fillna(0)
            predicted_swing = model.predict(X)
        else:
            # Fallback: apply national swing uniformly
            predicted_swing = np.full(len(pred_frame), pred_frame[f"national_swing_{party}"].iloc[0])

        new_share = (baseline + predicted_swing) / 100.0
        shares_df[party] = np.clip(new_share, 0, 1)

    # Renormalise rows
    party_cols = [p for p in PARTIES_2024 if p in shares_df.columns]
    row_sums = shares_df[party_cols].sum(axis=1)
    for p in party_cols:
        shares_df[p] = shares_df[p] / row_sums.clip(0.01)

    shares_df.to_parquet(ML_SHARES_PATH, index=False)
    logger.info("Saved ML constituency shares for %d seats to %s", len(shares_df), ML_SHARES_PATH)

    return shares_df


def load_ml_shares() -> pd.DataFrame:
    """Load the cached ML constituency shares."""
    if not ML_SHARES_PATH.exists():
        raise FileNotFoundError(
            f"{ML_SHARES_PATH} not found. Run predict_constituency_shares() first."
        )
    return pd.read_parquet(ML_SHARES_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Training ML swing models …")
    payload = train(force=True)
    print("\nEvaluation on 2024 holdout:")
    for party, metrics in payload["evaluation"].items():
        print(f"  {party.upper():8s} RMSE={metrics['rmse_pp']:.2f}pp  MAE={metrics['mae_pp']:.2f}pp  (n={metrics['n_test']})")

    from src.aggregator import current_estimates, load_state
    state = load_state()
    estimates = current_estimates(state)
    shares = predict_constituency_shares(estimates)
    print(f"\nML constituency shares: {len(shares)} seats")
    print(shares[["constituency_name"] + PARTIES_2024[:4]].head(10).to_string(index=False))
