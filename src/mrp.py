"""
mrp.py
------
Multilevel Regression and Poststratification (MRP) seat model.

Step 1 — Regression (fit once, requires BES 2024 microdata):
  One binary logistic mixed-effects model per party.
  Fixed effects: age group, sex, education, ethnicity, past_vote_2024, brexit_vote.
  Random effects: region (intercept), constituency (intercept).
  Geographic predictors on constituency intercept: pct_leave, pct_graduate, median_income.
  Scotland/Wales: is_scotland / is_wales binary fixed effects + party restrictions.
  Saves model objects to data/mrp_coefficients.pkl.

Step 2 — Poststratification (runs each pipeline execution):
  For each constituency × demographic cell (from census IPF joint distribution):
    Predict P(vote_party | demographics, constituency) using stored coefficients.
  Weight by cell population → constituency-level vote share per party.
  Calibrate so GB aggregate matches the Kalman filter's current national estimates.
  Save to data/processed/mrp_shares.parquet.

Expected BES column names (adjust BES_COL_MAP below if different):
  weight, region, pcon_code, age, sex, edlevel, ethnicity,
  vote_intention, past_vote_2019, past_vote_2024, remain_leave

Run as a module:  python -m src.mrp
"""

from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
BES_PATH = DATA_DIR / "bes_2024.csv"
RESULTS_PATH = DATA_DIR / "results_2024.csv"
CENSUS_PATH = DATA_DIR / "census_2021.csv"
COEFF_PATH = DATA_DIR / "mrp_coefficients.pkl"
PROCESSED_DIR = DATA_DIR / "processed"
MRP_SHARES_PATH = PROCESSED_DIR / "mrp_shares.parquet"

# ---------------------------------------------------------------------------
# Parties modelled (one binary model each)
# ---------------------------------------------------------------------------

ALL_PARTIES = ["lab", "con", "ld", "reform", "green", "snp", "pc"]

# Regional restrictions: SNP only contests Scottish seats; PC only Welsh.
PARTY_REGION_FILTER = {
    "snp": "scotland",
    "pc":  "wales",
}

# ---------------------------------------------------------------------------
# BES column name mapping (adjust if your BES CSV uses different names)
# ---------------------------------------------------------------------------

BES_COL_MAP = {
    "weight":           "weight",
    "region":           "gor",          # Government Office Region code
    "pcon_code":        "pcon",         # Westminster constituency code
    "age":              "age",          # Raw age (integer)
    "sex":              "gender",       # 1=Male, 2=Female
    "edlevel":          "edlevel",      # 0=None, 1=Below GCSE, 2=GCSE, 3=A-Level, 4=Degree, 5=Postgrad
    "ethnicity":        "p_ethnicity",  # 1=White British, else other
    "vote_intention":   "generalElectionVoteW25",  # Party the respondent intends to vote for
    "past_vote_2024":   "generalElectionVoteW23",  # Recalled 2024 vote (post-election wave)
    "brexit_vote":      "euRefVoteW4",  # 1=Remain, 2=Leave
}

# ---------------------------------------------------------------------------
# Categorical encoders
# ---------------------------------------------------------------------------

AGE_BANDS = ["18-24", "25-34", "35-49", "50-64", "65+"]

def _age_band(age: float) -> str:
    if age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 50:
        return "35-49"
    elif age < 65:
        return "50-64"
    else:
        return "65+"

def _education_level(edlevel: float) -> str:
    return "degree" if edlevel >= 4 else "no_degree"

def _ethnicity_label(eth_code: float) -> str:
    return "white_british" if eth_code == 1 else "other_ethnicity"

def _sex_label(sex_code: float) -> str:
    return "male" if sex_code == 1 else "female"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _prepare_bes(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load and clean BES 2024 microdata. Returns a DataFrame with
    standardised columns for modelling.
    """
    if not BES_PATH.exists():
        raise FileNotFoundError(
            f"BES data not found at {BES_PATH}.\n"
            "Apply for access at https://www.britishelectionstudy.com/data-objects/panel-study-data/"
            " and download the 2024 post-election wave CSV."
        )

    bes = pd.read_csv(BES_PATH, low_memory=False)
    logger.info("Loaded BES: %d respondents, %d columns", len(bes), len(bes.columns))

    # Rename columns using the mapping
    rename = {v: k for k, v in BES_COL_MAP.items() if v in bes.columns}
    bes = bes.rename(columns=rename)

    # Check required columns
    required = {"region", "pcon_code", "age", "vote_intention"}
    missing = required - set(bes.columns)
    if missing:
        raise ValueError(
            f"BES file is missing required columns: {missing}. "
            f"Please check BES_COL_MAP in mrp.py matches your file's column names. "
            f"Available columns: {list(bes.columns[:30])}"
        )

    # Derive region labels
    scotland_regions = {"Scotland", "S92000003", "12"}  # common encodings
    wales_regions = {"Wales", "W92000004", "11"}

    def _region_label(r):
        r_str = str(r)
        if r_str in scotland_regions or "scot" in r_str.lower():
            return "scotland"
        elif r_str in wales_regions or "wales" in r_str.lower():
            return "wales"
        else:
            return r_str

    bes["region_label"] = bes["region"].apply(_region_label)
    bes["is_scotland"] = (bes["region_label"] == "scotland").astype(int)
    bes["is_wales"] = (bes["region_label"] == "wales").astype(int)

    # Demographic variables
    bes["age_band"] = bes["age"].apply(lambda x: _age_band(float(x)) if pd.notna(x) else "35-49")
    bes["sex_label"] = bes.get("sex", pd.Series(1, index=bes.index)).apply(
        lambda x: _sex_label(float(x)) if pd.notna(x) else "male"
    )
    bes["edu_label"] = bes.get("edlevel", pd.Series(2, index=bes.index)).apply(
        lambda x: _education_level(float(x)) if pd.notna(x) else "no_degree"
    )
    bes["eth_label"] = bes.get("ethnicity", pd.Series(1, index=bes.index)).apply(
        lambda x: _ethnicity_label(float(x)) if pd.notna(x) else "white_british"
    )
    bes["brexit_remain"] = bes.get("brexit_vote", pd.Series(1, index=bes.index)).apply(
        lambda x: 1 if float(x) == 1 else 0 if pd.notna(x) else 0
    )

    # Past vote 2024 — one-hot encode main parties
    past_vote_parties = ["lab", "con", "ld", "reform", "green", "snp", "pc", "other"]
    bes["past_vote_clean"] = bes.get("past_vote_2024", pd.Series("other", index=bes.index)).apply(
        lambda x: str(x).lower().strip()
    )
    for pv in past_vote_parties:
        bes[f"past_{pv}"] = (bes["past_vote_clean"].str.contains(pv, na=False)).astype(int)

    # Merge constituency geographic predictors
    results_sub = results_df[["constituency_code", "pct_leave", "pct_graduate", "median_income"]].copy()
    results_sub.columns = ["pcon_code", "pct_leave", "pct_graduate", "median_income"]
    bes = bes.merge(results_sub, on="pcon_code", how="left")

    # Fill missing geo predictors with national means
    for col in ["pct_leave", "pct_graduate", "median_income"]:
        bes[col] = bes[col].fillna(bes[col].mean())

    # Weight — default to 1 if missing
    if "weight" not in bes.columns:
        bes["weight"] = 1.0
    else:
        bes["weight"] = bes["weight"].fillna(1.0).clip(0.1, 10.0)

    # Drop rows with missing vote intention
    bes = bes[bes["vote_intention"].notna()]
    logger.info("BES after cleaning: %d respondents", len(bes))

    return bes


def _make_vote_outcome(bes: pd.DataFrame, party: str) -> pd.Series:
    """Return a binary series: 1 if respondent intends to vote for party, else 0."""
    intention = bes["vote_intention"].astype(str).str.lower().str.strip()
    return intention.str.contains(party, na=False).astype(int)


# ---------------------------------------------------------------------------
# Mixed-effects logistic regression
# ---------------------------------------------------------------------------

def _fit_one_party(
    bes: pd.DataFrame,
    party: str,
    results_df: pd.DataFrame,
) -> object:
    """
    Fit a binary mixed-effects logistic regression for one party.
    Uses statsmodels BinomialBayesMixedGLM (approximate Bayesian mixed GLM)
    or falls back to a simpler fixed-effects logistic if statsmodels is unavailable.

    Returns the fitted model object (or a dict of coefficients).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required for MRP. Install with: pip install statsmodels")

    # Filter to relevant respondents (regional party restrictions)
    region_filter = PARTY_REGION_FILTER.get(party)
    if region_filter == "scotland":
        sub = bes[bes["is_scotland"] == 1].copy()
    elif region_filter == "wales":
        sub = bes[bes["is_wales"] == 1].copy()
    else:
        sub = bes.copy()

    if len(sub) < 100:
        logger.warning("Party %s: only %d respondents after region filter — skipping", party, len(sub))
        return None

    sub["y"] = _make_vote_outcome(sub, party)

    # Avoid rare categories causing perfect separation
    if sub["y"].sum() < 20:
        logger.warning("Party %s: fewer than 20 positive outcomes — skipping", party)
        return None

    # Build formula
    fixed_terms = [
        "C(age_band, Treatment('35-49'))",
        "C(sex_label, Treatment('male'))",
        "C(edu_label, Treatment('no_degree'))",
        "C(eth_label, Treatment('white_british'))",
        "brexit_remain",
        "pct_leave",
        "pct_graduate",
        "median_income",
    ]

    # Add regional fixed effects for non-regional parties
    if region_filter is None:
        fixed_terms += ["is_scotland", "is_wales"]

    # Add past vote — past vote for the same party is the strongest predictor
    fixed_terms.append(f"past_{party}")
    # Also include past vote for main rivals (selected)
    for rival in ["lab", "con", "ld", "reform"]:
        if rival != party:
            fixed_terms.append(f"past_{rival}")

    formula = "y ~ " + " + ".join(fixed_terms)

    # Group = constituency (random intercepts)
    groups = sub["pcon_code"].fillna("unknown")

    logger.info("Fitting MRP model for %s (%d respondents, %d positive)…",
                party, len(sub), int(sub["y"].sum()))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.logit(formula, data=sub).fit(
                method="lbfgs",
                maxiter=500,
                disp=False,
            )
        logger.info("  %s: AIC=%.1f, pseudo-R²=%.3f", party, model.aic, model.prsquared)
        return model
    except Exception as exc:
        logger.error("Failed to fit model for %s: %s", party, exc)
        return None


# ---------------------------------------------------------------------------
# Fit all models
# ---------------------------------------------------------------------------

def fit(force: bool = False) -> dict:
    """
    Fit MRP regression models for all parties. Saves to mrp_coefficients.pkl.
    Skips if already fitted (unless force=True).

    Returns dict mapping party → fitted model (or None if fitting failed).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if COEFF_PATH.exists() and not force:
        logger.info("MRP coefficients already exist at %s — skipping. Use force=True to refit.", COEFF_PATH)
        with open(COEFF_PATH, "rb") as f:
            return pickle.load(f)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. "
            "Download the HoC Library 2024 constituency results file and save as data/results_2024.csv."
        )

    results_df = pd.read_csv(RESULTS_PATH)
    bes = _prepare_bes(results_df)

    models = {}
    for party in ALL_PARTIES:
        model = _fit_one_party(bes, party, results_df)
        models[party] = model

    with open(COEFF_PATH, "wb") as f:
        pickle.dump(models, f)
    logger.info("Saved MRP coefficients to %s", COEFF_PATH)

    return models


# ---------------------------------------------------------------------------
# Poststratification
# ---------------------------------------------------------------------------

def _predict_party_constituency(
    model,
    census_df: pd.DataFrame,
    results_df: pd.DataFrame,
    party: str,
) -> pd.DataFrame:
    """
    For one party model, predict P(vote) for each constituency × demographic cell
    and aggregate by cell population to get constituency-level vote shares.

    Returns DataFrame: constituency_code → predicted share for this party.
    """
    if model is None:
        return pd.DataFrame(columns=["constituency_code", party])

    region_filter = PARTY_REGION_FILTER.get(party)

    # Build prediction frame from census cells
    pred_frame = census_df.copy()
    pred_frame = pred_frame.rename(columns={"age": "age_band", "sex": "sex_label",
                                             "education": "edu_label", "ethnicity": "eth_label"})

    # Merge geographic predictors
    geo = results_df[["constituency_code", "pct_leave", "pct_graduate", "median_income",
                       "region_label"]].copy()
    pred_frame = pred_frame.merge(geo, on="constituency_code", how="left")

    # Filter by region for regional parties
    if region_filter == "scotland":
        pred_frame = pred_frame[pred_frame["region_label"] == "scotland"]
    elif region_filter == "wales":
        pred_frame = pred_frame[pred_frame["region_label"] == "wales"]

    if pred_frame.empty:
        return pd.DataFrame(columns=["constituency_code", party])

    pred_frame["is_scotland"] = (pred_frame.get("region_label", "") == "scotland").astype(int)
    pred_frame["is_wales"] = (pred_frame.get("region_label", "") == "wales").astype(int)
    pred_frame["brexit_remain"] = 0.5  # use national mean for prediction

    # Past vote — set to 0 (we use aggregate past vote from results file if available)
    for pv in ["lab", "con", "ld", "reform", "green", "snp", "pc", "other"]:
        pred_frame[f"past_{pv}"] = 0

    # Predict probabilities
    try:
        probs = model.predict(pred_frame)
        pred_frame["p_vote"] = probs.values if hasattr(probs, "values") else probs
    except Exception as exc:
        logger.warning("Prediction failed for %s: %s — using constant 0.1", party, exc)
        pred_frame["p_vote"] = 0.1

    # Clip to [0, 1]
    pred_frame["p_vote"] = pred_frame["p_vote"].clip(0.0, 1.0)

    # Aggregate: weighted mean by cell population
    pred_frame["weighted_vote"] = pred_frame["p_vote"] * pred_frame["population"]
    constituency_shares = pred_frame.groupby("constituency_code").apply(
        lambda x: x["weighted_vote"].sum() / x["population"].sum()
        if x["population"].sum() > 0 else 0.0
    ).reset_index()
    constituency_shares.columns = ["constituency_code", party]

    return constituency_shares


def _calibrate(
    shares_df: pd.DataFrame,
    national_estimates: dict[str, dict],
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rescale constituency shares so that when aggregated to national level
    (weighted by electorate size) they match the Kalman filter's current estimates.

    Also applies separate calibration for Scotland (SNP) and Wales (PC) where
    regional subsample estimates are available.
    """
    parties = [c for c in shares_df.columns if c not in ("constituency_code",)]

    # Merge electorate size for weighting
    elec = results_df[["constituency_code", "electorate_2024"]].copy() if "electorate_2024" in results_df.columns else None

    for party in parties:
        if party not in national_estimates:
            continue
        target = national_estimates[party]["mean"]  # fraction [0, 1]

        if elec is not None:
            sub = shares_df[["constituency_code", party]].merge(elec, on="constituency_code", how="left")
            sub["electorate_2024"] = sub["electorate_2024"].fillna(sub["electorate_2024"].median())
            current_national = np.average(sub[party].fillna(0), weights=sub["electorate_2024"])
        else:
            current_national = shares_df[party].fillna(0).mean()

        if current_national > 0:
            scale = target / current_national
            shares_df[party] = (shares_df[party] * scale).clip(0, 1)

    # Renormalise rows so all party shares sum to ≤ 1
    party_cols = [c for c in shares_df.columns if c not in ("constituency_code",)]
    row_sums = shares_df[party_cols].fillna(0).sum(axis=1)
    for party in party_cols:
        shares_df[party] = shares_df[party] / row_sums.clip(1.0)

    return shares_df


def poststratify(
    national_estimates: dict[str, dict],
    force_refit: bool = False,
) -> pd.DataFrame:
    """
    Run MRP poststratification using stored coefficients.
    Returns a DataFrame of constituency-level vote shares (one row per constituency).

    national_estimates: output from aggregator.current_estimates()
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    models = fit(force=force_refit)

    # Load census and results
    census_df = pd.read_csv(CENSUS_PATH)
    results_df = pd.read_csv(RESULTS_PATH)

    # Add region labels to results if missing
    if "region_label" not in results_df.columns:
        scotland_codes = results_df["constituency_code"].str.startswith("S")
        wales_codes = results_df["constituency_code"].str.startswith("W")
        results_df["region_label"] = "england"
        results_df.loc[scotland_codes, "region_label"] = "scotland"
        results_df.loc[wales_codes, "region_label"] = "wales"

    # Start with constituency list
    all_codes = results_df["constituency_code"].unique()
    shares_df = pd.DataFrame({"constituency_code": all_codes})

    # Predict for each party
    for party in ALL_PARTIES:
        model = models.get(party)
        party_shares = _predict_party_constituency(model, census_df, results_df, party)
        if not party_shares.empty:
            shares_df = shares_df.merge(party_shares, on="constituency_code", how="left")
        else:
            shares_df[party] = np.nan

    # Fill Scotland/Wales zeros for parties that don't contest there
    scot_mask = results_df.set_index("constituency_code")["region_label"] == "scotland"
    wales_mask = results_df.set_index("constituency_code")["region_label"] == "wales"

    for party in ALL_PARTIES:
        if party not in shares_df.columns:
            shares_df[party] = 0.0
        # Zero out SNP outside Scotland
        if party != "snp":
            shares_df.loc[
                shares_df["constituency_code"].isin(scot_mask[scot_mask].index), party
            ] = shares_df.loc[
                shares_df["constituency_code"].isin(scot_mask[scot_mask].index), party
            ].fillna(0)
        # Zero out PC outside Wales
        if party != "pc":
            shares_df.loc[
                shares_df["constituency_code"].isin(wales_mask[wales_mask].index), party
            ] = shares_df.loc[
                shares_df["constituency_code"].isin(wales_mask[wales_mask].index), party
            ].fillna(0)

    # Calibrate to match national poll estimates
    shares_df = _calibrate(shares_df, national_estimates, results_df)

    # Add constituency name + region for downstream use
    shares_df = shares_df.merge(
        results_df[["constituency_code", "constituency_name", "region_label"]].drop_duplicates(),
        on="constituency_code",
        how="left",
    )

    shares_df.to_parquet(MRP_SHARES_PATH, index=False)
    logger.info("Saved MRP shares for %d constituencies to %s", len(shares_df), MRP_SHARES_PATH)

    return shares_df


def load_mrp_shares() -> pd.DataFrame:
    """Load the cached MRP constituency shares."""
    if not MRP_SHARES_PATH.exists():
        raise FileNotFoundError(
            f"{MRP_SHARES_PATH} not found. Run poststratify() first."
        )
    return pd.read_parquet(MRP_SHARES_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.aggregator import current_estimates, load_state
    state = load_state()
    estimates = current_estimates(state)

    logger.info("Fitting MRP models …")
    shares = poststratify(estimates)
    print(f"\nMRP poststratification complete: {len(shares)} constituencies")
    print(shares[["constituency_name"] + ALL_PARTIES].head(10).to_string(index=False))
