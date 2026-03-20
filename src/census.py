"""
census.py
---------
Pulls 2021 Census marginal distributions per Westminster constituency from the
ONS Nomis API and applies Iterative Proportional Fitting (IPF) to synthesise
the joint demographic distribution needed for MRP poststratification.

Strategy:
  The 2021 Census on Nomis has no Westminster constituency geography.
  We fetch data at Local Authority District level (TYPE154, ~330 LAs) to
  avoid the 25,000-row API limit that makes ward-level requests impractical.
  LA data is then allocated to constituencies proportionally using ward counts
  from the ONS Ward->PCON July 2024 lookup as population-weight proxies.

Tables downloaded (correct Nomis NM_ IDs verified against API 2026-03):
  NM_2027_1  TS007  -- age by single year of age
  NM_2028_1  TS008  -- sex
  NM_2084_1  TS067  -- highest level of qualification
  NM_2041_1  TS021  -- ethnic group

Output: data/census_2021.csv
  One row per constituency x demographic cell (40 cells per constituency).
  Columns: constituency_code, constituency_name, age, sex, education,
           ethnicity, population, proportion

This is static data -- it only needs to run once.

Run as a module:  python -m src.census
"""

from __future__ import annotations

import logging
import re
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR  = DATA_DIR / "raw"
OUT_CSV  = DATA_DIR / "census_2021.csv"

# ---------------------------------------------------------------------------
# Nomis API constants (verified 2026-03)
# ---------------------------------------------------------------------------

NOMIS_BASE = "https://www.nomisweb.co.uk/api/v01/dataset"

NM_AGE       = "NM_2027_1"   # TS007 - age by single year
NM_SEX       = "NM_2028_1"   # TS008 - sex
NM_EDUCATION = "NM_2084_1"   # TS067 - highest level of qualification
NM_ETHNICITY = "NM_2041_1"   # TS021 - ethnic group

# Local Authority District geography -- ~330 LAs, fits comfortably under the
# Nomis 25,000-row limit even with 100+ age categories
GEO_TYPE = "TYPE154"

# Ward -> 2024 Westminster constituency lookup (ONS July 2024)
WARD_LOOKUP_URL = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items"
    "/62eb9df29a2f4521b5076a419ff9a47e/csv?layers=0"
)
WARD_LOOKUP_CACHE = RAW_DIR / "ward_pcon24_lookup.csv"

# ---------------------------------------------------------------------------
# Age bands: TS007 single-year codes where code N = age (N-1)
# (code 1 = age 0, code 19 = age 18, ... code 102 = age 101)
# ---------------------------------------------------------------------------
AGE_BAND_CODES: dict[str, list[int]] = {
    "18-24": list(range(19, 26)),    # ages 18-24
    "25-34": list(range(26, 36)),    # ages 25-34
    "35-49": list(range(36, 50)),    # ages 35-49
    "50-64": list(range(51, 65)),    # ages 50-64
    "65+":   list(range(66, 103)),   # ages 65+
}

# TS008: 0=total, 1=male, 2=female
SEX_MALE_CODE   = 1
SEX_FEMALE_CODE = 2

# TS021: code 0=total, 1=White: English/Welsh/Scottish/NI/British
ETHNICITY_TOTAL_CODE         = 0
ETHNICITY_WHITE_BRITISH_CODE = 1

# TS067: code 0=total, 6=Level 4 qualifications and above (degree equivalent)
EDUCATION_TOTAL_CODE  = 0
EDUCATION_DEGREE_CODE = 6


# ---------------------------------------------------------------------------
# Nomis API helper
# ---------------------------------------------------------------------------

def _nomis_get(dataset: str, extra_params: dict, *, retries: int = 3) -> pd.DataFrame:
    """Query Nomis, return a cleaned DataFrame. Retries on transient failures."""
    url = f"{NOMIS_BASE}/{dataset}.data.csv"
    params = {
        "geography":   GEO_TYPE,
        "measures":    "20100",   # count
        "recordlimit": "50000",   # well above LA-level row counts
    }
    params.update(extra_params)

    for attempt in range(retries):
        try:
            logger.info("  Fetching %s (attempt %d)...", dataset, attempt + 1)
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            if not resp.text.strip():
                raise ValueError(f"Empty response from Nomis for {dataset}")
            df = pd.read_csv(StringIO(resp.text))
            df.columns = (
                df.columns
                .str.lower()
                .str.strip()
                .str.replace(r"\W+", "_", regex=True)
            )
            logger.info("    -> %d rows, %d LAs",
                        len(df), df["geography_code"].nunique())
            return df
        except Exception as exc:
            logger.warning("  Nomis request failed (attempt %d): %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {dataset} from Nomis after {retries} attempts")


# ---------------------------------------------------------------------------
# Ward -> constituency lookup and LA -> constituency allocation weights
# ---------------------------------------------------------------------------

def _load_ward_lookup() -> pd.DataFrame:
    """Download (or load cached) ONS ward -> 2024 PCON lookup."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if WARD_LOOKUP_CACHE.exists():
        logger.info("Loading cached ward lookup from %s", WARD_LOOKUP_CACHE)
        raw = pd.read_csv(WARD_LOOKUP_CACHE, dtype=str)
    else:
        logger.info("Downloading ONS ward->PCON24 lookup...")
        resp = requests.get(WARD_LOOKUP_URL, timeout=60)
        resp.raise_for_status()
        raw = pd.read_csv(StringIO(resp.text), dtype=str)
        raw.to_csv(WARD_LOOKUP_CACHE, index=False)
        logger.info("Saved ward lookup (%d rows) to %s", len(raw), WARD_LOOKUP_CACHE)

    # Strip BOM, quotes, and whitespace from column names and values
    raw.columns = [re.sub(r"[^A-Za-z0-9_]", "", c) for c in raw.columns]
    for col in raw.columns:
        raw[col] = raw[col].str.strip("'\" \t").str.strip()

    return raw


def _la_to_pcon_weights(ward_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Build fractional allocation weights from Local Authority to constituency.

    For each (LAD, PCON) pair, the weight = (wards in pair) / (wards in LAD).
    This uses ward count as a population proxy -- sufficient for MRP priors.

    Returns DataFrame with columns: lad_code, pcon_code, pcon_name, weight
    where weights for each LAD sum to 1.0.
    """
    # Identify columns (handle WD22CD or WD24CD variant)
    lad_col  = next(c for c in ward_lookup.columns if c.startswith("LAD") and c.endswith("CD"))
    pcon_col = next(c for c in ward_lookup.columns if c.startswith("PCON") and c.endswith("CD"))
    pcon_nm  = next(c for c in ward_lookup.columns if c.startswith("PCON") and c.endswith("NM"))

    counts = (
        ward_lookup
        .groupby([lad_col, pcon_col, pcon_nm], as_index=False)
        .size()
        .rename(columns={"size": "ward_count",
                         lad_col: "lad_code",
                         pcon_col: "pcon_code",
                         pcon_nm:  "pcon_name"})
    )
    lad_totals = counts.groupby("lad_code")["ward_count"].sum()
    counts["weight"] = counts["ward_count"] / counts["lad_code"].map(lad_totals)
    logger.info(
        "LA->constituency weights: %d LAs, %d constituencies",
        counts["lad_code"].nunique(), counts["pcon_code"].nunique(),
    )
    return counts[["lad_code", "pcon_code", "pcon_name", "weight"]]


def _la_to_pcon(la_df: pd.DataFrame, weights: pd.DataFrame,
                value_col: str) -> pd.DataFrame:
    """
    Allocate a per-LA value column to constituencies using fractional weights.

    la_df must have columns: geography_code (=LAD code), <value_col>
    Returns DataFrame with: pcon_code, pcon_name, <value_col>
    """
    merged = la_df.merge(weights, left_on="geography_code", right_on="lad_code", how="inner")
    if merged.empty:
        logger.warning("No LA codes matched the constituency lookup")
        return pd.DataFrame(columns=["pcon_code", "pcon_name", value_col])
    merged[value_col] = merged[value_col] * merged["weight"]
    result = merged.groupby(["pcon_code", "pcon_name"], as_index=False)[value_col].sum()
    return result


# ---------------------------------------------------------------------------
# Per-table fetch functions
# ---------------------------------------------------------------------------

def _fetch_age(weights: pd.DataFrame) -> pd.DataFrame:
    """Return constituency x age_band population counts.

    Fetches each age band in a separate API call to stay well under the
    Nomis 25,000-row limit (largest band has 37 codes x ~331 LAs = ~12K rows).
    """
    rows = []
    for band, codes in AGE_BAND_CODES.items():
        raw = _nomis_get(NM_AGE, {"c2021_age_102": ",".join(map(str, codes))})
        age_col = next(
            c for c in raw.columns
            if "age" in c and c not in ("geography_code", "geography_name")
        )
        la_agg = raw.groupby("geography_code", as_index=False)["obs_value"].sum()
        con_agg = _la_to_pcon(la_agg, weights, "obs_value")
        con_agg["age_band"] = band
        rows.append(con_agg)

    result = pd.concat(rows, ignore_index=True).rename(columns={"obs_value": "population"})
    return result[["pcon_code", "pcon_name", "age_band", "population"]]


def _fetch_sex(weights: pd.DataFrame) -> pd.DataFrame:
    """Return constituency x sex population counts."""
    raw = _nomis_get(NM_SEX, {"c_sex": f"{SEX_MALE_CODE},{SEX_FEMALE_CODE}"})
    sex_col = next(c for c in raw.columns if "sex" in c and c not in ("geography_code", "geography_name"))

    rows = []
    for code, label in [(SEX_MALE_CODE, "male"), (SEX_FEMALE_CODE, "female")]:
        la_agg = (
            raw[raw[sex_col] == code]
            .groupby("geography_code", as_index=False)["obs_value"]
            .sum()
        )
        con_agg = _la_to_pcon(la_agg, weights, "obs_value")
        con_agg["sex"] = label
        rows.append(con_agg)

    result = pd.concat(rows, ignore_index=True).rename(columns={"obs_value": "population"})
    return result[["pcon_code", "pcon_name", "sex", "population"]]


def _fetch_education(weights: pd.DataFrame) -> pd.DataFrame:
    """Return constituency x education (degree/no_degree) counts."""
    raw = _nomis_get(NM_EDUCATION, {})
    edu_col = next(
        c for c in raw.columns
        if ("hiqual" in c or "qual" in c) and c not in ("geography_code", "geography_name")
    )

    rows = []
    # Fetch total and degree; no_degree = total - degree
    for code, label in [(EDUCATION_TOTAL_CODE, "_total"), (EDUCATION_DEGREE_CODE, "degree")]:
        la_agg = (
            raw[raw[edu_col] == code]
            .groupby("geography_code", as_index=False)["obs_value"]
            .sum()
        )
        con_agg = _la_to_pcon(la_agg, weights, "obs_value")
        con_agg["education"] = label
        rows.append(con_agg)

    wide = pd.concat(rows, ignore_index=True)
    total_df  = wide[wide["education"] == "_total"][["pcon_code", "pcon_name", "obs_value"]].rename(columns={"obs_value": "total"})
    degree_df = wide[wide["education"] == "degree"][["pcon_code", "obs_value"]].rename(columns={"obs_value": "degree_pop"})
    merged = total_df.merge(degree_df, on="pcon_code", how="left").fillna(0)
    merged["no_degree"] = (merged["total"] - merged["degree_pop"]).clip(lower=0)

    result_rows = []
    for label, val_col in [("degree", "degree_pop"), ("no_degree", "no_degree")]:
        sub = merged[["pcon_code", "pcon_name", val_col]].copy()
        sub = sub.rename(columns={val_col: "population"})
        sub["education"] = label
        result_rows.append(sub)

    return pd.concat(result_rows, ignore_index=True)[["pcon_code", "pcon_name", "education", "population"]]


def _fetch_ethnicity(weights: pd.DataFrame) -> pd.DataFrame:
    """Return constituency x ethnicity (white_british/other) counts."""
    raw = _nomis_get(NM_ETHNICITY, {})
    eth_col = next(
        c for c in raw.columns
        if "eth" in c and c not in ("geography_code", "geography_name")
    )

    rows = []
    for code, label in [
        (ETHNICITY_TOTAL_CODE,         "_total"),
        (ETHNICITY_WHITE_BRITISH_CODE, "white_british"),
    ]:
        la_agg = (
            raw[raw[eth_col] == code]
            .groupby("geography_code", as_index=False)["obs_value"]
            .sum()
        )
        con_agg = _la_to_pcon(la_agg, weights, "obs_value")
        con_agg["ethnicity"] = label
        rows.append(con_agg)

    wide = pd.concat(rows, ignore_index=True)
    total_df = wide[wide["ethnicity"] == "_total"][["pcon_code", "pcon_name", "obs_value"]].rename(columns={"obs_value": "total"})
    wb_df    = wide[wide["ethnicity"] == "white_british"][["pcon_code", "obs_value"]].rename(columns={"obs_value": "wb_pop"})
    merged   = total_df.merge(wb_df, on="pcon_code", how="left").fillna(0)
    merged["other"] = (merged["total"] - merged["wb_pop"]).clip(lower=0)

    result_rows = []
    for label, val_col in [("white_british", "wb_pop"), ("other", "other")]:
        sub = merged[["pcon_code", "pcon_name", val_col]].copy()
        sub = sub.rename(columns={val_col: "population"})
        sub["ethnicity"] = label
        result_rows.append(sub)

    return pd.concat(result_rows, ignore_index=True)[["pcon_code", "pcon_name", "ethnicity", "population"]]


# ---------------------------------------------------------------------------
# IPF
# ---------------------------------------------------------------------------

def _ipf(seed: np.ndarray, marginals: list[np.ndarray],
         axes: list[list[int]], max_iter: int = 200, tol: float = 1e-4) -> np.ndarray:
    """Iterative Proportional Fitting."""
    x = seed.astype(float).copy()
    for _ in range(max_iter):
        x_prev = x.copy()
        for marg, ax in zip(marginals, axes):
            other = tuple(i for i in range(x.ndim) if i not in ax)
            current = x.sum(axis=other)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current > 0, marg / current, 0.0)
            shape = [1] * x.ndim
            for a in ax:
                shape[a] = x.shape[a]
            x = x * ratio.reshape(shape)
        if np.max(np.abs(x - x_prev)) < tol:
            break
    return x


def _build_joint(age_df: pd.DataFrame, sex_df: pd.DataFrame,
                 edu_df: pd.DataFrame, eth_df: pd.DataFrame) -> pd.DataFrame:
    """Run IPF per constituency -> flatten to one row per demographic cell."""
    AGE_LEVELS = ["18-24", "25-34", "35-49", "50-64", "65+"]
    SEX_LEVELS = ["male", "female"]
    EDU_LEVELS = ["degree", "no_degree"]
    ETH_LEVELS = ["white_british", "other"]

    constituencies = sorted(age_df["pcon_code"].unique())
    rows = []

    for pcon_code in constituencies:
        pcon_name = age_df.loc[age_df["pcon_code"] == pcon_code, "pcon_name"].iloc[0]

        def _marg(df, col, levels):
            sub = df[df["pcon_code"] == pcon_code].set_index(col)["population"]
            return np.array([sub.get(lv, 0.0) for lv in levels], dtype=float)

        m_age = _marg(age_df, "age_band",  AGE_LEVELS)
        m_sex = _marg(sex_df, "sex",       SEX_LEVELS)
        m_edu = _marg(edu_df, "education", EDU_LEVELS)
        m_eth = _marg(eth_df, "ethnicity", ETH_LEVELS)

        total = m_age.sum()
        if total <= 0:
            continue

        # Normalise all marginals to the age total
        for m in [m_sex, m_edu, m_eth]:
            if m.sum() > 0:
                m *= total / m.sum()

        seed = np.ones((5, 2, 2, 2)) * (total / (5 * 2 * 2 * 2))
        joint = _ipf(seed, [m_age, m_sex, m_edu, m_eth], [[0], [1], [2], [3]])

        for ai, age in enumerate(AGE_LEVELS):
            for si, sex in enumerate(SEX_LEVELS):
                for ei, edu in enumerate(EDU_LEVELS):
                    for xi, eth in enumerate(ETH_LEVELS):
                        rows.append({
                            "constituency_code": pcon_code,
                            "constituency_name": pcon_name,
                            "age":        age,
                            "sex":        sex,
                            "education":  edu,
                            "ethnicity":  eth,
                            "population": joint[ai, si, ei, xi],
                        })

    result = pd.DataFrame(rows)
    result["proportion"] = result.groupby("constituency_code")["population"].transform(
        lambda x: x / x.sum()
    )
    return result


# ---------------------------------------------------------------------------
# Scotland synthetic census
# ---------------------------------------------------------------------------
# ONS Nomis only covers England & Wales. Scotland uses the National Records of
# Scotland (NRS) census and a separate geography. We synthesise approximate
# demographic cells for each Scottish 2024 Westminster constituency using
# published 2022 Scottish Census aggregate proportions.
#
# Sources:
#   Age / sex:   NRS Table TS008 / TS007 (Scotland Census 2022)
#   Education:   NRS Table TS067 — ~47% with Level 4+ qualification
#   Ethnicity:   NRS Table TS021 — ~96% White (Scottish/British/Other)

_SCOTLAND_AGE_SHARES = {  # voting-age population (18+) proportions
    "18-24": 0.112,
    "25-34": 0.165,
    "35-49": 0.255,
    "50-64": 0.245,
    "65+":   0.223,
}
_SCOTLAND_SEX_SHARES = {"male": 0.488, "female": 0.512}
_SCOTLAND_EDU_SHARES = {"degree": 0.47, "no_degree": 0.53}
_SCOTLAND_ETH_SHARES = {"white_british": 0.960, "other": 0.040}


def _synthesise_scotland(results_path: Path) -> pd.DataFrame:
    """
    Build synthetic census cells for Scotland's 57 constituencies.
    Requires data/results_2024.csv for constituency codes, names, and
    electorates.
    """
    if not results_path.exists():
        logger.warning("results_2024.csv not found — cannot synthesise Scotland census")
        return pd.DataFrame()

    results = pd.read_csv(results_path)
    scotland = results[results["constituency_code"].str.startswith("S14")].copy()
    if scotland.empty:
        logger.warning("No S14 codes in results_2024.csv — Scotland census synthesis skipped")
        return pd.DataFrame()

    logger.info("Synthesising census cells for %d Scottish constituencies", len(scotland))

    rows = []
    for _, seat in scotland.iterrows():
        pcon_code = seat["constituency_code"]
        pcon_name = seat.get("constituency_name", pcon_code)
        electorate = seat.get("electorate_2024", 72000)

        for age, a_sh in _SCOTLAND_AGE_SHARES.items():
            for sex, s_sh in _SCOTLAND_SEX_SHARES.items():
                for edu, e_sh in _SCOTLAND_EDU_SHARES.items():
                    for eth, x_sh in _SCOTLAND_ETH_SHARES.items():
                        pop = electorate * a_sh * s_sh * e_sh * x_sh
                        rows.append({
                            "constituency_code": pcon_code,
                            "constituency_name": pcon_name,
                            "age":        age,
                            "sex":        sex,
                            "education":  edu,
                            "ethnicity":  eth,
                            "population": pop,
                        })

    df = pd.DataFrame(rows)
    df["proportion"] = df.groupby("constituency_code")["population"].transform(
        lambda x: x / x.sum()
    )
    return df


def patch_census_scotland(force: bool = False) -> pd.DataFrame:
    """
    Ensure census_2021.csv includes Scottish constituencies.
    If Scotland is already present and force=False, returns the existing CSV.
    Otherwise, synthesises Scottish cells and appends them.
    """
    results_path = DATA_DIR / "results_2024.csv"

    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        has_scotland = existing["constituency_code"].str.startswith("S14").any()
        if has_scotland and not force:
            logger.info("Census already has Scottish constituencies — skipping patch")
            return existing
        # Remove any stale Scottish rows before re-adding
        existing = existing[~existing["constituency_code"].str.startswith("S14")]
    else:
        existing = pd.DataFrame()

    scotland_df = _synthesise_scotland(results_path)
    if scotland_df.empty:
        return existing

    combined = pd.concat([existing, scotland_df], ignore_index=True)
    combined.to_csv(OUT_CSV, index=False)
    logger.info(
        "Patched census: %d total cells across %d constituencies (incl. %d Scottish)",
        len(combined),
        combined["constituency_code"].nunique(),
        scotland_df["constituency_code"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_census(force: bool = False) -> pd.DataFrame:
    """Download, aggregate, and save the joint census distribution."""
    if OUT_CSV.exists() and not force:
        logger.info("Census data already exists -- skipping (use --force to rerun)")
        return pd.read_csv(OUT_CSV)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ward->constituency lookup...")
    ward_lookup = _load_ward_lookup()
    weights     = _la_to_pcon_weights(ward_lookup)

    logger.info("Fetching age (TS007 / NM_2027_1)...")
    age_df = _fetch_age(weights)
    logger.info("  -> %d constituency x age_band rows", len(age_df))

    logger.info("Fetching sex (TS008 / NM_2028_1)...")
    sex_df = _fetch_sex(weights)
    logger.info("  -> %d rows", len(sex_df))

    logger.info("Fetching education (TS067 / NM_2084_1)...")
    edu_df = _fetch_education(weights)
    logger.info("  -> %d rows", len(edu_df))

    logger.info("Fetching ethnicity (TS021 / NM_2041_1)...")
    eth_df = _fetch_ethnicity(weights)
    logger.info("  -> %d rows", len(eth_df))

    logger.info("Running IPF to build joint distribution...")
    joint = _build_joint(age_df, sex_df, edu_df, eth_df)

    joint.to_csv(OUT_CSV, index=False)
    logger.info(
        "Saved %d cells across %d constituencies to %s",
        len(joint), joint["constituency_code"].nunique(), OUT_CSV,
    )

    # Extend with synthetic Scottish census (ONS Nomis covers E&W only)
    return patch_census_scotland(force=True)


def load() -> pd.DataFrame:
    """Load the cached census joint distribution."""
    if not OUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUT_CSV} not found. Run build_census() or python -m src.census first."
        )
    return pd.read_csv(OUT_CSV)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    joint = build_census(force=args.force)
    n_con = joint["constituency_code"].nunique()
    print(f"\nCensus: {len(joint)} cells across {n_con} constituencies")

    # Sanity check: White British share for a predominantly white constituency
    sample = joint[joint["constituency_code"] == joint["constituency_code"].iloc[0]]
    wb_share = sample[sample["ethnicity"] == "white_british"]["population"].sum() / sample["population"].sum()
    print(f"Sample constituency White British share: {wb_share:.1%}")
    print(joint.head(5).to_string(index=False))
