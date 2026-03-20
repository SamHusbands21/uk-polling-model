"""
census.py
---------
Pulls 2021 Census marginal distributions per Westminster constituency from the
ONS Nomis API and applies Iterative Proportional Fitting (IPF) to synthesise
the joint demographic distribution needed for MRP poststratification.

Tables downloaded:
  TS007  — age (5 bands: 18-24, 25-34, 35-49, 50-64, 65+)
  TS008  — sex (male, female)
  TS067  — highest qualification (degree / no degree)
  TS021  — ethnicity (White British / other)

Output: data/census_2021.csv
  One row per constituency × demographic cell (40 cells per constituency).
  Columns: constituency_code, constituency_name, age, sex, education,
           ethnicity, population, proportion

This is static data — it only needs to run once.

Run as a module:  python -m src.census
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_CSV = DATA_DIR / "census_2021.csv"

# ---------------------------------------------------------------------------
# Nomis API
# ---------------------------------------------------------------------------

NOMIS_BASE = "https://www.nomisweb.co.uk/api/v01/dataset"

# Geography type for 2024 Westminster constituencies = type 693
# (2023 Boundary Review constituencies used at the 2024 election)
GEOGRAPHY_TYPE = 693  # Westminster Parliamentary Constituencies 2024

# Age bands we want (TS007 category codes, 18+ only)
AGE_BANDS = {
    "18-24": [5],         # TS007 code 5 = 15-24, we use as proxy for 18-24
    "25-34": [6],
    "35-49": [7, 8],      # 35-44 + 45-54 collapsed
    "50-64": [9, 10],     # 55-64 split → merge
    "65+":   [11, 12, 13, 14],
}

# TS021 ethnicity: 2 = White: English/Welsh/Scottish/Northern Irish/British
ETHNICITY_WHITE_BRITISH_CODE = 2
# TS067 qualification level 4 or 5 = degree level or above
DEGREE_CODES = [4, 5]


# ---------------------------------------------------------------------------
# Nomis fetcher
# ---------------------------------------------------------------------------

def _nomis_get(dataset: str, params: dict) -> pd.DataFrame:
    """
    Query the Nomis API for a dataset and return a DataFrame.
    Retries on transient errors.
    """
    url = f"{NOMIS_BASE}/{dataset}.data.csv"
    params = {
        "geography": f"TYPE{GEOGRAPHY_TYPE}",
        "select": "geography_code,geography_name,cell,obs_value",
        **params,
    }
    for attempt in range(3):
        try:
            logger.info("Fetching %s (attempt %d)…", dataset, attempt + 1)
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            logger.info("  → %d rows", len(df))
            return df
        except requests.RequestException as exc:
            logger.warning("Nomis request failed: %s", exc)
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {dataset} from Nomis after 3 attempts")


# ---------------------------------------------------------------------------
# Download individual tables
# ---------------------------------------------------------------------------

def _fetch_age(geo_codes: list[str]) -> pd.DataFrame:
    """
    Download TS007 (age) and return constituency × age-band population.
    Returns DataFrame with columns: constituency_code, age_band, population.
    """
    # Fetch all age categories for 18+ (codes 5-14 in TS007)
    raw = _nomis_get("NM_2051_1", {"measures": "20100"})
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")

    # Map cell codes to age bands
    band_rows = []
    for band_name, codes in AGE_BANDS.items():
        subset = raw[raw["cell"].isin(codes)]
        agg = subset.groupby(["geography_code", "geography_name"])["obs_value"].sum().reset_index()
        agg["age_band"] = band_name
        band_rows.append(agg)

    result = pd.concat(band_rows, ignore_index=True)
    result.columns = ["constituency_code", "constituency_name", "population", "age_band"]
    return result[["constituency_code", "constituency_name", "age_band", "population"]]


def _fetch_sex(geo_codes: list[str]) -> pd.DataFrame:
    """
    Download TS008 (sex). Returns constituency × sex population.
    cell=1: Male, cell=2: Female
    """
    raw = _nomis_get("NM_2052_1", {"cell": "1,2", "measures": "20100"})
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")
    raw["sex"] = raw["cell"].map({1: "male", 2: "female"})
    raw = raw[raw["sex"].notna()]
    result = raw[["geography_code", "geography_name", "sex", "obs_value"]].copy()
    result.columns = ["constituency_code", "constituency_name", "sex", "population"]
    return result


def _fetch_education(geo_codes: list[str]) -> pd.DataFrame:
    """
    Download TS067 (highest qualification).
    Degree = NVQ4 and above (codes 4, 5); No degree = all others.
    Returns constituency × education population.
    """
    raw = _nomis_get("NM_2084_1", {"measures": "20100"})
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")

    degree = raw[raw["cell"].isin(DEGREE_CODES)].groupby(
        ["geography_code", "geography_name"]
    )["obs_value"].sum().reset_index()
    degree["education"] = "degree"

    total = raw.groupby(
        ["geography_code", "geography_name"]
    )["obs_value"].sum().reset_index()

    no_degree_pop = total["obs_value"].values - degree["obs_value"].values
    no_degree = degree.copy()
    no_degree["obs_value"] = no_degree_pop
    no_degree["education"] = "no_degree"

    result = pd.concat([degree, no_degree], ignore_index=True)
    result.columns = ["constituency_code", "constituency_name", "population", "education"]
    return result[["constituency_code", "constituency_name", "education", "population"]]


def _fetch_ethnicity(geo_codes: list[str]) -> pd.DataFrame:
    """
    Download TS021 (ethnicity).
    White British = cell 2; Other = everything else.
    """
    raw = _nomis_get("NM_2041_1", {"measures": "20100"})
    raw.columns = raw.columns.str.lower().str.replace(" ", "_")

    wb = raw[raw["cell"] == ETHNICITY_WHITE_BRITISH_CODE].groupby(
        ["geography_code", "geography_name"]
    )["obs_value"].sum().reset_index()
    wb["ethnicity"] = "white_british"

    total = raw.groupby(
        ["geography_code", "geography_name"]
    )["obs_value"].sum().reset_index()

    other = total.copy()
    other["obs_value"] = total["obs_value"].values - wb["obs_value"].values
    other["ethnicity"] = "other_ethnicity"

    result = pd.concat([wb, other], ignore_index=True)
    result.columns = ["constituency_code", "constituency_name", "population", "ethnicity"]
    return result[["constituency_code", "constituency_name", "ethnicity", "population"]]


# ---------------------------------------------------------------------------
# Iterative Proportional Fitting (IPF)
# ---------------------------------------------------------------------------

def _ipf(
    seed: np.ndarray,
    marginals: list[tuple[list[int], np.ndarray]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Apply IPF to fit a seed joint distribution to a set of marginals.

    seed: N-dimensional array (the initial joint distribution)
    marginals: list of (axes, target_marginal) tuples.
                axes = dimensions to sum over to get the marginal.
    Returns: fitted N-dimensional array.
    """
    result = seed.copy().astype(float)
    result[result == 0] = 0.1  # avoid zero cells

    for iteration in range(max_iter):
        prev = result.copy()
        for axes, target in marginals:
            current_marginal = result.sum(axis=tuple(
                i for i in range(result.ndim) if i not in axes
            ))
            # Compute ratio, broadcast back
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current_marginal > 0, target / current_marginal, 0)
            # Expand ratio to match result dimensions
            expand_dims = tuple(i for i in range(result.ndim) if i not in axes)
            for dim in sorted(expand_dims):
                ratio = np.expand_dims(ratio, axis=dim)
            result = result * ratio

        delta = np.max(np.abs(result - prev))
        if delta < tol:
            logger.debug("IPF converged in %d iterations", iteration + 1)
            break

    return result


def _build_joint_distribution(
    age_df: pd.DataFrame,
    sex_df: pd.DataFrame,
    edu_df: pd.DataFrame,
    eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each constituency, apply IPF to produce the joint distribution
    P(age × sex × education × ethnicity | constituency).

    Returns a long DataFrame with one row per cell.
    """
    age_bands = ["18-24", "25-34", "35-49", "50-64", "65+"]
    sexes = ["male", "female"]
    educations = ["degree", "no_degree"]
    ethnicities = ["white_british", "other_ethnicity"]

    all_rows = []

    constituencies = age_df["constituency_code"].unique()
    logger.info("Running IPF for %d constituencies …", len(constituencies))

    for code in constituencies:
        name = age_df.loc[age_df["constituency_code"] == code, "constituency_name"].iloc[0]

        # Marginals for this constituency
        a = age_df[age_df["constituency_code"] == code].set_index("age_band")["population"]
        s = sex_df[sex_df["constituency_code"] == code].set_index("sex")["population"]
        e = edu_df[edu_df["constituency_code"] == code].set_index("education")["population"]
        eth = eth_df[eth_df["constituency_code"] == code].set_index("ethnicity")["population"]

        a_vec = np.array([a.get(b, 0) for b in age_bands], dtype=float)
        s_vec = np.array([s.get(x, 0) for x in sexes], dtype=float)
        e_vec = np.array([e.get(x, 0) for x in educations], dtype=float)
        eth_vec = np.array([eth.get(x, 0) for x in ethnicities], dtype=float)

        # Uniform seed
        seed = np.ones((5, 2, 2, 2), dtype=float)
        total = a_vec.sum()
        if total == 0:
            continue

        # Normalise marginals to have same total
        s_vec = s_vec / s_vec.sum() * total if s_vec.sum() > 0 else s_vec
        e_vec = e_vec / e_vec.sum() * total if e_vec.sum() > 0 else e_vec
        eth_vec = eth_vec / eth_vec.sum() * total if eth_vec.sum() > 0 else eth_vec

        marginals = [
            ([0], a_vec),
            ([1], s_vec),
            ([2], e_vec),
            ([3], eth_vec),
        ]

        try:
            fitted = _ipf(seed, marginals)
        except Exception as exc:
            logger.warning("IPF failed for %s: %s — using proportional fallback", code, exc)
            fitted = seed * (total / seed.sum())

        # Normalise to sum to total population
        fitted = fitted / fitted.sum() * total

        # Flatten to rows
        for i_a, age in enumerate(age_bands):
            for i_s, sex in enumerate(sexes):
                for i_e, edu in enumerate(educations):
                    for i_eth, eth_label in enumerate(ethnicities):
                        pop = fitted[i_a, i_s, i_e, i_eth]
                        all_rows.append({
                            "constituency_code": code,
                            "constituency_name": name,
                            "age": age,
                            "sex": sex,
                            "education": edu,
                            "ethnicity": eth_label,
                            "population": round(pop, 2),
                        })

    result = pd.DataFrame(all_rows)

    # Add proportion within each constituency
    result["proportion"] = result.groupby("constituency_code")["population"].transform(
        lambda x: x / x.sum()
    )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_census(force: bool = False) -> pd.DataFrame:
    """
    Download census marginals from Nomis, run IPF, save to data/census_2021.csv.
    Skips download if the output file already exists (pass force=True to re-run).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_CSV.exists() and not force:
        logger.info("Census data already exists at %s — skipping. Use force=True to re-download.", OUT_CSV)
        return pd.read_csv(OUT_CSV)

    logger.info("Fetching census marginals from ONS Nomis API …")

    # All constituency codes (we pass an empty list; the API returns all geographies of type 693)
    geo_codes: list[str] = []

    age_df = _fetch_age(geo_codes)
    sex_df = _fetch_sex(geo_codes)
    edu_df = _fetch_education(geo_codes)
    eth_df = _fetch_ethnicity(geo_codes)

    # Save raw marginals for reference
    age_df.to_csv(RAW_DIR / "census_age_raw.csv", index=False)
    sex_df.to_csv(RAW_DIR / "census_sex_raw.csv", index=False)
    edu_df.to_csv(RAW_DIR / "census_edu_raw.csv", index=False)
    eth_df.to_csv(RAW_DIR / "census_eth_raw.csv", index=False)
    logger.info("Saved raw marginals to data/raw/")

    joint = _build_joint_distribution(age_df, sex_df, edu_df, eth_df)
    joint.to_csv(OUT_CSV, index=False)

    logger.info(
        "Saved %d demographic cells across %d constituencies to %s",
        len(joint),
        joint["constituency_code"].nunique(),
        OUT_CSV,
    )
    return joint


def load() -> pd.DataFrame:
    """Load the census joint distribution from CSV."""
    if not OUT_CSV.exists():
        raise FileNotFoundError(
            f"{OUT_CSV} not found. Run build_census() or python -m src.census first."
        )
    return pd.read_csv(OUT_CSV)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    joint = build_census(force=False)
    print(f"\nCensus joint distribution: {len(joint):,} rows, "
          f"{joint['constituency_code'].nunique()} constituencies")
    print(joint.head(10).to_string(index=False))
