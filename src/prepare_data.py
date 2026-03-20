"""
prepare_data.py
---------------
Processes raw data files into the CSVs expected by mrp.py, ml_model.py
and seat_projector.py.

Outputs:
  data/bes_2024.csv        -- BES Wave 30 microdata
  data/results_2024.csv    -- 2024 GE results + geographic features

BES file used: BES2024_W30_Panel_v30.1.dta
  (Combined Wave 1-30 Internet Panel, N=122,372, version 30.1)
  Key columns used:
    generalElectionVoteW30  — current vote intention (May 2025)
    generalElectionVoteW29  — pre-election vote intention (Wave 29)
    euRefVoteW4             — EU referendum vote (Wave 4, 2016)
    gorW30                  — Government Office Region string label
    pcon_codeW30            — ONS constituency code
    Age                     — respondent age (constant, capital A)
    gender                  — Male / Female (constant)
    p_edlevelW30            — education level
    p_ethnicityW30          — ethnicity
    wt_new_W30              — survey weight

Usage:
  python -m src.prepare_data            # skip if outputs exist
  python -m src.prepare_data --force    # always re-run
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

import re

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BES_CANDIDATES = [
    DATA_DIR / "BES2024_W30_Panel_v30.1.dta",   # Combined W1-30 Panel (v30.1)
    DATA_DIR / "BES2024_W30_Panel_v30.1.sav",
    DATA_DIR / "BES2024_W30_v30.1.dta",
    DATA_DIR / "BES2024_W30_v30.1.sav",
    DATA_DIR / "bes_wave30.dta",
    DATA_DIR / "bes_wave30.sav",
]
BES_STRINGS_PATH = DATA_DIR / "BES2024_W30Strings_v30.1.dta"
BES_OUT = DATA_DIR / "bes_2024.csv"

HOC_XLS = DATA_DIR / "HoC-GE2024-results-by-constituency.xlsx"
RESULTS_OUT = DATA_DIR / "results_2024.csv"
CENSUS_PATH = DATA_DIR / "census_2021.csv"
WARD_LOOKUP_PATH = RAW_DIR / "ward_pcon24_lookup.csv"

# EU referendum results (by Local Counting Area)
EU_REF_URL = (
    "https://www.electoralcommission.org.uk/sites/default/files/2019-07/"
    "EU-referendum-result-data.csv"
)
EU_REF_RAW = RAW_DIR / "eu_ref_lca.csv"

# ---------------------------------------------------------------------------
# BES column mapping (BES2024_W30_Panel_v30.1.dta — Combined Wave 1-30)
# ---------------------------------------------------------------------------
# Column names verified against the actual file. With convert_categoricals=True
# all categorical columns return string labels, not numeric codes.
BES_COL_MAP = {
    "weight":         "wt_new_W30",           # Survey weight for Wave 30 (~30k non-NaN)
    "region":         "gorW30",               # GOR string e.g. "Scotland", "South East"
    "pcon_code":      "pcon_codeW30",         # ONS code e.g. "E14001063"
    "age":            "Age",                  # Raw age (capital A; constant across waves)
    "sex":            "gender",               # "Male" / "Female" (constant)
    "edlevel":        "p_edlevelW30",         # "Undergraduate","GCSE","A-level","Postgrad","No qualifications","Below GCSE"
    "ethnicity":      "p_ethnicityW30",       # "White British", "Indian", etc.
    "vote_intention": "generalElectionVoteW30",  # Current vote intention (May 2025)
    "past_vote_2024": "generalElectionVoteW29",  # Pre-election intention (Wave 29, Summer 2024)
    "brexit_vote":    "euRefVoteW4",             # "Stay/remain in the EU" / "Leave the EU"
}

# String → normalised code mapping for vote columns
# Values confirmed from BES2024_W30_Panel_v30.1.dta with convert_categoricals=True
PARTY_LABEL_MAP = {
    "labour":                        "lab",
    "conservative":                  "con",
    "liberal democrat":              "ld",
    "liberal democrats":             "ld",
    "brexit party/reform uk":        "reform",
    "reform uk":                     "reform",
    "reform":                        "reform",
    "green party":                   "green",
    "green":                         "green",
    "scottish national party (snp)": "snp",
    "scottish national party":       "snp",
    "snp":                           "snp",
    "plaid cymru":                   "pc",
    "ukip":                          "reform",
    "uk independence party":         "reform",
    "brexit party":                  "reform",
    "an independent candidate":      "other",
    "i would/did not vote":          "notVote",
    "would not vote":                "notVote",
    "don't know":                    "dontKnow",
    "not sure":                      "dontKnow",
    "other":                         "other",
}

# BES GOR numeric → region label  (standard BES coding)
GOR_NUMERIC_MAP = {
    1: "north_east",
    2: "north_west",
    3: "yorkshire",
    4: "east_midlands",
    5: "west_midlands",
    6: "east",
    7: "london",
    8: "south_east",
    9: "south_west",
    10: "wales",
    11: "scotland",
    12: "northern_ireland",
}

# BES edlevel numeric → our two-level scheme (degree / no_degree)
def _edu_numeric(v: float) -> str:
    return "degree" if v >= 4 else "no_degree"

# BES gender numeric → sex label
def _sex_numeric(v: float) -> str:
    return "male" if v == 1 else "female"

# BES ethnicity numeric → white_british / other
def _ethnicity_numeric(v: float) -> str:
    return "white_british" if v == 1 else "other_ethnicity"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _normalise_party_label(raw: str) -> str:
    """Map a BES party string (any case) to our normalised code."""
    key = str(raw).lower().strip()
    for pattern, code in PARTY_LABEL_MAP.items():
        if pattern in key or key in pattern:
            return code
    return "other"


def _normalise_region(raw) -> str:
    """Map a BES GOR value (numeric or string) to a region label."""
    if pd.isna(raw):
        return "england"
    # Numeric code
    try:
        n = int(float(raw))
        return GOR_NUMERIC_MAP.get(n, "england")
    except (ValueError, TypeError):
        pass
    # String label
    s = str(raw).lower().strip()
    if "scotland" in s:
        return "scotland"
    if "wales" in s:
        return "wales"
    if "northern ireland" in s:
        return "northern_ireland"
    for num_label in GOR_NUMERIC_MAP.values():
        if num_label in s:
            return num_label
    return "england"


def _http_get_bytes(url: str, timeout: int = 30) -> bytes:
    headers = {"User-Agent": "uk-polling-model/1.0 (research project)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


# ---------------------------------------------------------------------------
# Part 1: BES microdata
# ---------------------------------------------------------------------------

def _find_bes_main() -> Path | None:
    """Return the path to the main BES dataset file, or None if not found."""
    for p in BES_CANDIDATES:
        if p.exists():
            return p
    return None


def _read_bes_dta(path: Path) -> pd.DataFrame:
    """
    Read a BES .dta file with categorical label expansion.
    Applies a monkey-patch for the Latin-1 strl encoding issue in pandas
    when reading large Stata 15+ format files.
    """
    logger.info("Reading BES Stata file: %s", path)

    import pandas.io.stata as stata_mod
    original_read_strls = stata_mod.StataReader._read_strls
    def _patched_strls(self):
        orig_enc = self._encoding
        self._encoding = "latin-1"
        try:
            original_read_strls(self)
        finally:
            self._encoding = orig_enc
    stata_mod.StataReader._read_strls = _patched_strls

    try:
        df = pd.read_stata(str(path), convert_categoricals=True)
    finally:
        stata_mod.StataReader._read_strls = original_read_strls

    logger.info("  Loaded: %d respondents, %d columns", len(df), len(df.columns))
    return df


def _read_bes_sav(path: Path) -> pd.DataFrame:
    """Read a BES .sav (SPSS) file."""
    logger.info("Reading BES SPSS file: %s", path)
    try:
        import pyreadstat  # type: ignore
        df, meta = pyreadstat.read_sav(str(path), apply_value_formats=True)
    except ImportError:
        logger.info("pyreadstat not available — reading SPSS via pandas")
        df = pd.read_spss(str(path))
    logger.info("  Loaded: %d respondents, %d columns", len(df), len(df.columns))
    return df


def _auto_detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Try to auto-detect BES column names when the defaults in BES_COL_MAP don't match.
    Looks for patterns in actual column names and returns an updated mapping.
    """
    cols = set(df.columns)
    mapping = dict(BES_COL_MAP)  # start with defaults

    # Vote intention: look for most recent generalElectionVoteW{n}
    vote_cols = sorted(
        [c for c in cols if c.startswith("generalElectionVoteW") and c[-2:].isdigit()],
        key=lambda c: int(c.replace("generalElectionVoteW", "")) if c.replace("generalElectionVoteW", "").isdigit() else 0,
        reverse=True,
    )
    # Filter out *Oth and *Ind variants (free text fields)
    vote_cols_main = [c for c in vote_cols if not any(x in c for x in ["Oth", "Ind", "text", "Post"])]
    if vote_cols_main:
        mapping["vote_intention"] = vote_cols_main[0]
        if len(vote_cols_main) > 1:
            mapping["past_vote_2024"] = vote_cols_main[1]
        logger.info("Auto-detected vote_intention: %s, past_vote: %s",
                    mapping["vote_intention"], mapping.get("past_vote_2024", "n/a"))

    # Weight
    wt_cols = [c for c in cols if c.startswith("wt_") or c == "finalWt" or c == "wt"]
    if wt_cols:
        # Pick the one with the highest wave number
        wt_w = [c for c in wt_cols if "W30" in c or "w30" in c.lower()]
        mapping["weight"] = wt_w[0] if wt_w else wt_cols[0]
        logger.info("Auto-detected weight: %s", mapping["weight"])

    # GOR — prefer the latest wave suffix (gorW30, gorW29, …)
    gor_wave_cols = sorted(
        [c for c in cols if c.startswith("gor") and c[3:].lstrip("W").isdigit()],
        key=lambda c: int(c.lstrip("gorW")),
        reverse=True,
    )
    fallback_gors = ["gorW30", "gorW1", "gor", "GOR", "region"]
    gor_candidates = (gor_wave_cols + fallback_gors)
    for candidate in gor_candidates:
        if candidate in cols:
            mapping["region"] = candidate
            logger.info("Auto-detected region: %s", candidate)
            break

    # PCON code — prefer pcon_codeW30 (ONS code), then pconW30 (name)
    for candidate in ["pcon_codeW30", "new_pcon_codeW30", "pconW30", "pcon", "PCON", "constituency"]:
        if candidate in cols:
            mapping["pcon_code"] = candidate
            logger.info("Auto-detected pcon_code: %s", candidate)
            break

    return mapping


def prepare_bes(force: bool = False) -> bool:
    """
    Process the main BES Wave 30 file into data/bes_2024.csv.
    Returns True if successful, False if main file not found.
    """
    if BES_OUT.exists() and not force:
        logger.info("BES CSV already exists at %s — skipping. Use --force to regenerate.", BES_OUT)
        return True

    bes_path = _find_bes_main()
    if bes_path is None:
        logger.warning(
            "\n"
            "═══════════════════════════════════════════════════════\n"
            "  BES MAIN DATASET NOT FOUND\n"
            "═══════════════════════════════════════════════════════\n"
            "  The file data/BES2024_W30Strings_v30.1.dta is a\n"
            "  SUPPLEMENTARY file (open-ended text only).\n"
            "  You also need the MAIN BES dataset.\n\n"
            "  Visit: https://www.britishelectionstudy.com/\n"
            "         data-objects/panel-study-data/\n"
            "  Download: BES Internet Panel Wave 30 (Stata or SPSS)\n"
            "  Save as:  data/BES2024_W30_v30.1.dta  (or .sav)\n"
            "═══════════════════════════════════════════════════════\n"
        )
        return False

    # Read file
    if bes_path.suffix.lower() == ".dta":
        df = _read_bes_dta(bes_path)
    else:
        df = _read_bes_sav(bes_path)

    # Auto-detect columns
    mapping = _auto_detect_columns(df)

    # Check required columns exist
    missing = []
    for std_name, raw_col in mapping.items():
        if raw_col not in df.columns:
            missing.append(f"{std_name} → {raw_col}")
    if missing:
        logger.warning("BES columns not found: %s", missing)
        logger.warning("Update BES_COL_MAP in src/prepare_data.py to match your file.")
        logger.warning("Available columns (first 40): %s", list(df.columns[:40]))

    # Filter to Wave 30 participants (rows where the weight is non-NaN)
    wt_col = mapping.get("weight", "wt_new_W30")
    if wt_col in df.columns:
        n_total = len(df)
        df = df[df[wt_col].notna()].copy()
        logger.info("Filtered to Wave 30 respondents: %d of %d rows", len(df), n_total)
        df = df.reset_index(drop=True)

    # Build output DataFrame with standardised names
    out = pd.DataFrame()
    out["id"] = df["id"] if "id" in df.columns else pd.RangeIndex(len(df))

    # Weight
    if wt_col in df.columns:
        out["weight"] = df[wt_col].clip(0.1, 10.0)
    else:
        out["weight"] = 1.0

    # Region → normalised label
    gor_col = mapping.get("region", "gor")
    if gor_col in df.columns:
        out["region"] = df[gor_col].apply(_normalise_region)
    else:
        out["region"] = "england"

    # Constituency code
    pcon_col = mapping.get("pcon_code", "pcon")
    out["pcon_code"] = df[pcon_col].astype(str) if pcon_col in df.columns else ""

    # Age
    age_col = mapping.get("age", "age")
    out["age"] = pd.to_numeric(df[age_col], errors="coerce").fillna(40) if age_col in df.columns else 40

    # Sex
    sex_col = mapping.get("sex", "gender")
    if sex_col in df.columns:
        raw_sex = df[sex_col]
        # Numeric or string
        if pd.api.types.is_numeric_dtype(raw_sex):
            out["sex"] = raw_sex.apply(lambda v: _sex_numeric(v) if pd.notna(v) else "male")
        else:
            out["sex"] = raw_sex.apply(lambda v: "male" if str(v).lower().startswith("m") else "female")
    else:
        out["sex"] = "male"

    # Education
    # BES string labels: "Undergraduate", "Postgrad", "A-level", "GCSE",
    #                    "Below GCSE", "No qualifications"
    edu_col = mapping.get("edlevel", "p_edlevelW30")
    if edu_col in df.columns:
        raw_edu = df[edu_col]
        if pd.api.types.is_numeric_dtype(raw_edu):
            out["edlevel"] = raw_edu.apply(lambda v: _edu_numeric(v) if pd.notna(v) else "no_degree")
        else:
            DEGREE_LABELS = {"undergraduate", "postgrad", "postgraduate", "degree", "university"}
            out["edlevel"] = raw_edu.apply(
                lambda v: "degree" if any(x in str(v).lower() for x in DEGREE_LABELS) else "no_degree"
            )
    else:
        out["edlevel"] = "no_degree"

    # Ethnicity
    eth_col = mapping.get("ethnicity", "p_ethnicity")
    if eth_col in df.columns:
        raw_eth = df[eth_col]
        if pd.api.types.is_numeric_dtype(raw_eth):
            out["ethnicity"] = raw_eth.apply(lambda v: _ethnicity_numeric(v) if pd.notna(v) else "white_british")
        else:
            out["ethnicity"] = raw_eth.apply(
                lambda v: "white_british" if "white british" in str(v).lower() else "other_ethnicity"
            )
    else:
        out["ethnicity"] = "white_british"

    # Vote intention → normalised party code
    vi_col = mapping.get("vote_intention", "generalElectionVoteW30")
    if vi_col in df.columns:
        out["vote_intention"] = df[vi_col].apply(_normalise_party_label)
    else:
        logger.warning("Vote intention column %s not found — MRP will not work", vi_col)
        out["vote_intention"] = np.nan

    # Past vote 2024 (pre-election intention)
    pv_col = mapping.get("past_vote_2024", "generalElectionVoteW29")
    if pv_col in df.columns:
        out["past_vote_2024"] = df[pv_col].apply(_normalise_party_label)
    else:
        out["past_vote_2024"] = np.nan

    # Brexit vote
    # BES string labels: "Stay/remain in the EU", "Leave the EU", "Don't know"
    # Numeric coding in older waves: 0=Leave, 1=Remain, 9999=N/A
    eu_col = mapping.get("brexit_vote", "euRefVoteW4")
    if eu_col in df.columns:
        raw_eu = df[eu_col]
        def _parse_eu_vote(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, (int, float)):
                if v == 1:
                    return 1  # Remain
                if v == 0:
                    return 0  # Leave
                if v == 2:
                    return 0  # Legacy coding: 2=Leave in older BES waves
                return np.nan
            s = str(v).lower()
            if "remain" in s or "stay" in s:
                return 1
            if "leave" in s:
                return 0
            return np.nan
        out["brexit_vote"] = raw_eu.apply(_parse_eu_vote)
    else:
        out["brexit_vote"] = np.nan

    # Drop respondents with no vote intention
    before = len(out)
    out = out[out["vote_intention"].notna() & ~out["vote_intention"].isin(["notVote", "dontKnow"])]
    logger.info("Dropped %d rows with missing/non-vote intention; %d remain", before - len(out), len(out))

    out.to_csv(BES_OUT, index=False)
    logger.info("Saved BES CSV: %d respondents, %d columns → %s", len(out), len(out.columns), BES_OUT)
    return True


# ---------------------------------------------------------------------------
# Part 2: EU ref pct_leave by constituency
# ---------------------------------------------------------------------------

def _load_ward_lookup() -> pd.DataFrame:
    """Load the ward → constituency lookup (same as in census.py)."""
    if not WARD_LOOKUP_PATH.exists():
        logger.warning("Ward lookup not found at %s — pct_leave will be unavailable.", WARD_LOOKUP_PATH)
        return pd.DataFrame()

    lk = pd.read_csv(WARD_LOOKUP_PATH, dtype=str)
    # Strip BOM and non-alphanumeric characters from column names (same as census.py)
    lk.columns = [re.sub(r"[^A-Za-z0-9_]", "", c) for c in lk.columns]
    for col in lk.columns:
        lk[col] = lk[col].str.strip("'\" \t").str.strip()
    return lk


def _fetch_eu_ref() -> pd.DataFrame:
    """Download EU referendum results by Local Counting Area (LAD-level codes)."""
    if EU_REF_RAW.exists():
        logger.info("Using cached EU ref data: %s", EU_REF_RAW)
        return pd.read_csv(EU_REF_RAW)

    logger.info("Downloading EU referendum results…")
    try:
        content = _http_get_bytes(EU_REF_URL)
        df = pd.read_csv(io.StringIO(content.decode("utf-8-sig", errors="replace")))
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(EU_REF_RAW, index=False)
        logger.info("Saved EU ref data (%d areas) to %s", len(df), EU_REF_RAW)
        return df
    except Exception as exc:
        logger.warning("Failed to download EU ref data: %s — pct_leave will default to 0.5", exc)
        return pd.DataFrame()


def _compute_pct_leave(results_df: pd.DataFrame) -> pd.Series:
    """
    Compute pct_leave for each constituency using:
      1. EU ref Local Counting Area data (LAD-level)
      2. Ward → constituency lookup to aggregate LAD → constituency
    Returns a Series indexed by constituency_code.
    """
    eu_ref = _fetch_eu_ref()
    if eu_ref.empty:
        return pd.Series(0.5, index=results_df["constituency_code"])

    lk = _load_ward_lookup()
    if lk.empty:
        return pd.Series(0.5, index=results_df["constituency_code"])

    # Identify columns
    # EU ref: Area_Code (LAD), Leave, Valid_Votes
    leave_col = next((c for c in eu_ref.columns if "leave" in c.lower() and "pct" not in c.lower() and "rej" not in c.lower()), None)
    valid_col = next((c for c in eu_ref.columns if "valid" in c.lower() and "votes" in c.lower()), None)
    area_col = next((c for c in eu_ref.columns if "area_code" in c.lower()), None)

    if not all([leave_col, valid_col, area_col]):
        logger.warning("EU ref data columns not as expected: %s", list(eu_ref.columns))
        return pd.Series(0.5, index=results_df["constituency_code"])

    eu_ref[leave_col] = pd.to_numeric(eu_ref[leave_col], errors="coerce").fillna(0)
    eu_ref[valid_col] = pd.to_numeric(eu_ref[valid_col], errors="coerce").fillna(1)
    eu_ref_clean = eu_ref[[area_col, leave_col, valid_col]].rename(
        columns={area_col: "lad_code", leave_col: "leave_votes", valid_col: "valid_votes"}
    )

    # Identify ward/LAD/constituency columns in lookup (same logic as census.py)
    ward_col = next((c for c in lk.columns if c.startswith("WD") and c.endswith("CD")), None)
    lad_col  = next((c for c in lk.columns if c.startswith("LAD") and c.endswith("CD")), None)
    pcon_col = next((c for c in lk.columns if c.startswith("PCON") and c.endswith("CD")), None)

    if not all([ward_col, lad_col, pcon_col]):
        logger.warning("Could not identify ward/LAD/PCON columns in lookup: %s", list(lk.columns))
        return pd.Series(0.5, index=results_df["constituency_code"])

    # Count wards per LAD per constituency
    ward_counts = (
        lk.groupby([lad_col, pcon_col])
        .size()
        .reset_index(name="n_wards")
    )
    lad_total = ward_counts.groupby(lad_col)["n_wards"].sum().reset_index(name="lad_total")
    ward_counts = ward_counts.merge(lad_total, on=lad_col)
    ward_counts["weight"] = ward_counts["n_wards"] / ward_counts["lad_total"]

    # Join EU ref data
    merged = ward_counts.merge(
        eu_ref_clean, left_on=lad_col, right_on="lad_code", how="left"
    )
    merged["leave_votes"] = merged["leave_votes"].fillna(0)
    merged["valid_votes"] = merged["valid_votes"].fillna(1)

    # Allocate leave/valid votes to constituency
    merged["alloc_leave"] = merged["leave_votes"] * merged["weight"]
    merged["alloc_valid"] = merged["valid_votes"] * merged["weight"]

    pcon_votes = (
        merged.groupby(pcon_col)
        .agg(total_leave=("alloc_leave", "sum"), total_valid=("alloc_valid", "sum"))
        .reset_index()
    )
    pcon_votes["pct_leave"] = (pcon_votes["total_leave"] / pcon_votes["total_valid"].clip(1)).clip(0, 1)
    pcon_votes = pcon_votes.rename(columns={pcon_col: "constituency_code"})

    result = results_df[["constituency_code"]].merge(
        pcon_votes[["constituency_code", "pct_leave"]], on="constituency_code", how="left"
    )
    result["pct_leave"] = result["pct_leave"].fillna(0.5)
    logger.info("pct_leave: computed for %d / %d constituencies",
                result["pct_leave"].notna().sum(), len(result))
    return result.set_index("constituency_code")["pct_leave"]


# ---------------------------------------------------------------------------
# Part 3: pct_graduate from census_2021.csv
# ---------------------------------------------------------------------------

def _compute_pct_graduate(results_df: pd.DataFrame) -> pd.Series:
    """Derive pct_graduate from the existing census_2021.csv."""
    if not CENSUS_PATH.exists():
        logger.warning("census_2021.csv not found — pct_graduate will default to 0.35")
        return pd.Series(0.35, index=results_df["constituency_code"])

    census = pd.read_csv(CENSUS_PATH)
    logger.info("Loaded census: %d rows, cols: %s", len(census), list(census.columns[:10]))

    # The census has: constituency_code, age, sex, education, ethnicity, population
    # We want: fraction with education == "degree" per constituency
    if "education" not in census.columns or "constituency_code" not in census.columns:
        logger.warning("census_2021.csv missing expected columns — pct_graduate defaults to 0.35")
        return pd.Series(0.35, index=results_df["constituency_code"])

    degree_pop = (
        census[census["education"] == "degree"]
        .groupby("constituency_code")["population"]
        .sum()
    )
    total_pop = census.groupby("constituency_code")["population"].sum()
    pct_grad = (degree_pop / total_pop.clip(1)).clip(0, 1).fillna(0.35)
    pct_grad.name = "pct_graduate"

    result = results_df[["constituency_code"]].merge(
        pct_grad.reset_index(), on="constituency_code", how="left"
    )
    result["pct_graduate"] = result["pct_graduate"].fillna(0.35)
    return result.set_index("constituency_code")["pct_graduate"]


# ---------------------------------------------------------------------------
# Part 4: HoC 2024 results → results_2024.csv
# ---------------------------------------------------------------------------

def prepare_results(force: bool = False) -> None:
    """
    Parse HoC-GE2024-results-by-constituency.xlsx into results_2024.csv.
    Includes vote shares, geographic features, and optionally 2019 data.
    """
    if RESULTS_OUT.exists() and not force:
        logger.info("results_2024.csv already exists — skipping. Use --force to regenerate.")
        return

    if not HOC_XLS.exists():
        raise FileNotFoundError(
            f"{HOC_XLS} not found.\n"
            "Download 'Detailed Results by Constituency' from:\n"
            "https://commonslibrary.parliament.uk/research-briefings/cbp-7529/"
        )

    logger.info("Parsing HoC 2024 results…")
    raw = pd.read_excel(HOC_XLS, sheet_name="Data", header=2)

    # Build the results DataFrame
    df = pd.DataFrame()
    df["constituency_code"] = raw["ONS ID"].astype(str).str.strip()
    df["constituency_name"] = raw["Constituency name"].astype(str).str.strip()
    df["electorate_2024"] = pd.to_numeric(raw["Electorate"], errors="coerce").fillna(0).astype(int)
    df["valid_votes_2024"] = pd.to_numeric(raw["Valid votes"], errors="coerce").fillna(0).astype(int)

    # Region label from HoC data
    def _region_label(row) -> str:
        country = str(row.get("Country name", "")).lower()
        region = str(row.get("Region name", "")).lower()
        if "scotland" in country:
            return "scotland"
        if "wales" in country:
            return "wales"
        if "northern ireland" in country:
            return "northern_ireland"
        return region if region else "england"

    df["region_label"] = raw.apply(_region_label, axis=1)

    # 2024 vote shares (party votes / valid votes)
    party_map_2024 = {
        "lab_2024":    "Lab",
        "con_2024":    "Con",
        "ld_2024":     "LD",
        "reform_2024": "RUK",
        "green_2024":  "Green",
        "snp_2024":    "SNP",
        "pc_2024":     "PC",
    }
    valid = df["valid_votes_2024"].replace(0, np.nan)
    for col_name, raw_col in party_map_2024.items():
        if raw_col in raw.columns:
            votes = pd.to_numeric(raw[raw_col], errors="coerce").fillna(0)
            df[col_name] = (votes / valid).clip(0, 1).fillna(0)
        else:
            df[col_name] = 0.0

    # Other candidates
    if "All other candidates" in raw.columns:
        others = pd.to_numeric(raw["All other candidates"], errors="coerce").fillna(0)
        df["others_2024"] = (others / valid).clip(0, 1).fillna(0)
    else:
        df["others_2024"] = 0.0

    # 2019 vote shares — these columns will be NaN if unavailable.
    # XGBoost handles missing gracefully (falls back to uniform national swing).
    # To add 2019 data: download HoC 2019 results and merge on constituency code.
    # Note: 2019 used OLD constituency boundaries — direct merge may miss ~100 seats.
    for col in ["lab_2019", "con_2019", "ld_2019", "brexit_2019",
                "green_2019", "snp_2019", "pc_2019", "others_2019"]:
        df[col] = np.nan

    logger.info("2019 vote share columns left as NaN — XGBoost will use uniform national swing.")
    logger.info("To add 2019 data, download HoC 2019 results and run prepare_results with --force.")

    # Geographic enrichment
    df["pct_leave"] = _compute_pct_leave(df).values
    df["pct_graduate"] = _compute_pct_graduate(df).values
    df["median_income"] = 0.0   # ONS ASHE data — add manually if needed
    df["urban_rural"] = 1       # 1=urban proxy; add ONS classification if needed

    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_OUT, index=False)
    logger.info("Saved results_2024.csv: %d constituencies, %d columns", len(df), len(df.columns))

    # Sanity-check a few known seats
    _sanity_check_results(df)


def _sanity_check_results(df: pd.DataFrame) -> None:
    """Log spot-checks for well-known constituencies."""
    checks = {
        "E14001063": ("Aldershot", "lab_2024", 0.35, 0.50),
        "E14001064": ("Aldridge-Brownhills", "con_2024", 0.30, 0.50),
        "S14000060": ("Aberdeen North", "snp_2024", 0.25, 0.55),
    }
    for code, (name, party_col, lo, hi) in checks.items():
        row = df[df["constituency_code"] == code]
        if row.empty:
            continue
        share = row[party_col].iloc[0]
        status = "OK" if lo <= share <= hi else "CHECK"
        logger.info("  %s %s %s=%.3f (expected %.2f-%.2f)",
                    status, name, party_col, share, lo, hi)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BES and results data files")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate outputs even if they already exist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. HoC results (can always run)
    logger.info("=" * 60)
    logger.info("Step 1: Preparing results_2024.csv …")
    prepare_results(force=args.force)

    # 2. BES microdata (requires main BES file)
    logger.info("=" * 60)
    logger.info("Step 2: Preparing bes_2024.csv …")
    bes_ok = prepare_bes(force=args.force)
    if not bes_ok:
        logger.warning("MRP model will be unavailable until the main BES file is downloaded.")

    logger.info("=" * 60)
    logger.info("Data preparation complete.")
    if bes_ok:
        logger.info("  Both files generated — run: python -m src.pipeline --full")
    else:
        logger.info("  results_2024.csv ready. Download main BES file to enable MRP.")


if __name__ == "__main__":
    main()
