"""
scraper.py
----------
Scrapes UK Westminster opinion polls from the Wikipedia page and saves cleaned
data to data/processed/polls.parquet. Raw HTML is cached to data/raw/ to avoid
hammering Wikipedia on every run.

Scottish subsample columns (scotland_snp, scotland_lab, scotland_con) are
preserved where available from pollsters who publish them.

Run as a module:  python -m src.scraper
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIKI_URL = (
    "https://en.wikipedia.org/wiki/"
    "Opinion_polling_for_the_next_United_Kingdom_general_election"
)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_HTML = RAW_DIR / "wiki_polls_raw.html"
OUT_PARQUET = PROCESSED_DIR / "polls.parquet"

# Canonical internal party keys
PARTIES = ["lab", "con", "ld", "reform", "green", "snp", "pc", "others"]

# Mapping of Wikipedia column header variants → internal keys
PARTY_MAP: dict[str, str] = {
    # Labour
    "lab": "lab", "labour": "lab",
    # Conservative
    "con": "con", "cons": "con", "conservative": "con",
    # Liberal Democrats
    "ld": "ld", "lib dem": "ld", "lib dems": "ld", "liberal democrat": "ld",
    "liberal democrats": "ld",
    # Reform UK / Brexit Party variants
    "ref": "reform", "reform": "reform", "reform uk": "reform",
    "brx": "reform", "brexit": "reform",
    # Green
    "grn": "green", "green": "green", "greens": "green",
    # SNP
    "snp": "snp",
    # Plaid Cymru
    "pc": "pc", "plaid": "pc", "plaid cymru": "pc",
    # Others
    "oth": "others", "other": "others", "others": "others",
}

# Columns that are NOT party vote-share columns
NON_PARTY_COLS = {
    "pollster", "polling firm", "client", "dates conducted",
    "fieldwork dates", "dates", "sample size", "n", "sample",
    "lead", "change",
}

# Canonical pollster names. Wikipedia inconsistently spaces / renames some
# firms across tables and over time (older tables say "FindOutNow", newer
# ones "Find Out Now"; "Lord Ashcroft" vs "Lord Ashcroft Polls"). Without
# canonicalisation the EM step in the aggregator estimates two conflicting
# house effects for what is really one pollster, fragmenting its history and
# corrupting the latent-state anchor.
#
# Keys are pre-lowered, whitespace-normalised lookups; values are the
# canonical display form we want persisted everywhere downstream.
POLLSTER_ALIASES: dict[str, str] = {
    "findoutnow":             "Find Out Now",
    "find out now":           "Find Out Now",
    "lord ashcroft":          "Lord Ashcroft Polls",
    "lord ashcroft polls":    "Lord Ashcroft Polls",
    "we think":               "WeThink",
    "wethink":                "WeThink",
    "more in common":         "More in Common",
    "morein common":          "More in Common",
    "bmg":                    "BMG Research",
    "bmg research":           "BMG Research",
    "yougov":                 "YouGov",
    "you gov":                "YouGov",
    "jl partners":            "JL Partners",
    "j l partners":           "JL Partners",
    "freshwater":             "Freshwater Strategy",
    "freshwater strategy":    "Freshwater Strategy",
    "merlin":                 "Merlin Strategy",
    "merlin strategy":        "Merlin Strategy",
    "focal data":             "Focaldata",
    "focaldata":              "Focaldata",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_footnotes(val: str) -> str:
    """Remove Wikipedia footnote markers like [a], [1], [note 2] from a string."""
    return re.sub(r"\[[\w\s]+\]", "", str(val)).strip()


def _canonical_pollster(name: str) -> str:
    """
    Return a canonical display name for a pollster, collapsing spelling and
    spacing variants (e.g. "FindOutNow" / "Find Out Now" -> "Find Out Now").

    Lookup is case-insensitive and whitespace-collapsed. Unknown pollsters
    fall back to the stripped original, so this is conservative — we don't
    merge anyone we don't explicitly know about.
    """
    cleaned = re.sub(r"\s+", " ", str(name)).strip()
    return POLLSTER_ALIASES.get(cleaned.lower(), cleaned)


def _parse_share(raw) -> float | None:
    """Convert a raw cell value to a float percentage (0–100 range)."""
    if pd.isna(raw):
        return None
    cleaned = _strip_footnotes(str(raw)).replace(",", "").replace("%", "").strip()
    if cleaned in ("", "—", "-", "N/A", "n/a"):
        return None
    try:
        val = float(cleaned)
        # Wikipedia uses integer percentages (e.g. 34), not fractions (0.34)
        if val > 1:
            return val
        # Some tables use fractions – normalise to percentage
        return val * 100
    except ValueError:
        return None


def _parse_date_range(raw: str, default_year: int | None = None) -> tuple[str | None, str | None]:
    """
    Parse a fieldwork date string like "1–5 Jan 2026", "18 Mar", "13–16 Mar".
    Returns (start_str, end_str) as ISO date strings, or (None, None) on failure.

    Wikipedia polling tables often omit the year for recent polls. When no year
    is found, `default_year` is used (caller should pass the current year).
    If the inferred date would be in the future, the prior year is tried.
    """
    import datetime as _dt
    raw = _strip_footnotes(str(raw)).strip()
    if not raw or raw in ("—", "-"):
        return None, None

    # Normalise en-dash / em-dash and non-breaking hyphens to ASCII hyphen
    raw = raw.replace("\u2013", "-").replace("\u2014", "-").replace("\u2010", "-")
    # Remove any stray Unicode replacement characters
    raw = re.sub(r"[\ufffd\x00-\x08\x0b-\x1f]", "", raw)

    today = _dt.date.today()
    yr = default_year or today.year

    def _infer_year(month_str: str, day: int) -> int:
        """Return year such that the resulting date is not in the future."""
        m = int(_month(month_str))
        candidate = _dt.date(yr, m, min(day, 28))
        if candidate > today:
            return yr - 1
        return yr

    # Attempt patterns in order of specificity
    patterns = [
        # "28 Dec 2025-2 Jan 2026" (cross-year range with both years explicit)
        (r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*-\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
         lambda m: (
             f"{m.group(3)}-{_month(m.group(2))}-{int(m.group(1)):02d}",
             f"{m.group(6)}-{_month(m.group(5))}-{int(m.group(4)):02d}",
         )),
        # "28 Jan-2 Feb 2026" (cross-month range, one year)
        (r"(\d{1,2})\s+([A-Za-z]+)\s*-\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
         lambda m: (
             f"{m.group(5)}-{_month(m.group(2))}-{int(m.group(1)):02d}",
             f"{m.group(5)}-{_month(m.group(4))}-{int(m.group(3)):02d}",
         )),
        # "1-5 Jan 2026" (same-month range with year)
        (r"(\d{1,2})\s*-\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
         lambda m: (
             f"{m.group(4)}-{_month(m.group(3))}-{int(m.group(1)):02d}",
             f"{m.group(4)}-{_month(m.group(3))}-{int(m.group(2)):02d}",
         )),
        # "5 Jan 2026" (single date with year)
        (r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})",
         lambda m: (
             f"{m.group(3)}-{_month(m.group(2))}-{int(m.group(1)):02d}",
             f"{m.group(3)}-{_month(m.group(2))}-{int(m.group(1)):02d}",
         )),
        # "Jan 2026" (month only, with year)
        (r"([A-Za-z]+)\s+(\d{4})",
         lambda m: (
             f"{m.group(2)}-{_month(m.group(1))}-01",
             f"{m.group(2)}-{_month(m.group(1))}-28",
         )),
        # "14 Dec-9 Jan" (cross-month range, no year) — end month determines year
        (r"(\d{1,2})\s+([A-Za-z]+)\s*-\s*(\d{1,2})\s+([A-Za-z]+)$",
         lambda m: (
             lambda end_y=_infer_year(m.group(4), int(m.group(3))),
                    start_m=_month(m.group(2)), start_d=int(m.group(1)),
                    end_m=_month(m.group(4)), end_d=int(m.group(3)):
             # If start month > end month, start is previous year
             (f"{end_y if int(start_m) <= int(end_m) else end_y-1}-{start_m}-{start_d:02d}",
              f"{end_y}-{end_m}-{end_d:02d}")
         )(),
        ),
        # "1-5 Mar" (same-month range, no year)
        (r"(\d{1,2})\s*-\s*(\d{1,2})\s+([A-Za-z]+)$",
         lambda m: (
             f"{_infer_year(m.group(3), int(m.group(2)))}-{_month(m.group(3))}-{int(m.group(1)):02d}",
             f"{_infer_year(m.group(3), int(m.group(2)))}-{_month(m.group(3))}-{int(m.group(2)):02d}",
         )),
        # "18 Mar" (single date, no year)
        (r"(\d{1,2})\s+([A-Za-z]+)$",
         lambda m: (
             f"{_infer_year(m.group(2), int(m.group(1)))}-{_month(m.group(2))}-{int(m.group(1)):02d}",
             f"{_infer_year(m.group(2), int(m.group(1)))}-{_month(m.group(2))}-{int(m.group(1)):02d}",
         )),
    ]

    for pattern, extractor in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            try:
                result = extractor(match)
                if result[0] and result[1]:
                    return result
            except Exception:
                continue

    return None, None


_MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def _month(name: str) -> str:
    return _MONTH_MAP.get(name.lower()[:3], "01")


# ---------------------------------------------------------------------------
# Download / cache
# ---------------------------------------------------------------------------

def _fetch_html(force_refresh: bool = False) -> str:
    """Return cached HTML or download fresh copy from Wikipedia."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_HTML.exists() and not force_refresh:
        logger.info("Loading cached Wikipedia HTML from %s", RAW_HTML)
        return RAW_HTML.read_text(encoding="utf-8")

    logger.info("Downloading Wikipedia polling page …")
    headers = {"User-Agent": "uk-polling-model/1.0 (sam_website research project)"}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    RAW_HTML.write_text(resp.text, encoding="utf-8")
    logger.info("Saved to %s", RAW_HTML)
    return resp.text


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _normalise_col(col: str) -> str:
    """Flatten a potentially multi-level column name to a single lowercase string.

    Wikipedia polling tables produce MultiIndex columns like
    ('Lab', 'Unnamed: 5_level_1') where only the first element is meaningful.
    We use the first non-nan, non-Unnamed element.
    """
    if isinstance(col, tuple):
        meaningful = [
            str(c) for c in col
            if str(c).lower() != "nan" and "unnamed" not in str(c).lower()
        ]
        col = meaningful[0] if meaningful else str(col[0])
    return re.sub(r"\s+", " ", str(col)).strip().lower()


def _identify_party_col(col_norm: str) -> str | None:
    """Return the internal party key if this column is a vote-share column."""
    # Direct lookup
    if col_norm in PARTY_MAP:
        return PARTY_MAP[col_norm]
    # Partial match (handles "(SNP)" or "Reform UK" etc.)
    for fragment, key in PARTY_MAP.items():
        if fragment in col_norm:
            return key
    return None


def _is_election_result_row(row: pd.Series) -> bool:
    """
    Heuristic: election result rows typically have 'election' or 'result'
    in the pollster / client cell, or have an abnormally high sum of shares
    that would indicate they are totals rather than single-party poll values.
    """
    pollster_val = str(row.get("pollster", "")).lower()
    return "election" in pollster_val or "result" in pollster_val


def _parse_tables(html: str) -> pd.DataFrame:
    """
    Extract all tables from the Wikipedia page and return a cleaned
    combined DataFrame of polls.
    """
    from io import StringIO
    tables = pd.read_html(StringIO(html), flavor="lxml")
    logger.info("Found %d tables on the page", len(tables))

    frames: list[pd.DataFrame] = []

    for tbl_idx, tbl in enumerate(tables):
        # Flatten multi-level columns
        tbl.columns = [_normalise_col(c) for c in tbl.columns]

        col_names = list(tbl.columns)
        logger.debug("Table %d columns: %s", tbl_idx, col_names)

        # Identify key columns
        pollster_col = None
        date_col = None
        n_col = None

        for c in col_names:
            cl = c.lower()
            if any(k in cl for k in ("polling firm", "pollster", "firm", "company")):
                pollster_col = c
            elif any(k in cl for k in ("fieldwork", "date", "conducted")):
                date_col = c
            elif any(k in cl for k in ("sample", "n (sample", "sample size")):
                n_col = c

        # Must have at least pollster + date to be a polls table
        if pollster_col is None or date_col is None:
            logger.debug("Table %d skipped (no pollster/date columns)", tbl_idx)
            continue

        # Map columns to party keys
        party_cols: dict[str, str] = {}  # col_name → party_key
        for c in col_names:
            if c in (pollster_col, date_col, n_col):
                continue
            if any(skip in c for skip in ("lead", "change", "±")):
                continue
            key = _identify_party_col(c)
            if key is not None:
                party_cols[c] = key

        if not party_cols:
            logger.debug("Table %d skipped (no party columns identified)", tbl_idx)
            continue

        logger.info(
            "Table %d: pollster='%s', date='%s', parties=%s",
            tbl_idx, pollster_col, date_col, list(party_cols.values()),
        )

        rows = []
        for _, row in tbl.iterrows():
            pollster = _strip_footnotes(str(row.get(pollster_col, ""))).strip()
            if not pollster or pollster.lower() in ("nan", "polling firm", "pollster"):
                continue

            date_raw = str(row.get(date_col, ""))
            start_str, end_str = _parse_date_range(date_raw, default_year=pd.Timestamp.now().year)
            if end_str is None:
                logger.debug("Skipping row, could not parse date: %r", date_raw)
                continue

            try:
                fieldwork_end = pd.to_datetime(end_str, format="%Y-%m-%d")
                fieldwork_start = pd.to_datetime(start_str, format="%Y-%m-%d") if start_str else fieldwork_end
            except Exception:
                continue

            # Only keep post-2024-election polls (election was 4 July 2024)
            if fieldwork_end < pd.Timestamp("2024-07-04"):
                continue

            n_raw = _parse_share(row.get(n_col)) if n_col else None
            sample_size = int(n_raw) if n_raw and n_raw > 100 else None

            shares: dict[str, float | None] = {}
            for col, party_key in party_cols.items():
                val = _parse_share(row.get(col))
                # For duplicate party cols (e.g. two columns map to "others"),
                # keep the first non-None value
                if party_key not in shares or shares[party_key] is None:
                    shares[party_key] = val

            record: dict = {
                "pollster": pollster,
                "fieldwork_start": fieldwork_start,
                "fieldwork_end": fieldwork_end,
                "sample_size": sample_size,
            }
            for p in PARTIES:
                record[p] = shares.get(p)

            # Scottish subsample — look for columns like "Scotland SNP", "Scot SNP" etc.
            for col, party_key in party_cols.items():
                if "scot" in col.lower():
                    record[f"scotland_{party_key}"] = _parse_share(row.get(col))

            if _is_election_result_row(pd.Series(record)):
                continue

            rows.append(record)

        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        raise ValueError("No valid poll tables found on the Wikipedia page")

    combined = pd.concat(frames, ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, sort, and validate the combined polls DataFrame."""
    # Canonicalise pollster names BEFORE dedup so that "FindOutNow" /
    # "Find Out Now" (and friends) collapse into a single row per
    # pollster/date/sample_size. Without this the aggregator's EM step
    # estimates two conflicting house effects for what's really one firm.
    before_canon = df["pollster"].nunique()
    df = df.copy()
    df["pollster"] = df["pollster"].map(_canonical_pollster)
    after_canon = df["pollster"].nunique()
    if after_canon < before_canon:
        logger.info("Pollster canonicalisation: %d -> %d unique names",
                    before_canon, after_canon)

    # Deduplicate on (pollster, fieldwork_end, sample_size)
    before = len(df)
    df = df.drop_duplicates(
        subset=["pollster", "fieldwork_end", "sample_size"],
        keep="first",
    )
    logger.info("Deduplication: %d -> %d rows", before, len(df))

    # Sort by fieldwork end date
    df = df.sort_values("fieldwork_end").reset_index(drop=True)

    # Validate: require at least Lab + Con shares for a row to be useful
    df = df[df["lab"].notna() & df["con"].notna()]

    # Clip any absurd values to [0, 100]
    for p in PARTIES:
        if p in df.columns:
            df[p] = df[p].clip(0, 100)

    # -----------------------------------------------------------------------
    # Filter out non-GB-wide polls (Scotland-only, Wales-only, leadership
    # approval tables, battleground / constituency-type sub-polls, etc.).
    # Heuristics, layered from coarsest to finest:
    #
    #   1. At least three of the four main GB parties must be present
    #      (Lab, Con, Reform, LD) and their combined share >= 50 pp.
    #   2. No single party share > 60 pp (rules out devolved-region tables
    #      where SNP/PC can dominate).
    #   3. Lab and Con must each be <= 55 pp (plausible GB range).
    #   4. Scotland-only (SNP > 12 pp) and Wales-only (PC > 8 pp) excluded.
    #   5. TOTAL share across all recorded parties must sum to >= 90 pp.
    #      Proper national polls sum to ~100%; YouGov's seat-type sub-polls
    #      ("Blue Wall", "Lab target", "Con-Reform battleground") have
    #      geographically filtered electorates that only sum to 65-82%
    #      because the excluded share goes to "other/none/DK" inflated by
    #      the filtering. This catches the class of sub-polls that pass
    #      checks 1-4 (normal party mix, plausible shares) but aren't
    #      actually national voting intent.
    #   6. Sample size must be >= 500. Anything smaller is a sub-sample,
    #      crosstab, or stunt poll. (NaN sample size is allowed through;
    #      not every Wikipedia row records n, but no live voting-intention
    #      pollster publishes national polls with n<500.)
    # -----------------------------------------------------------------------
    four_party_cols = [p for p in ("lab", "con", "reform", "ld") if p in df.columns]
    four_party_sum = df[four_party_cols].sum(axis=1, min_count=3)
    df = df[four_party_sum >= 50]

    for p in PARTIES:
        if p in df.columns:
            df = df[df[p].isna() | (df[p] <= 60)]

    df = df[df["lab"] <= 55]
    df = df[df["con"] <= 55]

    if "snp" in df.columns:
        df = df[df["snp"].isna() | (df["snp"] <= 12)]
    if "pc" in df.columns:
        df = df[df["pc"].isna() | (df["pc"] <= 8)]

    # Filter 5: total shares sum >= 90
    party_cols = [p for p in PARTIES if p in df.columns]
    total_shares = df[party_cols].sum(axis=1, min_count=1)
    before_sum = len(df)
    df = df[total_shares >= 90]
    dropped_sum = before_sum - len(df)
    if dropped_sum > 0:
        logger.info(
            "Dropped %d rows whose party shares summed to < 90%% "
            "(sub-polls / battleground / constituency-type samples)",
            dropped_sum,
        )

    # Filter 6: minimum sample size
    if "sample_size" in df.columns:
        before_n = len(df)
        df = df[df["sample_size"].isna() | (df["sample_size"] >= 500)]
        dropped_n = before_n - len(df)
        if dropped_n > 0:
            logger.info(
                "Dropped %d rows with sample_size < 500 (sub-samples / stunt polls)",
                dropped_n,
            )

    df = df.reset_index(drop=True)
    logger.info("Final cleaned dataset: %d polls", len(df))
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape(force_refresh: bool = False) -> pd.DataFrame:
    """
    Download (or load from cache) the Wikipedia polling table and return a
    cleaned DataFrame. Saves the result to data/processed/polls.parquet.
    """
    html = _fetch_html(force_refresh=force_refresh)
    raw = _parse_tables(html)
    cleaned = _clean(raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PARQUET, index=False)
    logger.info("Saved %d polls to %s", len(cleaned), OUT_PARQUET)

    return cleaned


def load() -> pd.DataFrame:
    """Load the cached processed polls from parquet (run scrape() first)."""
    if not OUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{OUT_PARQUET} not found. Run scrape() or python -m src.scraper first."
        )
    return pd.read_parquet(OUT_PARQUET)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    df = scrape(force_refresh=True)
    print(f"\nScraped {len(df)} polls.")
    print(df[["pollster", "fieldwork_end", "lab", "con", "reform", "ld"]].tail(15).to_string(index=False))
