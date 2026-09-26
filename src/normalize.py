#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 2: Normalization Pipeline

Vectorized text cleaning for business names and addresses across ~24M records.
All string operations use pandas .str accessor (C-backed) — no per-row Python loops.
Batch edit-distance via rapidfuzz's C backend where needed.
Results cached to parquet (pyarrow) to avoid recomputation.

Usage:
    python src/normalize.py                                   # defaults
    python src/normalize.py --train-dir dataset/train --test-dir dataset/test
    python src/normalize.py --force                           # rebuild cache
"""

import argparse
import os
import re
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 on Windows (data contains Devanagari, Telugu, Kannada, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Optional: pyarrow for parquet caching
try:
    import pyarrow  # noqa: F401
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

# ====================================================================
# CONSTANTS
# ====================================================================

# ---- Legal-suffix synonym groups → canonical token ----
COMPOUND_SUFFIX_DEFS = [
    ("private limited", "pvt ltd"),
    ("pvt limited",     "pvt ltd"),
    ("private ltd",     "pvt ltd"),
    ("pvt ltd",         "pvt ltd"),
    ("public limited",  "plc"),
]

SINGLE_SUFFIX_MAP = {
    "incorporated": "inc",  "inc": "inc",
    "corporation":  "corp", "corp": "corp",
    "limited":      "ltd",  "ltd": "ltd",
    "private":      "pvt",  "pvt": "pvt",
    "company":      "co",   "co": "co",
    "llc": "llc", "llp": "llp", "plc": "plc", "pc": "pc",
    "gmbh": "gmbh",
    "sarl": "sarl", "sa": "sa", "sas": "sas",
    "sci": "sci", "eurl": "eurl",
    "pte": "pte", "pty": "pty",
    "ag": "ag", "bv": "bv", "nv": "nv",
    "sdn": "sdn", "bhd": "bhd",
}

# Combined canonical map: raw → canonical
_SUFFIX_CANON: dict[str, str] = {}
for _pat, _can in COMPOUND_SUFFIX_DEFS:
    _SUFFIX_CANON[_pat] = _can
_SUFFIX_CANON.update(SINGLE_SUFFIX_MAP)

# Build ordered alternation (compounds first → longest-match priority)
_all_sfx = [c[0] for c in COMPOUND_SUFFIX_DEFS]
_all_sfx += sorted(SINGLE_SUFFIX_MAP.keys(), key=len, reverse=True)
_seen: set[str] = set()
_all_sfx_dedup: list[str] = []
for _p in _all_sfx:
    if _p not in _seen:
        _seen.add(_p)
        _all_sfx_dedup.append(_p)

_SFX_ALT = "|".join(re.escape(p) for p in _all_sfx_dedup)
SUFFIX_EXTRACT_RE = rf"(?:^|\s+)({_SFX_ALT})\s*$"
SUFFIX_REMOVE_RE  = rf"(?:^|\s+)(?:{_SFX_ALT})\s*$"

# ---- Honorific / junk patterns ----
HONORIFIC_RE     = r"^\s*\b(?:sri|shri|smt|mr|mrs|dr|ms)\b\.?\s+"
LEADING_JUNK_RE  = r"^[-*#<>]+\s*"
REPEATED_PUNCT_RE = r"[-*]{2,}"

# ---- Domain detection ----
# Intentionally excludes .co and .in (ambiguous with company / India)
DOMAIN_RE_STR = r"^[a-z0-9][-a-z0-9]*(?:\.[a-z0-9][-a-z0-9]*)*\.(com|net|org|biz|io|us)$"
DOMAIN_TLD_RE = r"\.(com|net|org|biz|io|us)$"

# ---- Non-Latin script detection ----
NON_LATIN_RE = re.compile(
    r"[\u0900-\u097F"   # Devanagari
    r"\u0980-\u09FF"    # Bengali
    r"\u0A00-\u0A7F"    # Gurmukhi
    r"\u0A80-\u0AFF"    # Gujarati
    r"\u0B00-\u0B7F"    # Odia
    r"\u0B80-\u0BFF"    # Tamil
    r"\u0C00-\u0C7F"    # Telugu
    r"\u0C80-\u0CFF"    # Kannada
    r"\u0D00-\u0D7F"    # Malayalam
    r"\u0600-\u06FF"    # Arabic
    r"\u4E00-\u9FFF"    # CJK
    r"]"
)

# ---- Address abbreviation dictionaries ----
# Generic (all countries) — expanded to long canonical form
ADDR_ABBREV_GENERIC = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "cir": "circle", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "trl": "trail",
    "apt": "apartment", "ste": "suite", "fl": "floor",
    "bldg": "building", "dept": "department",
}

# India-specific (applied only to country=India rows)
ADDR_ABBREV_INDIA = {
    "nr": "near", "opp": "opposite",
    "dist": "district", "tal": "taluk",
}

# France-specific (applied only to country=France rows)
ADDR_ABBREV_FRANCE = {
    "bd": "boulevard", "imp": "impasse", "crs": "cours",
}

# ---- Landmark keywords (for address_landmark extraction) ----
LANDMARK_KEYWORDS = [
    "adjacent to", "next to", "in front of",       # multi-word first
    "near", "nr", "opposite", "opp",
    "behind", "beside", "facing", "towards",
]
_LM_ALT = "|".join(re.escape(k) for k in LANDMARK_KEYWORDS)
LANDMARK_RE = rf",?\s*\b(?:{_LM_ALT})\b[^,]*"

# US state abbreviation → full name (for address token normalisation)
US_STATE_ABBREV = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}

# Indian state abbreviation → full name
INDIA_STATE_ABBREV = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam",
    "br": "bihar", "cg": "chhattisgarh", "ga": "goa", "gj": "gujarat",
    "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "or": "orissa", "od": "odisha", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "jk": "jammu and kashmir", "la": "ladakh",
}


# ====================================================================
# HELPERS
# ====================================================================

def _expand_abbreviations(series: pd.Series, abbrev_dict: dict) -> pd.Series:
    """Single-pass word-boundary regex expansion using a compiled alternation.

    Much faster than looping over dict entries: one regex pass per row
    instead of N passes (N = number of abbreviations).
    """
    if not abbrev_dict:
        return series
    sorted_keys = sorted(abbrev_dict.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in sorted_keys) + r")\b"
    return series.str.replace(
        pattern,
        lambda m: abbrev_dict[m.group(1)],
        regex=True,
    )


def _sort_tokens(token_list):
    """Sort a list of tokens; used with Series.map()."""
    if isinstance(token_list, list) and token_list:
        return " ".join(sorted(token_list))
    return ""


def _token_set_str(token_list):
    """Sorted unique tokens as pipe-separated string; used with Series.map()."""
    if isinstance(token_list, list) and token_list:
        return "|".join(sorted(set(token_list)))
    return ""


# ====================================================================
# normalize_name
# ====================================================================

def normalize_name(series: pd.Series) -> pd.DataFrame:
    """Vectorized business-name normalization.

    Parameters
    ----------
    series : pd.Series[str]
        Raw ``business_name`` column.

    Returns
    -------
    pd.DataFrame  — columns:
        name_normalized       Cleaned name with canonical legal suffix.
        name_core             Cleaned name with suffix removed entirely.
        legal_suffix          Extracted canonical legal suffix ('' if none).
        name_tokens_sorted    Alphabetically sorted tokens of name_core.
        name_token_set        Pipe-separated sorted unique tokens of name_core.
        was_domain_format     True if the raw name was a bare domain.
        name_has_non_latin    True if raw name has non-Latin script characters.
    """
    s = series.fillna("").astype(str)
    result = pd.DataFrame(index=series.index)

    # ── Phase 0: flags on the *original* text ──────────────────────
    result["name_has_non_latin"] = s.str.contains(NON_LATIN_RE, na=False)

    # ── Phase 1: basic cleaning (all vectorized .str ops) ──────────
    s = s.str.lower().str.strip()

    # Domain-format detection + cleanup
    result["was_domain_format"] = s.str.match(DOMAIN_RE_STR, na=False)
    dm = result["was_domain_format"]
    if dm.any():
        s_dom = (
            s.str.replace(DOMAIN_TLD_RE, "", regex=True)
             .str.replace(r"[-_.]", " ", regex=True)
             .str.strip()
        )
        s = s.where(~dm, s_dom)

    # Honorific prefixes (word-boundary, start of string)
    s = s.str.replace(HONORIFIC_RE, "", regex=True)

    # Leading junk chars (#, *, <, >, -)
    s = s.str.replace(LEADING_JUNK_RE, "", regex=True)

    # Repeated punctuation runs (--, ***, ...)
    s = s.str.replace(REPEATED_PUNCT_RE, " ", regex=True)

    # Remove parentheses (often around legal suffixes or noise tokens)
    s = s.str.replace(r"[()]", "", regex=True)

    # & → 'and' (canonical — more common in clean text)
    s = s.str.replace(r"\s*&\s*", " and ", regex=True)

    # Remove periods ("Inc." → "Inc", "L.L.C." → "LLC")
    s = s.str.replace(r"\.", "", regex=True)

    # Strip punctuation EXCEPT hyphens and apostrophes
    s = s.str.replace(r"[^\w\s'-]", " ", regex=True)

    # Collapse whitespace
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # ── Phase 2: legal suffix extraction ───────────────────────────
    extracted_raw = s.str.extract(SUFFIX_EXTRACT_RE, expand=False)
    result["legal_suffix"] = extracted_raw.map(_SUFFIX_CANON).fillna("")

    # Remove suffix to get core
    name_core = s.str.replace(SUFFIX_REMOVE_RE, "", regex=True).str.strip()
    result["name_core"] = name_core

    # Normalised = core + canonical suffix
    has_sfx = result["legal_suffix"] != ""
    result["name_normalized"] = name_core.copy()
    if has_sfx.any():
        result.loc[has_sfx, "name_normalized"] = (
            name_core[has_sfx] + " " + result.loc[has_sfx, "legal_suffix"]
        )

    # ── Phase 3: token-based columns ───────────────────────────────
    # .map() on pre-split token lists is the only practical approach for
    # per-element sorting in pandas.  This is NOT the slow df.apply(axis=1)
    # anti-pattern — it runs on a Series of lists, ~3-5 s for 5 M rows.
    tokens = name_core.str.split()
    result["name_tokens_sorted"] = tokens.map(_sort_tokens).fillna("")
    result["name_token_set"]     = tokens.map(_token_set_str).fillna("")

    return result


# ====================================================================
# normalize_address
# ====================================================================

def normalize_address(
    series: pd.Series,
    country_series: pd.Series,
) -> pd.DataFrame:
    """Vectorized business-address normalization.

    Parameters
    ----------
    series : pd.Series[str]
        Raw ``business_address`` column.
    country_series : pd.Series[str]
        ``country`` column — selects which abbreviation dictionaries to apply.
        Unknown values (e.g. ``"France"``) gracefully fall back to
        the generic/shared dictionary.

    Returns
    -------
    pd.DataFrame — columns:
        address_clean           Fully cleaned address string.
        address_core            Address with landmark phrases removed.
        address_landmark        Extracted landmark phrases ('' if none).
        address_tokens_sorted   Sorted tokens of address_core.
        address_token_set       Pipe-separated sorted unique tokens.
        address_is_empty        True if address was empty/missing.
    """
    s = series.fillna("").astype(str)
    result = pd.DataFrame(index=series.index)

    # ── Phase 0: remember which were originally empty ──────────────
    is_empty_orig = s.str.strip() == ""

    # ── Phase 1: basic cleaning ────────────────────────────────────
    s = s.str.lower().str.strip()

    # Placeholder tokens
    s = s.str.replace("<null>", "", regex=False)
    s = s.str.replace(r"\bnull\b",  "", regex=True)
    s = s.str.replace(r"\bnone\b",  "", regex=True)
    s = s.str.replace(r"\bn/?a\b",  "", regex=True)

    # Hash prefixes on numbers: "##7308" → "7308"
    s = s.str.replace(r"#+(\d)", r"\1", regex=True)

    # Parentheses & periods
    s = s.str.replace(r"[()]", "", regex=True)
    s = s.str.replace(r"\.",   "", regex=True)

    # ── Phase 2: abbreviation expansion (country-aware) ────────────
    country_lower = country_series.fillna("").str.lower().str.strip()

    # Generic → all rows
    s = _expand_abbreviations(s, ADDR_ABBREV_GENERIC)

    # India-specific
    india_mask = country_lower == "india"
    if india_mask.any():
        s = s.copy()
        s.loc[india_mask] = _expand_abbreviations(
            s.loc[india_mask], ADDR_ABBREV_INDIA
        )

    # France-specific
    france_mask = country_lower == "france"
    if france_mask.any():
        s = s.copy()
        s.loc[france_mask] = _expand_abbreviations(
            s.loc[france_mask], ADDR_ABBREV_FRANCE
        )

    # Any other / unknown country: only generic (already applied). No error.

    # ── Phase 3: post-expansion cleanup ────────────────────────────
    # Keep slashes (building numbers like 4/2A), hyphens, commas
    s = s.str.replace(r"[^\w\s,/'-]", " ", regex=True)
    s = s.str.replace(r"\s+",        " ",  regex=True)
    s = s.str.replace(r"\s*,\s*",    ", ", regex=True)
    s = s.str.replace(r",(\s*,)+",   ",",  regex=True)
    s = s.str.strip(" ,")

    result["address_clean"] = s

    # Updated empty flag (original empty OR became empty after cleaning)
    result["address_is_empty"] = is_empty_orig | (s.str.strip() == "")

    # ── Phase 4: landmark extraction ───────────────────────────────
    landmarks = s.str.findall(LANDMARK_RE)
    result["address_landmark"] = landmarks.map(
        lambda lst: " | ".join(x.strip(" ,") for x in lst) if lst else ""
    ).fillna("")

    # address_core = address with landmarks removed
    addr_core = s.str.replace(LANDMARK_RE, "", regex=True)
    addr_core = addr_core.str.replace(r",(\s*,)+", ",", regex=True)
    addr_core = addr_core.str.replace(r"\s+", " ", regex=True)
    addr_core = addr_core.str.strip(" ,")
    result["address_core"] = addr_core

    # ── Phase 5: token columns ─────────────────────────────────────
    # Strip punctuation before tokenizing so tokens don't carry trailing commas/slashes
    tokens = addr_core.str.replace(r"[^\w\s'-]", " ", regex=True).str.split()
    result["address_tokens_sorted"] = tokens.map(_sort_tokens).fillna("")
    result["address_token_set"]     = tokens.map(_token_set_str).fillna("")

    return result


# ====================================================================
# build_normalized_source
# ====================================================================

def build_normalized_source(
    path: str,
    cache_dir: str = "output/normalized",
    force: bool = False,
) -> pd.DataFrame:
    """Load a source TSV, normalise, and cache to parquet.

    Subsequent calls with the same path return the cached DataFrame
    instantly (~1 s vs ~2 min for a 5 M-row file).
    """
    stem = Path(path).stem
    cache_parquet = Path(cache_dir) / f"{stem}.parquet"
    cache_pkl     = Path(cache_dir) / f"{stem}.pkl"

    # ── try cache ──────────────────────────────────────────────────
    if not force:
        if HAS_PYARROW and cache_parquet.exists():
            print(f"  ✓ Loading cached {cache_parquet}")
            return pd.read_parquet(cache_parquet)
        if cache_pkl.exists():
            print(f"  ✓ Loading cached {cache_pkl}")
            return pd.read_pickle(cache_pkl)

    # ── load raw TSV ───────────────────────────────────────────────
    print(f"  Loading {path} ...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"    {len(df):>10,d} rows loaded in {time.time()-t0:.1f}s")

    # ── normalise names ────────────────────────────────────────────
    print(f"    Normalizing names ...")
    t0 = time.time()
    name_cols = normalize_name(df["business_name"])
    print(f"    done in {time.time()-t0:.1f}s")

    # ── normalise addresses ────────────────────────────────────────
    print(f"    Normalizing addresses ...")
    t0 = time.time()
    addr_cols = normalize_address(df["business_address"], df["country"])
    print(f"    done in {time.time()-t0:.1f}s")

    # ── combine ────────────────────────────────────────────────────
    result = pd.concat([df, name_cols, addr_cols], axis=1)

    # ── cache ──────────────────────────────────────────────────────
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if HAS_PYARROW:
        result.to_parquet(cache_parquet, engine="pyarrow", index=False)
        sz = cache_parquet.stat().st_size / 1e6
        print(f"    Cached → {cache_parquet}  ({sz:.0f} MB)")
    else:
        result.to_pickle(cache_pkl)
        sz = cache_pkl.stat().st_size / 1e6
        print(f"    Cached → {cache_pkl}  ({sz:.0f} MB)")

    return result


# ====================================================================
# Reporting helpers
# ====================================================================

def _print_sample(df: pd.DataFrame, name: str, n: int = 20) -> None:
    """Print before/after examples for visual quality check."""
    print(f"\n{'='*80}")
    print(f"  SAMPLE: {name}  ({n} random rows)")
    print(f"{'='*80}\n")

    sample = df.sample(n=min(n, len(df)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print(f"  [{i:2d}] entity_id   : {row['entity_id']}")
        print(f"       NAME raw    : {row['business_name']}")
        print(f"       NAME core   : {row['name_core']}")
        print(f"       NAME norm   : {row['name_normalized']}")
        print(f"       suffix      : {row['legal_suffix'] or '(none)'}")
        sfx_flags = []
        if row.get("was_domain_format"):
            sfx_flags.append("DOMAIN")
        if row.get("name_has_non_latin"):
            sfx_flags.append("NON-LATIN")
        if sfx_flags:
            print(f"       flags       : {', '.join(sfx_flags)}")
        print(f"       ADDR raw    : {row['business_address']}")
        print(f"       ADDR core   : {row['address_core']}")
        if row.get("address_landmark"):
            print(f"       ADDR landmk : {row['address_landmark']}")
        if row.get("address_is_empty"):
            print(f"       ADDR empty  : True")
        print()


def _print_stats(df: pd.DataFrame, name: str) -> None:
    """Print aggregate normalisation statistics."""
    n = len(df)
    print(f"\n--- Stats: {name}  ({n:,d} rows) ---\n")

    # Name stats
    raw_lower = df["business_name"].str.lower().str.strip()
    name_changed = (df["name_core"] != raw_lower).sum()
    has_suffix    = (df["legal_suffix"] != "").sum()
    is_domain     = df["was_domain_format"].sum()
    has_non_latin = df["name_has_non_latin"].sum()

    print(f"  name_core differs from raw (lowered)   : {name_changed:>10,d}  ({100*name_changed/n:.2f}%)")
    print(f"  legal_suffix extracted                  : {has_suffix:>10,d}  ({100*has_suffix/n:.2f}%)")
    print(f"  was_domain_format                      : {is_domain:>10,d}  ({100*is_domain/n:.2f}%)")
    print(f"  name_has_non_latin                     : {has_non_latin:>10,d}  ({100*has_non_latin/n:.2f}%)")

    # Address stats
    addr_empty = df["address_is_empty"].sum()
    has_landmark = (df["address_landmark"] != "").sum()

    print(f"  address_is_empty                       : {addr_empty:>10,d}  ({100*addr_empty/n:.2f}%)")
    print(f"  address_landmark extracted              : {has_landmark:>10,d}  ({100*has_landmark/n:.2f}%)")

    # Suffix distribution (top 10)
    if has_suffix > 0:
        print(f"\n  Top legal suffixes:")
        for sfx, cnt in df["legal_suffix"].value_counts().head(12).items():
            if sfx == "":
                continue
            print(f"    {sfx:15s}  {cnt:>10,d}  ({100*cnt/n:.2f}%)")


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Normalize business names and addresses"
    )
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir",  default="dataset/test")
    parser.add_argument("--cache-dir", default="output/normalized")
    parser.add_argument("--force",     action="store_true",
                        help="Force rebuild even if cache exists")
    parser.add_argument("--sample",    type=int, default=20,
                        help="Number of sample rows to print per train file")
    args = parser.parse_args()

    tracemalloc.start()

    files = [
        (os.path.join(args.train_dir, "train_source1.tsv"), "train_source1"),
        (os.path.join(args.train_dir, "train_source2.tsv"), "train_source2"),
        (os.path.join(args.train_dir, "train_source3.tsv"), "train_source3"),
        (os.path.join(args.test_dir,  "test_source1.tsv"),  "test_source1"),
        (os.path.join(args.test_dir,  "test_source2.tsv"),  "test_source2"),
        (os.path.join(args.test_dir,  "test_source3.tsv"),  "test_source3"),
    ]

    t_total = time.time()
    for path, name in files:
        print(f"\n{'━'*70}")
        print(f"  Processing {name}")
        print(f"{'━'*70}")
        if not os.path.exists(path):
            print(f"  ⚠ File not found: {path}  — skipping")
            continue

        df = build_normalized_source(path, args.cache_dir, force=args.force)

        # Print samples for train files only
        if "train" in name:
            _print_sample(df, name, n=args.sample)

        _print_stats(df, name)

    elapsed = time.time() - t_total
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n{'='*70}")
    print(f"  ALL DONE in {elapsed:.0f}s")
    print(f"  Peak traced memory: {peak / 1e9:.2f} GB")
    print(f"  Cache directory:    {Path(args.cache_dir).resolve()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
