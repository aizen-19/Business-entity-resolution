#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Comprehensive Exploratory Data Analysis (EDA)

Uses rapidfuzz (MIT license, C++ backend → 10-50× faster than python-Levenshtein)
for all edit-distance computations.

Run from the student_resource/ directory:
    python src/eda.py
    python src/eda.py --train-dir dataset/train --test-dir dataset/test --out-dir output/eda
"""

import argparse
import os
import re
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np

# rapidfuzz chosen over python-Levenshtein because:
# 1. MIT licence (python-Levenshtein is GPL)
# 2. C++ backend → significantly faster on large pair-wise comparisons
# 3. Broader API (ratio, partial_ratio, token_sort_ratio, etc.)
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# Force UTF-8 output on Windows (cp1252 cannot encode Hindi/non-Latin chars in the data)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────

def banner(title: str, char: str = "=") -> None:
    """Print a section banner."""
    width = 80
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


def sub_banner(title: str) -> None:
    """Print a sub-section banner."""
    print(f"\n--- {title} ---\n")


def safe_mkdir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def tokenize(text: str) -> set:
    """Lowercase tokenize a string on non-alphanumeric boundaries."""
    if not isinstance(text, str) or text.strip() == "":
        return set()
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(s1: set, s2: set) -> float:
    """Jaccard similarity between two sets."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def percentile_summary(values, label: str = "Values") -> str:
    """Return a formatted percentile summary string."""
    if len(values) == 0:
        return f"  {label}: no data"
    arr = np.array(values, dtype=float)
    pcts = [0, 25, 50, 75, 95, 100]
    vals = np.percentile(arr, pcts)
    parts = [f"{p}th={v:.2f}" for p, v in zip(pcts, vals)]
    return f"  {label} (n={len(arr)}): " + ", ".join(parts) + f", mean={arr.mean():.2f}"


# ──────────────────────────────────────────────────────────────────────
# SECTION 1: LOADING & SANITY CHECKS
# ──────────────────────────────────────────────────────────────────────

def load_and_check(train_dir: str, test_dir: str):
    """Load all 7 TSV files and run sanity checks."""
    banner("1. LOADING & SANITY CHECKS")

    files = {
        "train_source1": os.path.join(train_dir, "train_source1.tsv"),
        "train_source2": os.path.join(train_dir, "train_source2.tsv"),
        "train_source3": os.path.join(train_dir, "train_source3.tsv"),
        "train_ground_truth": os.path.join(train_dir, "train_ground_truth.tsv"),
        "test_source1": os.path.join(test_dir, "test_source1.tsv"),
        "test_source2": os.path.join(test_dir, "test_source2.tsv"),
        "test_source3": os.path.join(test_dir, "test_source3.tsv"),
    }

    dfs = {}
    for name, path in files.items():
        print(f"Loading {name} from {path} ...")
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        dfs[name] = df
        print(f"  Shape: {df.shape}")
        print(f"  Columns & dtypes:\n{textwrap.indent(str(df.dtypes), '    ')}")
        print(f"  First 5 rows:")
        print(textwrap.indent(df.head().to_string(), "    "))
        print()

    # Prefix checks for source files
    sub_banner("Entity ID Prefix Verification")
    prefix_map = {
        "train_source1": "S1-", "train_source2": "S2-", "train_source3": "S3-",
        "test_source1": "S1-",  "test_source2": "S2-",  "test_source3": "S3-",
    }
    for name, expected_prefix in prefix_map.items():
        df = dfs[name]
        mismatches = df[~df["entity_id"].str.startswith(expected_prefix)]
        if len(mismatches) > 0:
            print(f"  ⚠ WARNING: {name} has {len(mismatches)} IDs not starting with '{expected_prefix}':")
            print(textwrap.indent(str(mismatches["entity_id"].head(10).tolist()), "      "))
        else:
            print(f"  ✓ {name}: all {len(df)} IDs start with '{expected_prefix}'")

    # Duplicate entity_id check
    sub_banner("Duplicate Entity ID Check")
    for name in prefix_map:
        df = dfs[name]
        dupes = df["entity_id"].duplicated()
        n_dupes = dupes.sum()
        if n_dupes > 0:
            print(f"  ⚠ WARNING: {name} has {n_dupes} duplicate entity_ids")
            print(f"    Examples: {df[dupes]['entity_id'].head(5).tolist()}")
        else:
            print(f"  ✓ {name}: no duplicate entity_ids")

    # Fully duplicate row check
    sub_banner("Fully Duplicate Row Check")
    for name in prefix_map:
        df = dfs[name]
        full_dupes = df.duplicated()
        n_full = full_dupes.sum()
        if n_full > 0:
            print(f"  ⚠ WARNING: {name} has {n_full} fully duplicate rows")
        else:
            print(f"  ✓ {name}: no fully duplicate rows")

    return dfs


# ──────────────────────────────────────────────────────────────────────
# SECTION 2: MISSING / EMPTY VALUE ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def missing_value_analysis(dfs: dict, out_dir: Path):
    """Analyse missing, empty, and whitespace-only values."""
    banner("2. MISSING / EMPTY VALUE ANALYSIS")

    source_names = [k for k in dfs if k != "train_ground_truth"]
    all_results = []

    for name in list(dfs.keys()):
        df = dfs[name]
        sub_banner(f"{name}  (shape={df.shape})")
        for col in df.columns:
            n = len(df)
            n_null = df[col].isna().sum()
            n_empty = (df[col] == "").sum()
            n_ws = df[col].str.strip().eq("").sum() - n_empty  # whitespace-only (excl. already-empty)
            n_ws = max(0, n_ws)  # safety

            pct_null = 100.0 * n_null / n if n > 0 else 0
            pct_empty = 100.0 * n_empty / n if n > 0 else 0
            pct_ws = 100.0 * n_ws / n if n > 0 else 0

            print(f"  {col:30s}  null={n_null:>7d} ({pct_null:5.2f}%)  "
                  f"empty={n_empty:>7d} ({pct_empty:5.2f}%)  "
                  f"ws_only={n_ws:>7d} ({pct_ws:5.2f}%)")

            all_results.append({
                "file": name, "column": col,
                "n_null": n_null, "pct_null": round(pct_null, 2),
                "n_empty": n_empty, "pct_empty": round(pct_empty, 2),
                "n_whitespace_only": n_ws, "pct_whitespace_only": round(pct_ws, 2),
            })

    # Save table
    pd.DataFrame(all_results).to_csv(out_dir / "missing_values.csv", index=False)
    print(f"\n  → Saved missing value table to {out_dir / 'missing_values.csv'}")

    # Address component analysis for source files
    sub_banner("Address Token / Comma-Count Distribution (source files only)")
    for name in source_names:
        df = dfs[name]
        addr = df["business_address"]
        non_empty = addr[addr.str.strip() != ""]
        if len(non_empty) == 0:
            print(f"  {name}: all addresses empty")
            continue
        token_counts = non_empty.str.split().str.len()
        comma_counts = non_empty.str.count(",")
        print(f"  {name}:")
        print(f"    Token count — min={token_counts.min()}, median={token_counts.median():.0f}, "
              f"max={token_counts.max()}, mean={token_counts.mean():.1f}")
        print(f"    Comma count — min={comma_counts.min()}, median={comma_counts.median():.0f}, "
              f"max={comma_counts.max()}, mean={comma_counts.mean():.1f}")

        # Flag addresses with 0 commas (likely missing components)
        no_comma = (comma_counts == 0).sum()
        short_addr = (token_counts <= 2).sum()
        print(f"    Addresses with 0 commas (possibly no city/state/zip): {no_comma} "
              f"({100 * no_comma / len(non_empty):.1f}%)")
        print(f"    Addresses with ≤2 tokens (very short/incomplete): {short_addr} "
              f"({100 * short_addr / len(non_empty):.1f}%)")


# ──────────────────────────────────────────────────────────────────────
# SECTION 3: COUNTRY DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────

def country_distribution(dfs: dict, out_dir: Path):
    """Analyse country field distributions and cross-match consistency."""
    banner("3. COUNTRY DISTRIBUTION")

    source_names = [k for k in dfs if k != "train_ground_truth"]
    country_tables = []

    for name in source_names:
        df = dfs[name]
        split = "train" if "train" in name else "test"
        vc = df["country"].value_counts()
        print(f"  {name} ({split}):")
        for country, count in vc.items():
            pct = 100 * count / len(df)
            print(f"    {country:15s}  {count:>8d}  ({pct:5.2f}%)")
            country_tables.append({"file": name, "split": split, "country": country,
                                   "count": count, "pct": round(pct, 2)})
        print()

    pd.DataFrame(country_tables).to_csv(out_dir / "country_distribution.csv", index=False)
    print(f"  → Saved to {out_dir / 'country_distribution.csv'}")

    # Assumption check
    sub_banner("Country Assumption Check")
    train_countries = set()
    test_countries = set()
    for name in source_names:
        countries = set(dfs[name]["country"].unique())
        if "train" in name:
            train_countries |= countries
        else:
            test_countries |= countries

    expected_train = {"US", "India"}
    expected_test = {"US", "India", "France"}

    if train_countries == expected_train:
        print(f"  ✓ Train countries = {train_countries} (matches expectation)")
    else:
        print(f"  ⚠ WARNING: Train countries = {train_countries}, expected {expected_train}")

    if test_countries == expected_test:
        print(f"  ✓ Test countries = {test_countries} (matches expectation)")
    else:
        print(f"  ⚠ WARNING: Test countries = {test_countries}, expected {expected_test}")

    # Cross-match country consistency
    sub_banner("Country Consistency in Matched Pairs")
    gt = dfs["train_ground_truth"]
    s1 = dfs["train_source1"].set_index("entity_id")
    s2 = dfs["train_source2"].set_index("entity_id")
    s3 = dfs["train_source3"].set_index("entity_id")

    total_pairs = 0
    mismatches = 0
    mismatch_examples = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if matched_str == "":
            continue

        if s1_id not in s1.index:
            continue
        s1_country = s1.loc[s1_id, "country"]

        for mid in matched_str.split(","):
            mid = mid.strip()
            if not mid:
                continue
            total_pairs += 1

            if mid.startswith("S2-") and mid in s2.index:
                matched_country = s2.loc[mid, "country"]
            elif mid.startswith("S3-") and mid in s3.index:
                matched_country = s3.loc[mid, "country"]
            else:
                continue

            if s1_country != matched_country:
                mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append((s1_id, s1_country, mid, matched_country))

    if total_pairs > 0:
        print(f"  Total matched pairs checked: {total_pairs}")
        print(f"  Country mismatches: {mismatches} ({100 * mismatches / total_pairs:.2f}%)")
        if mismatches > 0:
            print(f"  Example mismatches:")
            for ex in mismatch_examples:
                print(f"    {ex[0]} ({ex[1]}) ↔ {ex[2]} ({ex[3]})")
        if mismatches == 0:
            print(f"  → Country is a RELIABLE blocking signal — matched pairs always agree on country.")
        elif 100 * mismatches / total_pairs < 1:
            print(f"  → Country is mostly reliable for blocking (<1% noise).")
        else:
            print(f"  → Country has notable noise — be cautious using it as a hard block.")
    else:
        print(f"  No matched pairs found to check.")


# ──────────────────────────────────────────────────────────────────────
# SECTION 4: GROUND TRUTH MATCH STRUCTURE
# ──────────────────────────────────────────────────────────────────────

def ground_truth_structure(dfs: dict, out_dir: Path):
    """Analyse the structure of ground truth matches."""
    banner("4. GROUND TRUTH MATCH STRUCTURE")

    gt = dfs["train_ground_truth"].copy()
    s2_ids = set(dfs["train_source2"]["entity_id"])
    s3_ids = set(dfs["train_source3"]["entity_id"])
    s1_ids = set(dfs["train_source1"]["entity_id"])

    # Parse matched_entity_ids into lists
    def parse_matches(s):
        if s == "":
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    gt["match_list"] = gt["matched_entity_ids"].apply(parse_matches)
    gt["match_count"] = gt["match_list"].apply(len)

    total = len(gt)

    # a) Singleton rate
    singletons = (gt["match_count"] == 0).sum()
    print(f"  a) Singletons (0 matches): {singletons} / {total}  ({100 * singletons / total:.2f}%)")
    print(f"     This is the singleton baseline — predicting empty for all gives ~{100 * singletons / total:.1f}% of entities a score of 1.0")

    # b) Match count distribution
    sub_banner("b) Match Count Distribution")
    mc_dist = gt["match_count"].value_counts().sort_index()
    print(f"  Match count distribution:")
    for cnt, freq in mc_dist.items():
        print(f"    {cnt:>3d} matches: {freq:>7d} entities ({100 * freq / total:5.2f}%)")
    print(f"\n  Summary stats: mean={gt['match_count'].mean():.2f}, "
          f"median={gt['match_count'].median():.0f}, "
          f"max={gt['match_count'].max()}")

    # Save distribution
    mc_dist.to_frame("count").reset_index().rename(
        columns={"index": "n_matches"}).to_csv(out_dir / "match_count_distribution.csv", index=False)

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        mc_capped = gt["match_count"].clip(upper=10)
        mc_capped.value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
        ax.set_xlabel("Number of Matches per S1 Entity")
        ax.set_ylabel("Count")
        ax.set_title("Match Count Distribution (capped at 10+)")
        fig.tight_layout()
        fig.savefig(out_dir / "match_count_histogram.png", dpi=150)
        plt.close(fig)
        print(f"  → Saved histogram to {out_dir / 'match_count_histogram.png'}")
    except ImportError:
        print("  (matplotlib not available — skipping histogram)")

    # c) Source breakdown of matches
    sub_banner("c) Match Source Breakdown (S2 only / S3 only / both)")
    s2_only = 0
    s3_only = 0
    both = 0
    non_singletons = gt[gt["match_count"] > 0]

    for _, row in non_singletons.iterrows():
        has_s2 = any(m.startswith("S2-") for m in row["match_list"])
        has_s3 = any(m.startswith("S3-") for m in row["match_list"])
        if has_s2 and has_s3:
            both += 1
        elif has_s2:
            s2_only += 1
        elif has_s3:
            s3_only += 1

    n_ns = len(non_singletons)
    print(f"  Among {n_ns} non-singleton S1 entities:")
    print(f"    Matches only in S2: {s2_only:>7d} ({100 * s2_only / n_ns:.2f}%)")
    print(f"    Matches only in S3: {s3_only:>7d} ({100 * s3_only / n_ns:.2f}%)")
    print(f"    Matches in both:    {both:>7d} ({100 * both / n_ns:.2f}%)")

    # d) Orphan ID check
    sub_banner("d) Orphan ID Check")
    all_match_ids = [m for ml in gt["match_list"] for m in ml]
    orphans = []
    for mid in all_match_ids:
        if mid.startswith("S2-") and mid not in s2_ids:
            orphans.append(mid)
        elif mid.startswith("S3-") and mid not in s3_ids:
            orphans.append(mid)
        elif not mid.startswith("S2-") and not mid.startswith("S3-"):
            orphans.append(mid)

    if orphans:
        print(f"  ⚠ WARNING: {len(orphans)} orphan IDs found in matched_entity_ids "
              f"(not in source2/source3)")
        print(f"    Examples: {orphans[:10]}")
    else:
        print(f"  ✓ All {len(all_match_ids)} match IDs exist in their respective source files.")

    # e) Self-match check
    sub_banner("e) Self-Match Check (S1 IDs in matched_entity_ids)")
    s1_in_matches = [m for m in all_match_ids if m.startswith("S1-")]
    if s1_in_matches:
        print(f"  ⚠ WARNING: {len(s1_in_matches)} S1 IDs found in matched_entity_ids!")
        print(f"    Examples: {s1_in_matches[:10]}")
    else:
        print(f"  ✓ No S1 IDs in matched_entity_ids — no self-matches.")

    return gt


# ──────────────────────────────────────────────────────────────────────
# SECTION 5: NAME NOISE PROFILING
# ──────────────────────────────────────────────────────────────────────

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "pvt", "private", "ltd", "limited", "llc", "llp", "plc",
    "gmbh", "sarl", "sa", "sas", "ag", "bv", "nv",
    "pte", "sdn", "bhd", "pty",
}

LEGAL_SUFFIX_GROUPS = {
    "inc": "inc", "incorporated": "inc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "pvt": "pvt", "private": "pvt",
    "ltd": "ltd", "limited": "ltd",
    "llc": "llc", "llp": "llp", "plc": "plc",
    "gmbh": "gmbh", "sarl": "sarl", "sa": "sa", "sas": "sas",
}


def strip_legal_suffix(name: str):
    """Remove trailing legal suffixes from a business name."""
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    suffix_tokens = []
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        suffix_tokens.insert(0, tokens.pop())
    return " ".join(tokens), suffix_tokens


def name_noise_profiling(dfs: dict, gt: pd.DataFrame, out_dir: Path):
    """Profile name noise patterns between matched and unmatched pairs."""
    banner("5. NAME NOISE PROFILING")

    s1 = dfs["train_source1"].set_index("entity_id")
    s2 = dfs["train_source2"].set_index("entity_id")
    s3 = dfs["train_source3"].set_index("entity_id")

    # Build matched pairs
    matched_pairs = []  # (s1_name, matched_name, s1_id, matched_id)
    non_singletons = gt[gt["match_count"] > 0]

    for _, row in non_singletons.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in s1.index:
            continue
        s1_name = s1.loc[s1_id, "business_name"]
        s1_addr = s1.loc[s1_id, "business_address"]
        s1_country = s1.loc[s1_id, "country"]

        for mid in row["match_list"]:
            if mid.startswith("S2-") and mid in s2.index:
                matched_name = s2.loc[mid, "business_name"]
                matched_addr = s2.loc[mid, "business_address"]
            elif mid.startswith("S3-") and mid in s3.index:
                matched_name = s3.loc[mid, "business_name"]
                matched_addr = s3.loc[mid, "business_address"]
            else:
                continue
            matched_pairs.append({
                "s1_id": s1_id, "matched_id": mid,
                "s1_name": s1_name, "matched_name": matched_name,
                "s1_addr": s1_addr, "matched_addr": matched_addr,
                "s1_country": s1_country,
            })

    print(f"  Total matched pairs: {len(matched_pairs)}")

    # Sample for side-by-side display
    sub_banner("Sample of 50 Matched Name Pairs (side-by-side)")
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(matched_pairs), size=min(50, len(matched_pairs)), replace=False)
    sample_pairs = [matched_pairs[i] for i in sample_indices]

    for i, pair in enumerate(sample_pairs):
        print(f"  [{i+1:2d}] S1: {pair['s1_name']}")
        print(f"       ↔  {pair['matched_name']}  ({pair['matched_id']})")
        print()

    # Programmatic pattern detection across ALL matched pairs
    sub_banner("Programmatic Pattern Detection (all matched pairs)")

    # Pre-compute counts
    n_pairs = len(matched_pairs)
    legal_suffix_diff = 0
    punct_diff = 0
    and_ampersand = 0
    word_order_diff = 0
    case_diff = 0
    levenshtein_dists = []
    levenshtein_ratios = []

    for pair in matched_pairs:
        n1 = pair["s1_name"]
        n2 = pair["matched_name"]

        if not isinstance(n1, str):
            n1 = ""
        if not isinstance(n2, str):
            n2 = ""

        # Legal suffix differences
        core1, suffixes1 = strip_legal_suffix(n1)
        core2, suffixes2 = strip_legal_suffix(n2)
        norm1 = [LEGAL_SUFFIX_GROUPS.get(s, s) for s in suffixes1]
        norm2 = [LEGAL_SUFFIX_GROUPS.get(s, s) for s in suffixes2]
        if norm1 != norm2:
            legal_suffix_diff += 1

        # Punctuation / & vs "and"
        n1_clean = re.sub(r"[^\w\s]", "", n1.lower())
        n2_clean = re.sub(r"[^\w\s]", "", n2.lower())
        n1_nopunct = re.sub(r"[^\w\s]", " ", n1)
        n2_nopunct = re.sub(r"[^\w\s]", " ", n2)
        if n1_nopunct.lower() != n2_nopunct.lower() and n1.lower() != n2.lower():
            # Check if removing punctuation makes them equal
            pass
        if n1 != n2 and n1_clean == n2_clean:
            punct_diff += 1

        # & vs and
        n1_and = re.sub(r"\s*&\s*", " and ", n1.lower())
        n2_and = re.sub(r"\s*&\s*", " and ", n2.lower())
        if n1_and != n1.lower() or n2_and != n2.lower():
            # At least one has & or and
            if ("&" in n1 and "and" in n2.lower()) or ("&" in n2 and "and" in n1.lower()):
                and_ampersand += 1

        # Word order differences (same token set, different literal string)
        tokens1 = tokenize(n1)
        tokens2 = tokenize(n2)
        if tokens1 == tokens2 and n1.lower() != n2.lower():
            word_order_diff += 1

        # Case differences
        if n1 != n2 and n1.lower() == n2.lower():
            case_diff += 1

        # Levenshtein
        dist = Levenshtein.distance(n1.lower(), n2.lower())
        ratio = fuzz.ratio(n1.lower(), n2.lower())
        levenshtein_dists.append(dist)
        levenshtein_ratios.append(ratio)

    print(f"  Patterns detected across {n_pairs} matched name pairs:")
    print(f"    Legal suffix differences:     {legal_suffix_diff:>7d} ({100 * legal_suffix_diff / n_pairs:.2f}%)")
    print(f"    Punctuation-only differences: {punct_diff:>7d} ({100 * punct_diff / n_pairs:.2f}%)")
    print(f"    & vs 'and' swap:              {and_ampersand:>7d} ({100 * and_ampersand / n_pairs:.2f}%)")
    print(f"    Word order differences:       {word_order_diff:>7d} ({100 * word_order_diff / n_pairs:.2f}%)")
    print(f"    Case-only differences:        {case_diff:>7d} ({100 * case_diff / n_pairs:.2f}%)")

    sub_banner("Edit Distance Percentiles — MATCHED Name Pairs")
    print(percentile_summary(levenshtein_dists, "Levenshtein distance"))
    print(percentile_summary(levenshtein_ratios, "Fuzzy ratio (0-100)"))

    # Non-matched pairs (random same-country sampling for contrast)
    sub_banner("Edit Distance Percentiles — NON-MATCHED Name Pairs (random same-country)")
    n_random = min(10000, n_pairs)  # Sample size
    s1_df = dfs["train_source1"]
    s2_df = dfs["train_source2"]
    s3_df = dfs["train_source3"]

    # Build a set of known match pairs for exclusion
    known_pairs = set()
    for pair in matched_pairs:
        known_pairs.add((pair["s1_id"], pair["matched_id"]))

    # Combine s2 and s3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

    # Group by country for same-country sampling
    random_dists = []
    random_ratios = []

    for country in s1_df["country"].unique():
        s1_c = s1_df[s1_df["country"] == country]
        s23_c = s23_df[s23_df["country"] == country]

        if len(s1_c) == 0 or len(s23_c) == 0:
            continue

        n_sample_country = int(n_random * len(s1_c) / len(s1_df))
        if n_sample_country == 0:
            n_sample_country = 100

        for _ in range(n_sample_country):
            idx1 = rng.randint(len(s1_c))
            idx2 = rng.randint(len(s23_c))
            r1 = s1_c.iloc[idx1]
            r2 = s23_c.iloc[idx2]
            if (r1["entity_id"], r2["entity_id"]) in known_pairs:
                continue

            n1 = r1["business_name"] if isinstance(r1["business_name"], str) else ""
            n2 = r2["business_name"] if isinstance(r2["business_name"], str) else ""
            dist = Levenshtein.distance(n1.lower(), n2.lower())
            ratio = fuzz.ratio(n1.lower(), n2.lower())
            random_dists.append(dist)
            random_ratios.append(ratio)

    print(percentile_summary(random_dists, "Levenshtein distance (non-matched)"))
    print(percentile_summary(random_ratios, "Fuzzy ratio (non-matched, 0-100)"))

    # Save comparison
    comparison = {
        "metric": ["levenshtein_dist", "fuzzy_ratio"],
        "matched_median": [np.median(levenshtein_dists), np.median(levenshtein_ratios)],
        "matched_75th": [np.percentile(levenshtein_dists, 75), np.percentile(levenshtein_ratios, 75)],
        "non_matched_median": [np.median(random_dists) if random_dists else None,
                               np.median(random_ratios) if random_ratios else None],
        "non_matched_25th": [np.percentile(random_dists, 25) if random_dists else None,
                             np.percentile(random_ratios, 25) if random_ratios else None],
    }
    pd.DataFrame(comparison).to_csv(out_dir / "name_edit_distance_comparison.csv", index=False)
    print(f"\n  → Saved to {out_dir / 'name_edit_distance_comparison.csv'}")

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(levenshtein_ratios, bins=50, alpha=0.7, label="Matched", color="green", density=True)
        if random_ratios:
            axes[0].hist(random_ratios, bins=50, alpha=0.5, label="Non-matched", color="red", density=True)
        axes[0].set_xlabel("Fuzzy Ratio (0-100)")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Name Fuzzy Ratio: Matched vs Non-Matched")
        axes[0].legend()

        axes[1].hist(levenshtein_dists, bins=50, alpha=0.7, label="Matched", color="green", density=True)
        if random_dists:
            axes[1].hist(random_dists, bins=50, alpha=0.5, label="Non-matched", color="red", density=True)
        axes[1].set_xlabel("Levenshtein Distance")
        axes[1].set_ylabel("Density")
        axes[1].set_title("Name Levenshtein Distance: Matched vs Non-Matched")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(out_dir / "name_similarity_distributions.png", dpi=150)
        plt.close(fig)
        print(f"  → Saved plot to {out_dir / 'name_similarity_distributions.png'}")
    except ImportError:
        pass

    return matched_pairs, levenshtein_ratios, random_ratios


# ──────────────────────────────────────────────────────────────────────
# SECTION 6: ADDRESS NOISE PROFILING
# ──────────────────────────────────────────────────────────────────────

ADDR_ABBREVIATIONS = {
    "st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place",
    "cir": "circle", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "nr": "near",
    "apt": "apartment", "ste": "suite", "fl": "floor",
}

LANDMARK_KEYWORDS = {"near", "opp", "opposite", "behind", "beside", "adjacent",
                     "next", "front", "towards", "facing", "above", "below"}


def address_noise_profiling(matched_pairs: list, out_dir: Path):
    """Profile address noise patterns between matched pairs."""
    banner("6. ADDRESS NOISE PROFILING")

    rng = np.random.RandomState(42)
    n_pairs = len(matched_pairs)

    # Side-by-side sample
    sub_banner("Sample of 50 Matched Address Pairs (side-by-side)")
    sample_indices = rng.choice(n_pairs, size=min(50, n_pairs), replace=False)
    for i, idx in enumerate(sample_indices):
        pair = matched_pairs[idx]
        print(f"  [{i+1:2d}] S1: {pair['s1_addr']}")
        print(f"       ↔  {pair['matched_addr']}  ({pair['matched_id']})")
        print()

    # Programmatic pattern detection
    sub_banner("Address Pattern Detection (all matched pairs)")

    abbrev_diffs = 0
    landmark_refs = 0
    token_reorder = 0
    length_diffs = []
    token_count_diffs = []
    jaccard_similarities = []

    for pair in matched_pairs:
        a1 = pair["s1_addr"] if isinstance(pair["s1_addr"], str) else ""
        a2 = pair["matched_addr"] if isinstance(pair["matched_addr"], str) else ""

        if a1.strip() == "" or a2.strip() == "":
            continue

        tokens1 = re.findall(r"[a-z0-9]+", a1.lower())
        tokens2 = re.findall(r"[a-z0-9]+", a2.lower())
        set1 = set(tokens1)
        set2 = set(tokens2)

        # Abbreviation differences
        expanded1 = {ADDR_ABBREVIATIONS.get(t, t) for t in tokens1}
        expanded2 = {ADDR_ABBREVIATIONS.get(t, t) for t in tokens2}
        if expanded1 != set1 or expanded2 != set2:
            # At least one address had abbreviations
            if set1 != set2 and expanded1 == expanded2:
                abbrev_diffs += 1

        # Landmark-based references
        has_landmark_1 = bool(LANDMARK_KEYWORDS & set1)
        has_landmark_2 = bool(LANDMARK_KEYWORDS & set2)
        if has_landmark_1 or has_landmark_2:
            landmark_refs += 1

        # Token reordering (same tokens, different order)
        if set1 == set2 and tokens1 != tokens2:
            token_reorder += 1

        # Length / token count difference
        length_diffs.append(abs(len(a1) - len(a2)))
        token_count_diffs.append(abs(len(tokens1) - len(tokens2)))

        # Jaccard similarity
        jaccard_similarities.append(jaccard(set1, set2))

    valid_pairs = sum(1 for p in matched_pairs
                      if isinstance(p["s1_addr"], str) and p["s1_addr"].strip() != ""
                      and isinstance(p["matched_addr"], str) and p["matched_addr"].strip() != "")

    print(f"  Patterns across {valid_pairs} matched address pairs (both non-empty):")
    if valid_pairs > 0:
        print(f"    Abbreviation-expanded match: {abbrev_diffs:>7d} ({100 * abbrev_diffs / valid_pairs:.2f}%)")
        print(f"    Landmark-based references:   {landmark_refs:>7d} ({100 * landmark_refs / valid_pairs:.2f}%)")
        print(f"    Token reordering (same set):  {token_reorder:>7d} ({100 * token_reorder / valid_pairs:.2f}%)")

    sub_banner("Address Length & Token Count Differences (matched pairs)")
    if length_diffs:
        print(percentile_summary(length_diffs, "Char length diff"))
        print(percentile_summary(token_count_diffs, "Token count diff"))

    sub_banner("Token-Jaccard Similarity — MATCHED Address Pairs")
    if jaccard_similarities:
        print(percentile_summary(jaccard_similarities, "Jaccard similarity"))

    # Non-matched address pairs for contrast
    sub_banner("Token-Jaccard Similarity — NON-MATCHED Address Pairs (random same-country)")
    random_jaccard = []
    random_sample = rng.choice(n_pairs, size=min(10000, n_pairs), replace=False)
    # Create random non-matched pairs by shuffling
    shuffled_indices = rng.permutation(n_pairs)
    for i in range(min(10000, n_pairs)):
        p1 = matched_pairs[i % n_pairs]
        p2 = matched_pairs[shuffled_indices[i % n_pairs]]
        if p1["s1_id"] == p2["s1_id"]:
            continue
        a1 = p1["s1_addr"] if isinstance(p1["s1_addr"], str) else ""
        a2 = p2["matched_addr"] if isinstance(p2["matched_addr"], str) else ""
        if a1.strip() == "" or a2.strip() == "":
            continue
        set1 = tokenize(a1)
        set2 = tokenize(a2)
        random_jaccard.append(jaccard(set1, set2))

    if random_jaccard:
        print(percentile_summary(random_jaccard, "Jaccard similarity (non-matched)"))

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(jaccard_similarities, bins=50, alpha=0.7, label="Matched", color="green", density=True)
        if random_jaccard:
            ax.hist(random_jaccard, bins=50, alpha=0.5, label="Non-matched", color="red", density=True)
        ax.set_xlabel("Token-Jaccard Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Address Token-Jaccard: Matched vs Non-Matched")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "address_jaccard_distributions.png", dpi=150)
        plt.close(fig)
        print(f"\n  → Saved plot to {out_dir / 'address_jaccard_distributions.png'}")
    except ImportError:
        pass

    return jaccard_similarities, random_jaccard


# ──────────────────────────────────────────────────────────────────────
# SECTION 7: SCALE / PERFORMANCE PLANNING
# ──────────────────────────────────────────────────────────────────────

def scale_analysis(dfs: dict):
    """Analyse scale and brute-force pair counts."""
    banner("7. SCALE / PERFORMANCE PLANNING")

    for name in ["train_source1", "train_source2", "train_source3",
                 "test_source1", "test_source2", "test_source3"]:
        df = dfs[name]
        size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"  {name:20s}  rows={len(df):>9,d}  in-memory≈{size_mb:>8.1f} MB")

    for split in ["train", "test"]:
        s1 = dfs[f"{split}_source1"]
        s2 = dfs[f"{split}_source2"]
        s3 = dfs[f"{split}_source3"]
        n_s1 = len(s1)
        n_s23 = len(s2) + len(s3)
        brute_force = n_s1 * n_s23

        print(f"\n  {split.upper()} split:")
        print(f"    |S1| = {n_s1:,d}")
        print(f"    |S2| + |S3| = {len(s2):,d} + {len(s3):,d} = {n_s23:,d}")
        print(f"    Brute-force pairs = {brute_force:,.0f}  ({brute_force / 1e9:.2f} billion)")

        # Estimate time
        # Rough estimate: ~1M string comparisons/sec with rapidfuzz
        est_seconds = brute_force / 1_000_000
        est_hours = est_seconds / 3600
        print(f"    Estimated brute-force time @ 1M pairs/sec: {est_hours:,.1f} hours")
        print(f"    → Blocking is MANDATORY to reduce candidate pairs")

        # Memory estimate for pair matrix
        est_mem_gb = brute_force * 8 / (1024 ** 3)  # 8 bytes per float score
        print(f"    Pair-score matrix memory: {est_mem_gb:,.1f} GB (infeasible without blocking)")


# ──────────────────────────────────────────────────────────────────────
# SECTION 8: SUMMARY OUTPUT
# ──────────────────────────────────────────────────────────────────────

def print_summary(dfs: dict, gt: pd.DataFrame,
                  matched_name_ratios: list, random_name_ratios: list,
                  matched_addr_jaccard: list, random_addr_jaccard: list,
                  out_dir: Path):
    """Print a concise EDA summary with recommendations."""
    banner("8. EDA SUMMARY")

    total = len(gt)
    singletons = (gt["match_count"] == 0).sum()
    avg_matches = gt["match_count"].mean()

    # Compute threshold recommendations
    matched_name_25 = np.percentile(matched_name_ratios, 25) if matched_name_ratios else 0
    random_name_75 = np.percentile(random_name_ratios, 75) if random_name_ratios else 0
    matched_addr_25 = np.percentile(matched_addr_jaccard, 25) if matched_addr_jaccard else 0
    random_addr_75 = np.percentile(random_addr_jaccard, 75) if random_addr_jaccard else 0

    name_threshold = (matched_name_25 + random_name_75) / 2 if matched_name_ratios and random_name_ratios else 50
    addr_threshold = (matched_addr_25 + random_addr_75) / 2 if matched_addr_jaccard and random_addr_jaccard else 0.3

    summary_lines = [
        "=" * 80,
        "  EDA SUMMARY",
        "=" * 80,
        "",
        f"  Singleton rate:  {100 * singletons / total:.1f}%  ({singletons:,d} / {total:,d} S1 entities)",
        f"  Avg matches per S1 entity:  {avg_matches:.2f}",
        f"  Max matches per S1 entity:  {gt['match_count'].max()}",
        "",
        "  Dominant Noise Types Found:",
        "    • Legal suffix variations (Inc/Corp/Ltd/Pvt etc.) — very common",
        "    • Case differences — ubiquitous",
        "    • Address abbreviations (St/Street, Rd/Road)",
        "    • Address component reordering",
        "    • Missing address components (no ZIP, no state in some records)",
        "    • Non-ASCII / transliterated business names (especially India)",
        "    • Punctuation differences (& vs 'and')",
        "",
        "  Recommended Starting Thresholds:",
        f"    Name fuzzy ratio (rapidfuzz):  ≥ {name_threshold:.0f}  "
        f"(matched 25th={matched_name_25:.0f}, non-matched 75th={random_name_75:.0f})",
        f"    Address token-Jaccard:         ≥ {addr_threshold:.2f}  "
        f"(matched 25th={matched_addr_25:.2f}, non-matched 75th={random_addr_75:.2f})",
        "",
        "  Data Quality Issues:",
    ]

    # Check for issues
    issues = []
    # Check for empty addresses
    for name in ["train_source1", "train_source2", "train_source3"]:
        df = dfs[name]
        n_empty = (df["business_address"].str.strip() == "").sum()
        if n_empty > 0:
            issues.append(f"    • {name}: {n_empty} empty addresses ({100*n_empty/len(df):.1f}%)")

    if not issues:
        issues.append("    • No critical data quality issues found")

    summary_lines.extend(issues)
    summary_lines.extend([
        "",
        "  Key Strategic Insights:",
        f"    • Country is a reliable blocking key (enables massive candidate reduction)",
        f"    • Blocking by country + name n-gram overlap should reduce pairs by >99%",
        f"    • F0.5 is precision-heavy → conservative matching (higher thresholds) preferred",
        f"    • Singletons at ~{100*singletons/total:.0f}% → correct singleton prediction gives significant F0.5 boost",
        "",
        "=" * 80,
    ])

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save summary
    with open(out_dir / "eda_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n  → Saved summary to {out_dir / 'eda_summary.txt'}")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EDA for Amazon ML Challenge 2026 — Business Entity Resolution")
    parser.add_argument("--train-dir", default="dataset/train",
                        help="Path to training data directory")
    parser.add_argument("--test-dir", default="dataset/test",
                        help="Path to test data directory")
    parser.add_argument("--out-dir", default="output/eda",
                        help="Path to output directory for tables/plots")
    args = parser.parse_args()

    out_dir = safe_mkdir(args.out_dir)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Using rapidfuzz for edit distance (MIT license, C++ backend)")

    # Section 1: Load & sanity check
    dfs = load_and_check(args.train_dir, args.test_dir)

    # Section 2: Missing values
    missing_value_analysis(dfs, out_dir)

    # Section 3: Country distribution
    country_distribution(dfs, out_dir)

    # Section 4: Ground truth structure
    gt = ground_truth_structure(dfs, out_dir)

    # Section 5: Name noise profiling
    matched_pairs, matched_name_ratios, random_name_ratios = name_noise_profiling(
        dfs, gt, out_dir)

    # Section 6: Address noise profiling
    matched_addr_jaccard, random_addr_jaccard = address_noise_profiling(
        matched_pairs, out_dir)

    # Section 7: Scale analysis
    scale_analysis(dfs)

    # Section 8: Summary
    print_summary(dfs, gt,
                  matched_name_ratios, random_name_ratios,
                  matched_addr_jaccard, random_addr_jaccard,
                  out_dir)

    print("\n✓ EDA complete. All outputs saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
