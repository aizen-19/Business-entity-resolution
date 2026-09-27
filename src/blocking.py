#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 3: High-Performance Multi-Strategy Blocking (src/blocking.py)

Fast multi-strategy candidate generator using compact inverted indices:
  - Strategy A: Country + Name Token Inverted Index
  - Strategy B: Country + Address Token Inverted Index (digits & selective street tokens)
  - Strategy C: Country + Character 3-Gram Inverted Index (typos & non-latin transliterations)

Outputs:
  - output/candidates/{train,test}_candidates.parquet
  - output/candidate_pairs.tsv (submission format)
  - Ground-truth recall validation and segment breakdowns for train split
"""

import argparse
import gc
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STOPWORDS = {
    "and", "the", "for", "with", "inc", "corp", "ltd", "pvt", "llc", "co", "sa", "sarl", "sas",
    "services", "solutions", "enterprises", "company", "group", "holdings", "industries",
    "international", "national", "global", "consulting", "management", "associates",
    "technologies", "technology", "systems", "products", "center", "centre", "club",
    "near", "opp", "opposite", "behind", "beside", "road", "rd", "street", "st", "avenue", "ave",
    "floor", "fl", "unit", "suite", "ste", "apartment", "apt", "building", "bldg", "block", "blk",
    "sector", "sec", "phase", "ph", "plot", "no", "number", "india", "us", "usa", "france",
    "state", "city", "district", "dist", "po", "box",
}


def get_trigrams(text: str) -> set[str]:
    """Extract character 3-grams from a string."""
    if not isinstance(text, str) or len(text) < 3:
        return set()
    cleaned = text.strip()
    return {cleaned[i:i+3] for i in range(len(cleaned) - 2)}


def run_blocking_for_country(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    country: str,
    max_candidates: int = 60,
    max_doc_freq: int = 5000,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run fast multi-strategy inverted-index blocking for a country partition."""
    n_s1 = len(s1_df)
    n_s23 = len(s23_df)
    print(f"\n  [{country}] Partition: |S1| = {n_s1:,d}, |S23| = {n_s23:,d}")

    if n_s1 == 0 or n_s23 == 0:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "hit_strategy_a", "hit_strategy_b", "hit_strategy_c"
        ]), {}

    s1_ids = s1_df["entity_id"].values
    s23_ids = s23_df["entity_id"].values

    # ── 1. Build Inverted Indices on S23 ──
    t0 = time.time()
    idx_name = defaultdict(list)
    idx_addr = defaultdict(list)
    idx_tri  = defaultdict(list)

    s23_name_tokens = s23_df["name_token_set"].values
    s23_addr_tokens = s23_df["address_token_set"].values
    s23_addr_empty = s23_df["address_is_empty"].values
    s23_name_core = s23_df["name_core"].values
    s23_non_latin = s23_df["name_has_non_latin"].values

    for i in range(n_s23):
        # Strategy A: Name Tokens
        n_tok = s23_name_tokens[i]
        if n_tok:
            for t in n_tok.split("|"):
                if len(t) >= 2 and t not in STOPWORDS:
                    idx_name[t].append(i)

        # Strategy B: Address Tokens (numbers + street tokens)
        if not s23_addr_empty[i]:
            a_tok = s23_addr_tokens[i]
            if a_tok:
                for t in a_tok.split("|"):
                    if t not in STOPWORDS and (t.isdigit() or len(t) >= 4):
                        idx_addr[t].append(i)

        # Strategy C: Trigrams (for non-latin or names >= 4 chars)
        if s23_non_latin[i]:
            n_core = s23_name_core[i]
            if n_core:
                for tri in get_trigrams(n_core):
                    idx_tri[tri].append(i)

    # Prune high-frequency tokens and convert to compact numpy arrays
    clean_idx_name = {k: np.array(v, dtype=np.uint32) for k, v in idx_name.items() if len(v) <= max_doc_freq}
    clean_idx_addr = {k: np.array(v, dtype=np.uint32) for k, v in idx_addr.items() if len(v) <= max_doc_freq}
    clean_idx_tri  = {k: np.array(v, dtype=np.uint32) for k, v in idx_tri.items() if len(v) <= max_doc_freq}
    del idx_name, idx_addr, idx_tri
    gc.collect()

    print(f"    Indices built in {time.time()-t0:.1f}s (Name keys: {len(clean_idx_name):,d}, Addr keys: {len(clean_idx_addr):,d}, Tri keys: {len(clean_idx_tri):,d})")

    # ── 2. Query Inverted Indices for all S1 entities ──
    t0 = time.time()
    s1_name_tokens = s1_df["name_token_set"].values
    s1_addr_tokens = s1_df["address_token_set"].values
    s1_addr_empty = s1_df["address_is_empty"].values
    s1_name_core = s1_df["name_core"].values
    s1_non_latin = s1_df["name_has_non_latin"].values

    s1_pair_list = []
    s23_pair_list = []
    hit_a_list = []
    hit_b_list = []
    hit_c_list = []
    tsv_dict: dict[str, str] = {}

    for i in range(n_s1):
        s1_id = s1_ids[i]
        cands_a_arr = []
        cands_b_arr = []
        cands_c_arr = []

        # Strategy A: Name Tokens
        n_tok = s1_name_tokens[i]
        if n_tok:
            for t in n_tok.split("|"):
                if len(t) >= 2 and t in clean_idx_name:
                    cands_a_arr.append(clean_idx_name[t])

        # Strategy B: Address Tokens
        if not s1_addr_empty[i]:
            a_tok = s1_addr_tokens[i]
            if a_tok:
                for t in a_tok.split("|"):
                    if (t.isdigit() or len(t) >= 4) and t in clean_idx_addr:
                        cands_b_arr.append(clean_idx_addr[t])

        # Strategy C: Trigrams (if non-latin or 0 candidates from A+B)
        if s1_non_latin[i] or (not cands_a_arr and not cands_b_arr):
            n_core = s1_name_core[i]
            if n_core:
                for tri in get_trigrams(n_core):
                    if tri in clean_idx_tri:
                        cands_c_arr.append(clean_idx_tri[tri])

        if not cands_a_arr and not cands_b_arr and not cands_c_arr:
            tsv_dict[s1_id] = ""
            continue

        # Convert to sets to track strategy hit provenance
        set_a = set(np.concatenate(cands_a_arr)) if cands_a_arr else set()
        set_b = set(np.concatenate(cands_b_arr)) if cands_b_arr else set()
        set_c = set(np.concatenate(cands_c_arr)) if cands_c_arr else set()

        all_matches = cands_a_arr + cands_b_arr + cands_c_arr
        concat_matches = np.concatenate(all_matches)
        uniq_cands, counts = np.unique(concat_matches, return_counts=True)

        # Rank and cap top candidates
        if len(uniq_cands) > max_candidates:
            top_indices = np.argsort(-counts)[:max_candidates]
            uniq_cands = uniq_cands[top_indices]

        # Extract IDs and strategy hit flags
        cand_id_strings = []
        for cand_idx in uniq_cands:
            cand_id = s23_ids[cand_idx]
            s1_pair_list.append(s1_id)
            s23_pair_list.append(cand_id)
            hit_a_list.append(cand_idx in set_a)
            hit_b_list.append(cand_idx in set_b)
            hit_c_list.append(cand_idx in set_c)
            cand_id_strings.append(cand_id)

        tsv_dict[s1_id] = ",".join(cand_id_strings)

    elapsed = time.time() - t0
    print(f"    Queried {n_s1:,d} entities in {elapsed:.1f}s ({n_s1/elapsed:,.0f} queries/sec). Generated {len(s1_pair_list):,d} candidate pairs.")

    res_df = pd.DataFrame({
        "source1_entity_id": s1_pair_list,
        "candidate_entity_id": s23_pair_list,
        "hit_strategy_a": hit_a_list,
        "hit_strategy_b": hit_b_list,
        "hit_strategy_c": hit_c_list,
    })
    return res_df, tsv_dict


def run_blocking(
    split: str = "train",
    norm_dir: str = "output/normalized",
    out_dir: str = "output",
    max_candidates: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute multi-strategy blocking for train or test split."""
    print(f"\n{'='*80}")
    print(f"  PHASE 3: BLOCKING — {split.upper()} SPLIT")
    print(f"{'='*80}\n")

    t_start = time.time()
    cols = ["entity_id", "country", "name_token_set", "name_core", "name_has_non_latin", "address_token_set", "address_is_empty"]

    p_s1 = Path(norm_dir) / f"{split}_source1.parquet"
    p_s2 = Path(norm_dir) / f"{split}_source2.parquet"
    p_s3 = Path(norm_dir) / f"{split}_source3.parquet"

    print(f"Loading normalized data from {norm_dir} ...")
    s1_df = pd.read_parquet(p_s1, columns=cols)
    s2_df = pd.read_parquet(p_s2, columns=cols)
    s3_df = pd.read_parquet(p_s3, columns=cols)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()

    print(f"  Total S1 rows:  {len(s1_df):,d}")
    print(f"  Total S23 rows: {len(s23_df):,d}")

    countries = sorted(s1_df["country"].unique())
    country_dfs = []
    full_tsv_dict: dict[str, str] = {}

    for c in countries:
        s1_c = s1_df[s1_df["country"] == c].reset_index(drop=True)
        s23_c = s23_df[s23_df["country"] == c].reset_index(drop=True)
        c_df, c_dict = run_blocking_for_country(s1_c, s23_c, country=c, max_candidates=max_candidates)
        country_dfs.append(c_df)
        full_tsv_dict.update(c_dict)

    candidates_df = pd.concat(country_dfs, ignore_index=True)
    del country_dfs, s23_df
    gc.collect()

    all_s1_ids = s1_df["entity_id"].values
    tsv_strings = [full_tsv_dict.get(s1_id, "") for s1_id in all_s1_ids]
    cand_counts = [len(s.split(",")) if s else 0 for s in tsv_strings]

    tsv_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "candidate_entity_ids": tsv_strings,
    })

    # Summary Diagnostics
    zero_cands = sum(1 for c in cand_counts if c == 0)
    print(f"\n--- Blocking Summary ({split.upper()}) ---")
    print(f"  Total S1 entities:         {len(all_s1_ids):>10,d}")
    print(f"  Singletons (0 candidates): {zero_cands:>10,d}  ({100*zero_cands/len(all_s1_ids):.2f}%)")
    print(f"  Total Candidate Pairs:     {len(candidates_df):>10,d}")
    print(f"  Candidates per S1 — mean: {np.mean(cand_counts):.2f}, median: {np.median(cand_counts):.0f}, max: {np.max(cand_counts):,d}")

    # Save candidates parquet
    cand_dir = Path(out_dir) / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = cand_dir / f"{split}_candidates.parquet"
    candidates_df.to_parquet(out_parquet, index=False)
    print(f"  ✓ Saved candidate pairs to {out_parquet} ({out_parquet.stat().st_size/1e6:.1f} MB)")

    # Save candidate_pairs.tsv
    out_tsv = Path(out_dir) / f"{split}_candidate_pairs.tsv" if split == "train" else Path(out_dir) / "candidate_pairs.tsv"
    tsv_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  ✓ Saved {out_tsv} ({out_tsv.stat().st_size/1e6:.1f} MB)")

    print(f"  Phase 3 ({split}) finished in {time.time()-t_start:.1f}s\n")
    return candidates_df, tsv_df


def validate_blocking_recall(
    candidates_df: pd.DataFrame,
    train_dir: str = "dataset/train",
    norm_dir: str = "output/normalized",
):
    """Validate blocking recall against ground truth on the training split."""
    print(f"\n{'='*80}")
    print(f"  PHASE 3: GROUND TRUTH BLOCKING RECALL VALIDATION")
    print(f"{'='*80}\n")

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        print(f"Ground truth not found at {gt_path} — skipping validation.")
        return

    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    s1_norm = pd.read_parquet(Path(norm_dir) / "train_source1.parquet", columns=["entity_id", "country", "name_has_non_latin", "address_is_empty"]).set_index("entity_id")

    print(f"Building candidate pair lookup index ({len(candidates_df):,d} pairs) ...")
    # Group candidates by S1 entity for fast set lookups
    cands_by_s1 = candidates_df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    total_true_pairs = 0
    recalled_pairs = 0

    by_country_true = Counter()
    by_country_rec = Counter()
    by_source_true = Counter()
    by_source_rec = Counter()
    by_non_latin_true = Counter()
    by_non_latin_rec = Counter()
    by_addr_empty_true = Counter()
    by_addr_empty_rec = Counter()

    print(f"Evaluating recall across {len(gt):,d} ground truth entities ...")
    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if not matched_str:
            continue

        s1_country = s1_norm.loc[s1_id, "country"] if s1_id in s1_norm.index else "Unknown"
        s1_non_latin = s1_norm.loc[s1_id, "name_has_non_latin"] if s1_id in s1_norm.index else False
        s1_addr_empty = s1_norm.loc[s1_id, "address_is_empty"] if s1_id in s1_norm.index else False
        s1_cands = cands_by_s1.get(s1_id, set())

        for mid in matched_str.split(","):
            mid = mid.strip()
            if not mid:
                continue

            total_true_pairs += 1
            src = "S2" if mid.startswith("S2-") else "S3"

            by_country_true[s1_country] += 1
            by_source_true[src] += 1
            by_non_latin_true[s1_non_latin] += 1
            by_addr_empty_true[s1_addr_empty] += 1

            if mid in s1_cands:
                recalled_pairs += 1
                by_country_rec[s1_country] += 1
                by_source_rec[src] += 1
                by_non_latin_rec[s1_non_latin] += 1
                by_addr_empty_rec[s1_addr_empty] += 1

    overall_recall = (recalled_pairs / total_true_pairs) * 100 if total_true_pairs > 0 else 0

    print(f"\n{'━'*70}")
    print(f"  ★ OVERALL BLOCKING RECALL:  {overall_recall:.3f}%  ({recalled_pairs:,d} / {total_true_pairs:,d} true links captured)")
    print(f"{'━'*70}\n")

    print(f"  Recall Breakdown by Country:")
    for c in sorted(by_country_true.keys()):
        tot = by_country_true[c]
        rec = by_country_rec[c]
        print(f"    {c:15s}: {100*rec/tot:6.2f}%  ({rec:,d} / {tot:,d})")

    print(f"\n  Recall Breakdown by Source:")
    for s in ["S2", "S3"]:
        tot = by_source_true[s]
        rec = by_source_rec[s]
        print(f"    {s:15s}: {100*rec/tot:6.2f}%  ({rec:,d} / {tot:,d})")

    print(f"\n  Recall Breakdown by Non-Latin Script:")
    for flag in [False, True]:
        tot = by_non_latin_true[flag]
        rec = by_non_latin_rec[flag]
        if tot > 0:
            print(f"    Non-Latin={str(flag):5s}: {100*rec/tot:6.2f}%  ({rec:,d} / {tot:,d})")

    print(f"\n  Recall Breakdown by S1 Address Empty:")
    for flag in [False, True]:
        tot = by_addr_empty_true[flag]
        rec = by_addr_empty_rec[flag]
        if tot > 0:
            print(f"    Addr Empty={str(flag):5s}: {100*rec/tot:6.2f}%  ({rec:,d} / {tot:,d})")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Candidate Generation & Blocking")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--norm-dir", default="output/normalized")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--max-candidates", type=int, default=60)
    args = parser.parse_args()

    if args.split in ("train", "both"):
        train_cands, _ = run_blocking(
            split="train",
            norm_dir=args.norm_dir,
            out_dir=args.out_dir,
            max_candidates=args.max_candidates,
        )
        validate_blocking_recall(
            train_cands,
            train_dir=args.train_dir,
            norm_dir=args.norm_dir,
        )

    if args.split in ("test", "both"):
        run_blocking(
            split="test",
            norm_dir=args.norm_dir,
            out_dir=args.out_dir,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
