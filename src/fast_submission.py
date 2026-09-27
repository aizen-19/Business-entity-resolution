#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Fast Submission Generator (src/fast_submission.py)

Generates an immediate, high-quality competition submission file in under 2 minutes
by ranking blocked candidate pairs using vectorized token overlap and fuzzy metrics,
applying precision thresholding and singleton detection.

Output: output/submission.tsv (TAB-separated, matches ground truth format)
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rapidfuzz import fuzz

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def generate_fast_submission(
    test_candidates_path: str = "output/candidates/test_candidates.parquet",
    norm_dir: str = "output/normalized",
    test_s1_path: str = "dataset/test/test_source1.tsv",
    output_path: str = "output/submission.tsv",
    min_name_sim: float = 65.0,
    max_matches_per_entity: int = 5,
):
    print(f"\n{'='*80}")
    print(f"  FAST COMPETITION SUBMISSION GENERATOR")
    print(f"{'='*80}\n")
    t0 = time.time()

    # 1. Load test S1 list to guarantee 100% complete ordering
    print(f"Loading test S1 entity list from {test_s1_path} ...")
    test_s1_raw = pd.read_csv(test_s1_path, sep="\t", dtype=str, keep_default_na=False)
    all_s1_ids = test_s1_raw["entity_id"].values
    n_s1 = len(all_s1_ids)
    print(f"  Total test S1 entities: {n_s1:,d}")

    # 2. Load compact normalized string dictionaries for rapid lookup
    print(f"Loading normalized text lookups ...")
    s1_norm = pd.read_parquet(Path(norm_dir) / "test_source1.parquet", columns=["entity_id", "name_core", "address_core"]).set_index("entity_id")
    s2_norm = pd.read_parquet(Path(norm_dir) / "test_source2.parquet", columns=["entity_id", "name_core", "address_core"])
    s3_norm = pd.read_parquet(Path(norm_dir) / "test_source3.parquet", columns=["entity_id", "name_core", "address_core"])
    s23_norm = pd.concat([s2_norm, s3_norm], ignore_index=True).set_index("entity_id")
    del s2_norm, s3_norm

    s1_name_dict = s1_norm["name_core"].to_dict()
    s1_addr_dict = s1_norm["address_core"].to_dict()
    s23_name_dict = s23_norm["name_core"].to_dict()
    s23_addr_dict = s23_norm["address_core"].to_dict()
    del s1_norm, s23_norm

    # 3. Read test candidates
    print(f"Loading candidate pairs from {test_candidates_path} ...")
    cands_df = pd.read_parquet(test_candidates_path, columns=["source1_entity_id", "candidate_entity_id", "hit_strategy_a", "hit_strategy_b"])
    print(f"  Total candidate pairs: {len(cands_df):,d}")

    # 4. Group candidate pairs by S1 entity
    print(f"Evaluating candidate matches per S1 entity ...")
    cands_grouped = cands_df.groupby("source1_entity_id")

    matched_dict = {}
    total_links = 0
    singleton_count = 0
    multi_count = 0

    processed = 0
    for s1_id, grp in cands_grouped:
        processed += 1
        s1_name = s1_name_dict.get(s1_id, "")
        s1_addr = s1_addr_dict.get(s1_id, "")

        scored_cands = []
        for cand_id, hit_a, hit_b in zip(grp["candidate_entity_id"], grp["hit_strategy_a"], grp["hit_strategy_b"]):
            cand_name = s23_name_dict.get(cand_id, "")
            if not s1_name or not cand_name:
                continue

            # Fast fuzzy name matching
            name_score = fuzz.token_set_ratio(s1_name, cand_name)
            if name_score < min_name_sim:
                continue

            cand_addr = s23_addr_dict.get(cand_id, "")
            addr_score = 0
            if s1_addr and cand_addr:
                addr_score = fuzz.token_set_ratio(s1_addr, cand_addr)
                final_score = 0.65 * name_score + 0.35 * addr_score
            else:
                final_score = name_score

            # Bonus for multi-strategy hits
            if hit_a and hit_b:
                final_score += 5.0

            scored_cands.append((cand_id, final_score))

        if scored_cands:
            # Sort by score descending
            scored_cands.sort(key=lambda x: -x[1])
            # Keep matches within 15 points of top match
            top_score = scored_cands[0][1]
            selected = [c[0] for c in scored_cands if c[1] >= max(top_score - 15.0, min_name_sim)][:max_matches_per_entity]
            matched_dict[s1_id] = selected
            total_links += len(selected)
            if len(selected) > 1:
                multi_count += 1

        if processed % 300000 == 0:
            print(f"  Processed {processed:>10,d} / {n_s1:,d} entities ({100*processed/n_s1:.1f}%) ...")

    # 5. Build submission dataframe
    print(f"\nBuilding final submission dataframe ...")
    submission_rows = []
    for s1_id in all_s1_ids:
        matches = matched_dict.get(s1_id, [])
        if not matches:
            submission_rows.append("")
            singleton_count += 1
        else:
            submission_rows.append(",".join(matches))

    sub_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": submission_rows,
    })

    # Save to disk
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, sep="\t", index=False)
    print(f"\n✓ Saved submission file to {output_path} ({Path(output_path).stat().st_size/1e6:.1f} MB)")

    # Also save as matching_results.tsv
    alt_path = Path(output_path).parent / "matching_results.tsv"
    sub_df.to_csv(alt_path, sep="\t", index=False)
    print(f"✓ Saved copy to {alt_path}")

    print(f"\n{'━'*70}")
    print(f"  FAST SUBMISSION SUMMARY")
    print(f"{'━'*70}")
    print(f"  Total S1 rows:             {len(sub_df):>10,d} (Exact match: {len(sub_df) == n_s1})")
    print(f"  Singletons (0 matches):    {singleton_count:>10,d} ({100*singleton_count/n_s1:.2f}%)")
    print(f"  Multi-match entities:      {multi_count:>10,d} ({100*multi_count/n_s1:.2f}%)")
    print(f"  Total predicted links:     {total_links:>10,d}")
    print(f"  Avg matches per S1 entity: {total_links/n_s1:.2f}")
    print(f"  Total Elapsed Time:        {time.time()-t0:.1f}s")
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    generate_fast_submission()
