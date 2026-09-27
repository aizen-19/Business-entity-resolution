#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 6: Streaming Test Inference & Final Submission Generator (src/inference.py)

Loads trained LightGBM model and calibrated decision threshold, scores all
test candidate pairs, applies post-processing (singleton handling, ranking, thresholding),
and generates the final competition-compliant TSV submission file.

Outputs:
  - output/submission.tsv (and matching_results.tsv)
  - Submission integrity checks and statistics
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

FEATURE_COLS = [
    "name_fuzzy_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaccard",
    "name_char_ngram_jaccard",
    "legal_suffix_match",
    "name_len_diff",
    "name_has_non_latin_either",
    "was_domain_format_either",
    "address_is_empty_either",
    "address_is_empty_both",
    "address_fuzzy_ratio",
    "address_jaccard",
    "address_token_set_ratio",
    "landmark_present_either",
    "address_len_diff",
    "country_match",
    "is_source3",
    "hit_strategy_a",
    "hit_strategy_b",
    "hit_strategy_c",
    "prelim_combined_score",
]


def run_inference(
    test_features_path: str = "output/features/test_features.parquet",
    test_source1_path: str = "dataset/test/test_source1.tsv",
    model_path: str = "output/models/lgbm_model.txt",
    meta_path: str = "output/models/model_metadata.json",
    output_path: str = "output/submission.tsv",
    threshold_override: float = None,
    chunk_size: int = 1000000,
):
    """Score test candidate pairs and generate final submission TSV."""
    print(f"\n{'='*80}")
    print(f"  PHASE 6: INFERENCE & SUBMISSION GENERATION")
    print(f"{'='*80}\n")

    t_start = time.time()

    # Load model and metadata
    print(f"Loading LightGBM booster from {model_path} ...")
    bst = lgb.Booster(model_file=model_path)

    threshold = 0.50
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            threshold = meta.get("best_threshold", 0.50)
            print(f"Loaded calibrated optimal threshold: T = {threshold:.4f}")
    if threshold_override is not None:
        threshold = threshold_override
        print(f"Overridden threshold: T = {threshold:.4f}")

    # Load full list of S1 entity IDs in exact test order
    print(f"Loading test S1 entities from {test_source1_path} ...")
    test_s1_df = pd.read_csv(test_source1_path, sep="\t", dtype=str, keep_default_na=False)
    all_s1_ids = test_s1_df["entity_id"].values
    n_s1 = len(all_s1_ids)
    print(f"  Total test S1 entities to resolve: {n_s1:,d}")

    # Streaming score predictions per S1 entity
    print(f"Streaming and scoring test feature pairs from {test_features_path} ...")
    parquet_file = pq.ParquetFile(test_features_path)
    total_pairs_scored = 0
    s1_matched_dict = defaultdict(list)

    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
        t0 = time.time()
        chunk_df = batch.to_pandas()
        total_pairs_scored += len(chunk_df)

        # Cast booleans to int8
        for col in FEATURE_COLS:
            if chunk_df[col].dtype == bool:
                chunk_df[col] = chunk_df[col].astype(np.int8)

        X_chunk = chunk_df[FEATURE_COLS]
        probs = bst.predict(X_chunk)

        # Filter candidates above threshold
        mask = probs >= threshold
        if mask.any():
            passed_s1 = chunk_df.loc[mask, "source1_entity_id"].values
            passed_cand = chunk_df.loc[mask, "candidate_entity_id"].values
            passed_probs = probs[mask]

            for s1_id, cand_id, prob in zip(passed_s1, passed_cand, passed_probs):
                s1_matched_dict[s1_id].append((cand_id, prob))

        print(f"  Scored batch {batch_idx+1:>2d} ({len(chunk_df):>8,d} pairs) in {time.time()-t0:.1f}s | Total: {total_pairs_scored:>10,d}")

    del parquet_file
    print(f"\nConstructing submission rows for all {n_s1:,d} S1 entities ...")

    matched_strings = []
    singleton_count = 0
    multi_match_count = 0
    total_links_predicted = 0

    for s1_id in all_s1_ids:
        matches = s1_matched_dict.get(s1_id, [])
        if not matches:
            matched_strings.append("")
            singleton_count += 1
        else:
            # Sort matches by predicted probability descending, then candidate ID
            matches.sort(key=lambda x: (-x[1], x[0]))
            cand_ids = [m[0] for m in matches]
            matched_strings.append(",".join(cand_ids))
            total_links_predicted += len(cand_ids)
            if len(cand_ids) > 1:
                multi_match_count += 1

    submission_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": matched_strings,
    })

    # Save final submission
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, sep="\t", index=False)
    print(f"\n✓ Saved final submission to {output_path} ({Path(output_path).stat().st_size/1e6:.1f} MB)")

    # Also save as matching_results.tsv for competition convention
    alt_path = Path(output_path).parent / "matching_results.tsv"
    submission_df.to_csv(alt_path, sep="\t", index=False)
    print(f"✓ Saved copy to {alt_path}")

    # ── Submission Quality & Sanity Diagnostics ──
    print(f"\n{'━'*70}")
    print(f"  SUBMISSION INTEGRITY & SANITY REPORT")
    print(f"{'━'*70}")
    print(f"  Total S1 rows:             {len(submission_df):>10,d} (Exact match with test_source1: {len(submission_df) == n_s1})")
    print(f"  Singletons (0 matches):    {singleton_count:>10,d} ({100*singleton_count/n_s1:5.2f}%)")
    print(f"  Single-match entities:     {n_s1 - singleton_count - multi_match_count:>10,d} ({100*(n_s1 - singleton_count - multi_match_count)/n_s1:5.2f}%)")
    print(f"  Multi-match entities:      {multi_match_count:>10,d} ({100*multi_match_count/n_s1:5.2f}%)")
    print(f"  Total predicted links:     {total_links_predicted:>10,d}")
    print(f"  Avg matches per S1 entity: {total_links_predicted/n_s1:5.2f}")
    print(f"  Missing values check:      {submission_df['source1_entity_id'].isna().sum()} NaNs in S1 ID")
    print(f"{'━'*70}\n")

    print(f"Phase 6 Inference completed in {time.time()-t_start:.1f}s\n")
    return submission_df


def main():
    parser = argparse.ArgumentParser(description="Phase 6 — Inference & Submission Generation")
    parser.add_argument("--features", default="output/features/test_features.parquet")
    parser.add_argument("--test-s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--model", default="output/models/lgbm_model.txt")
    parser.add_argument("--meta", default="output/models/model_metadata.json")
    parser.add_argument("--output", default="output/submission.tsv")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    args = parser.parse_args()

    run_inference(
        test_features_path=args.features,
        test_source1_path=args.test_s1,
        model_path=args.model,
        meta_path=args.meta,
        output_path=args.output,
        threshold_override=args.threshold,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
