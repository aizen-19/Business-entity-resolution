#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
High-Throughput LightGBM Test Inference (src/inference_lgbm.py)

1. Loads trained LightGBM booster and calibrated threshold T*.
2. Streams test candidates from output/candidates/test_candidates.parquet.
3. Computes features on the fly in vectorized C-backed chunks.
4. Predicts match probabilities, applies calibrated threshold and singleton logic.
5. Emits competition-grade output/submission.tsv and matching_results.tsv.
"""

import argparse
import gc
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
from rapidfuzz import fuzz

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


def compute_token_jaccard_batch(tokens1_arr: np.ndarray, tokens2_arr: np.ndarray) -> np.ndarray:
    n = len(tokens1_arr)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1, s2 = tokens1_arr[i], tokens2_arr[i]
        if s1 and s2:
            set1 = set(s1.split("|"))
            set2 = set(s2.split("|"))
            u = len(set1 | set2)
            if u > 0:
                res[i] = len(set1 & set2) / u
    return res


def compute_trigram_jaccard_batch(text1_arr: np.ndarray, text2_arr: np.ndarray) -> np.ndarray:
    n = len(text1_arr)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t1, t2 = text1_arr[i], text2_arr[i]
        if t1 and t2 and len(t1) >= 3 and len(t2) >= 3:
            tri1 = {t1[j:j+3] for j in range(len(t1) - 2)}
            tri2 = {t2[j:j+3] for j in range(len(t2) - 2)}
            u = len(tri1 | tri2)
            if u > 0:
                res[i] = len(tri1 & tri2) / u
    return res


def compute_fuzzy_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1, s2 = arr1[i], arr2[i]
        if s1 and s2:
            res[i] = fuzz.ratio(s1, s2)
    return res


def compute_token_sort_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1, s2 = arr1[i], arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_sort_ratio(s1, s2)
    return res


def compute_token_set_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1, s2 = arr1[i], arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_set_ratio(s1, s2)
    return res


def extract_features_df(chunk: pd.DataFrame) -> pd.DataFrame:
    s1_nc = chunk["s1_name_core"].fillna("").values
    cand_nc = chunk["cand_name_core"].fillna("").values
    s1_nt = chunk["s1_name_token_set"].fillna("").values
    cand_nt = chunk["cand_name_token_set"].fillna("").values

    name_fuzzy = compute_fuzzy_ratio_batch(s1_nc, cand_nc)
    name_tok_sort = compute_token_sort_ratio_batch(s1_nc, cand_nc)
    name_tok_set = compute_token_set_ratio_batch(s1_nc, cand_nc)
    name_jaccard = compute_token_jaccard_batch(s1_nt, cand_nt)
    name_tri_jaccard = compute_trigram_jaccard_batch(s1_nc, cand_nc)

    s1_sfx = chunk["s1_legal_suffix"].fillna("").values
    cand_sfx = chunk["cand_legal_suffix"].fillna("").values
    legal_sfx_match = (s1_sfx == cand_sfx) & (s1_sfx != "")

    s1_len = np.array([len(s) for s in s1_nc], dtype=np.int16)
    cand_len = np.array([len(s) for s in cand_nc], dtype=np.int16)
    name_len_diff = np.abs(s1_len - cand_len)

    name_non_latin = chunk["s1_name_has_non_latin"].fillna(False).values | chunk["cand_name_has_non_latin"].fillna(False).values
    was_domain = chunk["s1_was_domain_format"].fillna(False).values | chunk["cand_was_domain_format"].fillna(False).values

    s1_ae = chunk["s1_address_is_empty"].fillna(True).values
    cand_ae = chunk["cand_address_is_empty"].fillna(True).values
    addr_empty_either = s1_ae | cand_ae
    addr_empty_both = s1_ae & cand_ae

    s1_ac = chunk["s1_address_core"].fillna("").values
    cand_ac = chunk["cand_address_core"].fillna("").values
    s1_at = chunk["s1_address_token_set"].fillna("").values
    cand_at = chunk["cand_address_token_set"].fillna("").values

    raw_addr_fuzzy = compute_fuzzy_ratio_batch(s1_ac, cand_ac)
    raw_addr_tok_set = compute_token_set_ratio_batch(s1_ac, cand_ac)
    raw_addr_jaccard = compute_token_jaccard_batch(s1_at, cand_at)

    addr_fuzzy = np.where(addr_empty_either, -1.0, raw_addr_fuzzy).astype(np.float32)
    addr_tok_set = np.where(addr_empty_either, -1.0, raw_addr_tok_set).astype(np.float32)
    addr_jaccard = np.where(addr_empty_either, -1.0, raw_addr_jaccard).astype(np.float32)

    s1_lm = chunk["s1_address_landmark"].fillna("").values
    cand_lm = chunk["cand_address_landmark"].fillna("").values
    landmark_present = (s1_lm != "") | (cand_lm != "")

    s1_alen = np.array([len(s) for s in s1_ac], dtype=np.int16)
    cand_alen = np.array([len(s) for s in cand_ac], dtype=np.int16)
    addr_len_diff = np.where(addr_empty_either, -1, np.abs(s1_alen - cand_alen)).astype(np.int16)

    country_match = (chunk["s1_country"].fillna("").values == chunk["cand_country"].fillna("").values)
    cand_ids = chunk["candidate_entity_id"].values
    is_source3 = np.array([cid.startswith("S3-") for cid in cand_ids], dtype=bool)

    prelim_score = np.where(
        addr_jaccard >= 0,
        0.6 * name_jaccard + 0.4 * addr_jaccard,
        name_jaccard,
    ).astype(np.float32)

    feat_dict = {
        "source1_entity_id": chunk["source1_entity_id"].values,
        "candidate_entity_id": cand_ids,
        "name_fuzzy_ratio": name_fuzzy,
        "name_token_sort_ratio": name_tok_sort,
        "name_token_set_ratio": name_tok_set,
        "name_jaccard": name_jaccard,
        "name_char_ngram_jaccard": name_tri_jaccard,
        "legal_suffix_match": legal_sfx_match.astype(np.int8),
        "name_len_diff": name_len_diff,
        "name_has_non_latin_either": name_non_latin.astype(np.int8),
        "was_domain_format_either": was_domain.astype(np.int8),
        "address_is_empty_either": addr_empty_either.astype(np.int8),
        "address_is_empty_both": addr_empty_both.astype(np.int8),
        "address_fuzzy_ratio": addr_fuzzy,
        "address_jaccard": addr_jaccard,
        "address_token_set_ratio": addr_tok_set,
        "landmark_present_either": landmark_present.astype(np.int8),
        "address_len_diff": addr_len_diff,
        "country_match": country_match.astype(np.int8),
        "is_source3": is_source3.astype(np.int8),
        "hit_strategy_a": chunk["hit_strategy_a"].astype(np.int8).values,
        "hit_strategy_b": chunk["hit_strategy_b"].astype(np.int8).values,
        "hit_strategy_c": chunk["hit_strategy_c"].astype(np.int8).values,
        "prelim_combined_score": prelim_score,
    }
    return pd.DataFrame(feat_dict)


def main():
    parser = argparse.ArgumentParser(description="Streaming LightGBM Test Inference")
    parser.add_argument("--candidates", default="output/candidates/test_candidates.parquet")
    parser.add_argument("--test-s1", default="dataset/test/test_source1.tsv")
    parser.add_argument("--norm-dir", default="output/normalized")
    parser.add_argument("--model-path", default="output/models/lgbm_model.txt")
    parser.add_argument("--meta-path", default="output/models/model_metadata.json")
    parser.add_argument("--output-path", default="output/submission.tsv")
    parser.add_argument("--chunk-size", type=int, default=1000000)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  LIGHTGBM STREAMING INFERENCE & SUBMISSION GENERATION")
    print(f"{'='*80}\n")
    t_start = time.time()

    # 1. Load Model & Threshold
    print(f"Loading LightGBM model booster from {args.model_path} ...")
    bst = lgb.Booster(model_file=args.model_path)

    threshold = 0.45
    if os.path.exists(args.meta_path):
        with open(args.meta_path) as f:
            meta = json.load(f)
            threshold = meta.get("best_threshold", 0.45)
            print(f"Loaded calibrated threshold: T = {threshold:.4f}")

    # 2. Load Test S1 Raw Query List
    print(f"Loading test queries from {args.test_s1} ...")
    test_s1 = pd.read_csv(args.test_s1, sep="\t", dtype=str, keep_default_na=False)
    all_s1_ids = test_s1["entity_id"].values
    n_s1 = len(all_s1_ids)
    print(f"  Total test queries: {n_s1:,d}")

    # 3. Load Normalized Lookups
    cols_to_load = [
        "entity_id", "country", "legal_suffix", "name_core", "name_token_set",
        "name_has_non_latin", "was_domain_format", "address_core", "address_landmark",
        "address_is_empty", "address_token_set"
    ]
    print("Loading normalized source tables ...")
    s1_df = pd.read_parquet(Path(args.norm_dir) / "test_source1.parquet", columns=cols_to_load).set_index("entity_id")
    s2_df = pd.read_parquet(Path(args.norm_dir) / "test_source2.parquet", columns=cols_to_load)
    s3_df = pd.read_parquet(Path(args.norm_dir) / "test_source3.parquet", columns=cols_to_load)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True).set_index("entity_id")
    del s2_df, s3_df
    gc.collect()

    # 4. Stream Candidates, Compute Features, and Score
    print(f"\nStreaming test candidate batches from {args.candidates} ...")
    parquet_file = pq.ParquetFile(args.candidates)
    matched_by_s1 = defaultdict(list)
    total_scored = 0

    for b_idx, batch in enumerate(parquet_file.iter_batches(batch_size=args.chunk_size)):
        t0 = time.time()
        chunk = batch.to_pandas()
        total_scored += len(chunk)

        # Join normalized features
        chunk = chunk.join(s1_df.add_prefix("s1_"), on="source1_entity_id")
        chunk = chunk.join(s23_df.add_prefix("cand_"), on="candidate_entity_id")

        feats_chunk = extract_features_df(chunk)
        X = feats_chunk[FEATURE_COLS]
        probs = bst.predict(X)

        # Retain candidate predictions >= threshold
        mask = probs >= threshold
        if mask.any():
            p_s1 = feats_chunk.loc[mask, "source1_entity_id"].values
            p_cand = feats_chunk.loc[mask, "candidate_entity_id"].values
            p_prob = probs[mask]
            for s1_id, c_id, pr in zip(p_s1, p_cand, p_prob):
                matched_by_s1[s1_id].append((c_id, float(pr)))

        elapsed = time.time() - t0
        speed = len(chunk) / max(elapsed, 0.001)
        print(f"  Batch {b_idx+1:>2d} ({len(chunk):>8,d} pairs) scored in {elapsed:4.1f}s ({speed:,.0f} pairs/s) | Total: {total_scored:>10,d}")

    del s1_df, s23_df, parquet_file
    gc.collect()

    # 5. Build Final Submission TSV
    print(f"\nConstructing final submission for all {n_s1:,d} test entities ...")
    submission_rows = []
    singleton_count = 0
    multi_count = 0
    total_links = 0

    for s1_id in all_s1_ids:
        matches = matched_by_s1.get(s1_id, [])
        if not matches:
            submission_rows.append("")
            singleton_count += 1
        else:
            # Sort by probability descending
            matches.sort(key=lambda x: -x[1])
            cand_ids = [m[0] for m in matches[:6]]
            submission_rows.append(",".join(cand_ids))
            total_links += len(cand_ids)
            if len(cand_ids) > 1:
                multi_count += 1

    sub_df = pd.DataFrame({
        "source1_entity_id": all_s1_ids,
        "matched_entity_ids": submission_rows,
    })

    # Save final submission
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(args.output_path, sep="\t", index=False)
    print(f"\n✓ Saved LightGBM submission to {args.output_path} ({Path(args.output_path).stat().st_size/1e6:.1f} MB)")

    # Save copy as matching_results.tsv
    alt_path = Path(args.output_path).parent / "matching_results.tsv"
    sub_df.to_csv(alt_path, sep="\t", index=False)
    print(f"✓ Saved copy to {alt_path}")

    # Submission Report
    print(f"\n{'━'*70}")
    print(f"  LIGHTGBM FINAL SUBMISSION REPORT")
    print(f"{'━'*70}")
    print(f"  Total S1 rows:             {len(sub_df):>10,d} (Exact match: {len(sub_df) == n_s1})")
    print(f"  Singletons (0 matches):    {singleton_count:>10,d} ({100*singleton_count/n_s1:5.2f}%)")
    print(f"  Multi-match entities:      {multi_count:>10,d} ({100*multi_count/n_s1:5.2f}%)")
    print(f"  Total predicted links:     {total_links:>10,d}")
    print(f"  Avg matches per S1 entity: {total_links/n_s1:5.2f}")
    print(f"  Total Elapsed Time:        {time.time()-t_start:.1f}s")
    print(f"{'━'*70}\n")


if __name__ == "__main__":
    main()
