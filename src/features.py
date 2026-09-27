#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 4: Feature Engineering (src/features.py)

Extracts discriminative string similarity, token overlap, structural, and
heuristic features for all surviving (S1, candidate) pairs.
Processes in chunked batches with rapidfuzz's C-accelerated string functions.

Outputs:
  - output/features/train_features.parquet (with binary label column)
  - output/features/test_features.parquet (unlabeled, for inference)
  - Feature summary statistics and class separation diagnostics
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def compute_token_jaccard(tokens1_arr: np.ndarray, tokens2_arr: np.ndarray) -> np.ndarray:
    """Compute token Jaccard similarity across two arrays of pipe-separated token strings."""
    n = len(tokens1_arr)
    jaccard_vals = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s1 = tokens1_arr[i]
        s2 = tokens2_arr[i]
        if not s1 or not s2:
            continue
        set1 = set(s1.split("|"))
        set2 = set(s2.split("|"))
        union_len = len(set1 | set2)
        if union_len > 0:
            jaccard_vals[i] = len(set1 & set2) / union_len

    return jaccard_vals


def compute_trigram_jaccard(text1_arr: np.ndarray, text2_arr: np.ndarray) -> np.ndarray:
    """Compute character 3-gram Jaccard similarity across two text arrays."""
    n = len(text1_arr)
    jaccard_vals = np.zeros(n, dtype=np.float32)

    for i in range(n):
        t1 = text1_arr[i]
        t2 = text2_arr[i]
        if not t1 or not t2 or len(t1) < 3 or len(t2) < 3:
            continue
        tri1 = {t1[j:j+3] for j in range(len(t1) - 2)}
        tri2 = {t2[j:j+3] for j in range(len(t2) - 2)}
        union_len = len(tri1 | tri2)
        if union_len > 0:
            jaccard_vals[i] = len(tri1 & tri2) / union_len

    return jaccard_vals


def compute_batch_fuzzy_ratio(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.ratio(s1, s2)
    return res


def compute_batch_token_sort_ratio(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz token_sort_ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_sort_ratio(s1, s2)
    return res


def compute_batch_token_set_ratio(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz token_set_ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_set_ratio(s1, s2)
    return res


def extract_features_chunk(chunk_df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
    """Extract all feature columns for a chunk of paired records."""
    # ── 1. Name Features ──
    s1_name_core = chunk_df["s1_name_core"].fillna("").values
    cand_name_core = chunk_df["cand_name_core"].fillna("").values
    s1_name_tok = chunk_df["s1_name_token_set"].fillna("").values
    cand_name_tok = chunk_df["cand_name_token_set"].fillna("").values

    name_fuzzy = compute_batch_fuzzy_ratio(s1_name_core, cand_name_core)
    name_tok_sort = compute_batch_token_sort_ratio(s1_name_core, cand_name_core)
    name_tok_set = compute_batch_token_set_ratio(s1_name_core, cand_name_core)
    name_jaccard = compute_token_jaccard(s1_name_tok, cand_name_tok)
    name_tri_jaccard = compute_trigram_jaccard(s1_name_core, cand_name_core)

    s1_sfx = chunk_df["s1_legal_suffix"].fillna("").values
    cand_sfx = chunk_df["cand_legal_suffix"].fillna("").values
    legal_sfx_match = (s1_sfx == cand_sfx)

    s1_name_len = np.array([len(s) for s in s1_name_core], dtype=np.int16)
    cand_name_len = np.array([len(s) for s in cand_name_core], dtype=np.int16)
    name_len_diff = np.abs(s1_name_len - cand_name_len)

    name_non_latin = chunk_df["s1_name_has_non_latin"].values | chunk_df["cand_name_has_non_latin"].values
    was_domain = chunk_df["s1_was_domain_format"].values | chunk_df["cand_was_domain_format"].values

    # ── 2. Address Features ──
    s1_addr_empty = chunk_df["s1_address_is_empty"].values
    cand_addr_empty = chunk_df["cand_address_is_empty"].values
    addr_empty_either = s1_addr_empty | cand_addr_empty
    addr_empty_both = s1_addr_empty & cand_addr_empty

    s1_addr_core = chunk_df["s1_address_core"].fillna("").values
    cand_addr_core = chunk_df["cand_address_core"].fillna("").values
    s1_addr_tok = chunk_df["s1_address_token_set"].fillna("").values
    cand_addr_tok = chunk_df["cand_address_token_set"].fillna("").values

    # Compute address string metrics
    raw_addr_fuzzy = compute_batch_fuzzy_ratio(s1_addr_core, cand_addr_core)
    raw_addr_tok_set = compute_batch_token_set_ratio(s1_addr_core, cand_addr_core)
    raw_addr_jaccard = compute_token_jaccard(s1_addr_tok, cand_addr_tok)

    # Sentinel -1.0 for missing addresses so model learns absence vs mismatch
    addr_fuzzy = np.where(addr_empty_either, -1.0, raw_addr_fuzzy)
    addr_tok_set = np.where(addr_empty_either, -1.0, raw_addr_tok_set)
    addr_jaccard = np.where(addr_empty_either, -1.0, raw_addr_jaccard)

    s1_landmark = chunk_df["s1_address_landmark"].fillna("").values
    cand_landmark = chunk_df["cand_address_landmark"].fillna("").values
    landmark_present = (s1_landmark != "") | (cand_landmark != "")

    s1_addr_len = np.array([len(s) for s in s1_addr_core], dtype=np.int16)
    cand_addr_len = np.array([len(s) for s in cand_addr_core], dtype=np.int16)
    addr_len_diff = np.where(addr_empty_either, -1, np.abs(s1_addr_len - cand_addr_len))

    # ── 3. Cross & Structural Features ──
    s1_country = chunk_df["s1_country"].values
    cand_country = chunk_df["cand_country"].values
    country_match = (s1_country == cand_country)

    cand_ids = chunk_df["candidate_entity_id"].values
    is_source3 = np.array([cid.startswith("S3-") for cid in cand_ids], dtype=bool)

    hit_a = chunk_df["hit_strategy_a"].values
    hit_b = chunk_df["hit_strategy_b"].values
    hit_c = chunk_df["hit_strategy_c"].values

    # Preliminary heuristic score
    prelim_score = np.where(
        addr_jaccard >= 0,
        0.6 * name_jaccard + 0.4 * addr_jaccard,
        name_jaccard,
    )

    feat_dict = {
        "source1_entity_id": chunk_df["source1_entity_id"].values,
        "candidate_entity_id": cand_ids,
        # Name features
        "name_fuzzy_ratio": name_fuzzy,
        "name_token_sort_ratio": name_tok_sort,
        "name_token_set_ratio": name_tok_set,
        "name_jaccard": name_jaccard,
        "name_char_ngram_jaccard": name_tri_jaccard,
        "legal_suffix_match": legal_sfx_match,
        "name_len_diff": name_len_diff,
        "name_has_non_latin_either": name_non_latin,
        "was_domain_format_either": was_domain,
        # Address features
        "address_is_empty_either": addr_empty_either,
        "address_is_empty_both": addr_empty_both,
        "address_fuzzy_ratio": addr_fuzzy,
        "address_jaccard": addr_jaccard,
        "address_token_set_ratio": addr_tok_set,
        "landmark_present_either": landmark_present,
        "address_len_diff": addr_len_diff,
        # Structural & hit flags
        "country_match": country_match,
        "is_source3": is_source3,
        "hit_strategy_a": hit_a,
        "hit_strategy_b": hit_b,
        "hit_strategy_c": hit_c,
        "prelim_combined_score": prelim_score,
    }

    if is_train and "label" in chunk_df.columns:
        feat_dict["label"] = chunk_df["label"].values

    return pd.DataFrame(feat_dict)


def build_features(
    split: str = "train",
    cand_path: str = None,
    norm_dir: str = "output/normalized",
    train_dir: str = "dataset/train",
    out_dir: str = "output/features",
    chunk_size: int = 500000,
) -> pd.DataFrame:
    """Build full feature dataset for train or test candidate pairs."""
    print(f"\n{'='*80}")
    print(f"  PHASE 4: FEATURE ENGINEERING — {split.upper()} SPLIT")
    print(f"{'='*80}\n")

    t_start = time.time()
    if cand_path is None:
        cand_path = f"output/candidates/{split}_candidates.parquet"

    print(f"Loading candidate pairs from {cand_path} ...")
    cands_df = pd.read_parquet(cand_path)
    total_pairs = len(cands_df)
    print(f"  Total candidate pairs: {total_pairs:,d}")

    # Load normalized source tables
    cols_to_load = [
        "entity_id", "country", "legal_suffix", "name_core", "name_token_set",
        "name_has_non_latin", "was_domain_format", "address_core", "address_landmark",
        "address_is_empty", "address_token_set"
    ]
    print(f"Loading normalized source records ...")
    s1_df = pd.read_parquet(Path(norm_dir) / f"{split}_source1.parquet", columns=cols_to_load).set_index("entity_id")
    s2_df = pd.read_parquet(Path(norm_dir) / f"{split}_source2.parquet", columns=cols_to_load)
    s3_df = pd.read_parquet(Path(norm_dir) / f"{split}_source3.parquet", columns=cols_to_load)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True).set_index("entity_id")
    del s2_df, s3_df
    gc.collect()

    # Ground truth lookup for training split
    true_pairs_set = set()
    is_train = (split == "train")
    if is_train:
        gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
        print(f"Loading ground truth labels from {gt_path} ...")
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
        for _, row in gt.iterrows():
            s1_id = row["source1_entity_id"]
            m_str = row["matched_entity_ids"]
            if m_str:
                for mid in m_str.split(","):
                    if mid.strip():
                        true_pairs_set.add((s1_id, mid.strip()))
        print(f"  Loaded {len(true_pairs_set):,d} true match pairs.")

    # Process candidates in chunks
    out_features_list = []
    n_chunks = int(np.ceil(total_pairs / chunk_size))
    print(f"Extracting features in {n_chunks} chunks of ~{chunk_size:,d} pairs ...")

    for c_idx in range(n_chunks):
        t0 = time.time()
        start = c_idx * chunk_size
        end = min(start + chunk_size, total_pairs)
        chunk = cands_df.iloc[start:end].copy()

        # Join S1 records
        chunk = chunk.join(s1_df.add_prefix("s1_"), on="source1_entity_id")
        # Join Candidate records
        chunk = chunk.join(s23_df.add_prefix("cand_"), on="candidate_entity_id")

        if is_train:
            # Assign binary label: 1 if true link, 0 otherwise
            chunk_pairs = list(zip(chunk["source1_entity_id"], chunk["candidate_entity_id"]))
            chunk["label"] = np.array([1 if p in true_pairs_set else 0 for p in chunk_pairs], dtype=np.int8)

        feat_chunk = extract_features_chunk(chunk, is_train=is_train)
        out_features_list.append(feat_chunk)

        print(f"  Chunk {c_idx+1:>2d}/{n_chunks} ({len(chunk):,d} pairs) processed in {time.time()-t0:.1f}s")

    del s1_df, s23_df, cands_df
    gc.collect()

    features_df = pd.concat(out_features_list, ignore_index=True)
    del out_features_list
    gc.collect()

    # Diagnostics & Class Balance
    if is_train and "label" in features_df.columns:
        n_pos = (features_df["label"] == 1).sum()
        n_neg = (features_df["label"] == 0).sum()
        print(f"\n--- Training Class Balance ---")
        print(f"  Positive matches (label=1): {n_pos:>10,d}  ({100*n_pos/len(features_df):.2f}%)")
        print(f"  Negative pairs   (label=0): {n_neg:>10,d}  ({100*n_neg/len(features_df):.2f}%)")
        print(f"  Imbalance ratio (neg:pos):  {n_neg/n_pos:.1f} : 1")

        print(f"\n--- Feature Means by Class (Separation Sanity Check) ---")
        numeric_cols = [
            "name_fuzzy_ratio", "name_token_set_ratio", "name_jaccard",
            "name_char_ngram_jaccard", "address_fuzzy_ratio", "address_jaccard",
            "prelim_combined_score", "legal_suffix_match", "hit_strategy_a", "hit_strategy_b"
        ]
        means_by_label = features_df.groupby("label")[numeric_cols].mean().T
        means_by_label.columns = ["Class 0 (Non-Match)", "Class 1 (True Match)"]
        print(means_by_label.to_string())

    # Save features to parquet
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_parquet = Path(out_dir) / f"{split}_features.parquet"
    features_df.to_parquet(out_parquet, index=False)
    file_size_mb = out_parquet.stat().st_size / 1e6
    mem_size_mb = features_df.memory_usage(deep=True).sum() / 1e6

    print(f"\n✓ Saved features to {out_parquet}")
    print(f"  Rows: {len(features_df):,d} | Columns: {features_df.shape[1]}")
    print(f"  File size on disk: {file_size_mb:.1f} MB | In-memory size: {mem_size_mb:.1f} MB")
    print(f"  Phase 4 ({split}) completed in {time.time()-t_start:.1f}s\n")

    return features_df


def main():
    parser = argparse.ArgumentParser(description="Phase 4 — Feature Engineering")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--norm-dir", default="output/normalized")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--out-dir", default="output/features")
    parser.add_argument("--chunk-size", type=int, default=500000)
    args = parser.parse_args()

    if args.split in ("train", "both"):
        build_features(
            split="train",
            norm_dir=args.norm_dir,
            train_dir=args.train_dir,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
        )

    if args.split in ("test", "both"):
        build_features(
            split="test",
            norm_dir=args.norm_dir,
            train_dir=args.train_dir,
            out_dir=args.out_dir,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
