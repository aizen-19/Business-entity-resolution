#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 4: High-Throughput Parallel Feature Engineering (src/features.py)

Extracts 22 discriminative string similarity, token overlap, structural, and
heuristic features for candidate pairs generated in Phase 3.
Features are computed with C-accelerated rapidfuzz and streamed directly to
Parquet via PyArrow to prevent memory spikes.

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def compute_token_jaccard_batch(tokens1_arr: np.ndarray, tokens2_arr: np.ndarray) -> np.ndarray:
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


def compute_trigram_jaccard_batch(text1_arr: np.ndarray, text2_arr: np.ndarray) -> np.ndarray:
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


def compute_fuzzy_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.ratio(s1, s2)
    return res


def compute_token_sort_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz token_sort_ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_sort_ratio(s1, s2)
    return res


def compute_token_set_ratio_batch(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Compute rapidfuzz token_set_ratio across paired string arrays."""
    n = len(arr1)
    res = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s1 = arr1[i]
        s2 = arr2[i]
        if s1 and s2:
            res[i] = fuzz.token_set_ratio(s1, s2)
    return res


def extract_features_subchunk(
    s1_ids: np.ndarray,
    cand_ids: np.ndarray,
    s1_country: np.ndarray,
    cand_country: np.ndarray,
    s1_name_core: np.ndarray,
    cand_name_core: np.ndarray,
    s1_name_tok: np.ndarray,
    cand_name_tok: np.ndarray,
    s1_sfx: np.ndarray,
    cand_sfx: np.ndarray,
    s1_non_latin: np.ndarray,
    cand_non_latin: np.ndarray,
    s1_was_domain: np.ndarray,
    cand_was_domain: np.ndarray,
    s1_addr_core: np.ndarray,
    cand_addr_core: np.ndarray,
    s1_addr_tok: np.ndarray,
    cand_addr_tok: np.ndarray,
    s1_landmark: np.ndarray,
    cand_landmark: np.ndarray,
    s1_addr_empty: np.ndarray,
    cand_addr_empty: np.ndarray,
    hit_a: np.ndarray,
    hit_b: np.ndarray,
    hit_c: np.ndarray,
    labels: np.ndarray = None,
) -> dict:
    """Compute feature dictionary for a batch of aligned numpy arrays."""
    # ── 1. Name Features ──
    name_fuzzy = compute_fuzzy_ratio_batch(s1_name_core, cand_name_core)
    name_tok_sort = compute_token_sort_ratio_batch(s1_name_core, cand_name_core)
    name_tok_set = compute_token_set_ratio_batch(s1_name_core, cand_name_core)
    name_jaccard = compute_token_jaccard_batch(s1_name_tok, cand_name_tok)
    name_tri_jaccard = compute_trigram_jaccard_batch(s1_name_core, cand_name_core)

    legal_sfx_match = (s1_sfx == cand_sfx) & (s1_sfx != "")
    s1_len = np.array([len(s) for s in s1_name_core], dtype=np.int16)
    cand_len = np.array([len(s) for s in cand_name_core], dtype=np.int16)
    name_len_diff = np.abs(s1_len - cand_len)

    name_non_latin_flag = s1_non_latin | cand_non_latin
    was_domain_flag = s1_was_domain | cand_was_domain

    # ── 2. Address Features ──
    addr_empty_either = s1_addr_empty | cand_addr_empty
    addr_empty_both = s1_addr_empty & cand_addr_empty

    raw_addr_fuzzy = compute_fuzzy_ratio_batch(s1_addr_core, cand_addr_core)
    raw_addr_tok_set = compute_token_set_ratio_batch(s1_addr_core, cand_addr_core)
    raw_addr_jaccard = compute_token_jaccard_batch(s1_addr_tok, cand_addr_tok)

    addr_fuzzy = np.where(addr_empty_either, -1.0, raw_addr_fuzzy).astype(np.float32)
    addr_tok_set = np.where(addr_empty_either, -1.0, raw_addr_tok_set).astype(np.float32)
    addr_jaccard = np.where(addr_empty_either, -1.0, raw_addr_jaccard).astype(np.float32)

    landmark_present = (s1_landmark != "") | (cand_landmark != "")
    s1_alen = np.array([len(s) for s in s1_addr_core], dtype=np.int16)
    cand_alen = np.array([len(s) for s in cand_addr_core], dtype=np.int16)
    addr_len_diff = np.where(addr_empty_either, -1, np.abs(s1_alen - cand_alen)).astype(np.int16)

    # ── 3. Cross & Structural Features ──
    country_match = (s1_country == cand_country)
    is_source3 = np.array([cid.startswith("S3-") for cid in cand_ids], dtype=bool)

    prelim_score = np.where(
        addr_jaccard >= 0,
        0.6 * name_jaccard + 0.4 * addr_jaccard,
        name_jaccard,
    ).astype(np.float32)

    res = {
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "name_fuzzy_ratio": name_fuzzy,
        "name_token_sort_ratio": name_tok_sort,
        "name_token_set_ratio": name_tok_set,
        "name_jaccard": name_jaccard,
        "name_char_ngram_jaccard": name_tri_jaccard,
        "legal_suffix_match": legal_sfx_match,
        "name_len_diff": name_len_diff,
        "name_has_non_latin_either": name_non_latin_flag,
        "was_domain_format_either": was_domain_flag,
        "address_is_empty_either": addr_empty_either,
        "address_is_empty_both": addr_empty_both,
        "address_fuzzy_ratio": addr_fuzzy,
        "address_jaccard": addr_jaccard,
        "address_token_set_ratio": addr_tok_set,
        "landmark_present_either": landmark_present,
        "address_len_diff": addr_len_diff,
        "country_match": country_match,
        "is_source3": is_source3,
        "hit_strategy_a": hit_a,
        "hit_strategy_b": hit_b,
        "hit_strategy_c": hit_c,
        "prelim_combined_score": prelim_score,
    }
    if labels is not None:
        res["label"] = labels

    return res


def build_features(
    split: str = "train",
    cand_path: str = None,
    norm_dir: str = "output/normalized",
    train_dir: str = "dataset/train",
    out_dir: str = "output/features",
    chunk_size: int = 500000,
    max_train_negatives_per_s1: int = None,
) -> None:
    """Build streaming Parquet feature dataset for train or test candidate pairs."""
    print(f"\n{'='*80}")
    print(f"  PHASE 4: STREAMING FEATURE ENGINEERING — {split.upper()} SPLIT")
    print(f"{'='*80}\n")

    t_start = time.time()
    if cand_path is None:
        cand_path = f"output/candidates/{split}_candidates.parquet"

    print(f"Loading candidate pairs from {cand_path} ...")
    cands_df = pd.read_parquet(cand_path)
    total_pairs = len(cands_df)
    print(f"  Total raw candidate pairs: {total_pairs:,d}")

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

    is_train = (split == "train")
    true_pairs_set = set()

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

    # Prepare streaming output Parquet writer
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_parquet = Path(out_dir) / f"{split}_features.parquet"
    if out_parquet.exists():
        out_parquet.unlink()

    writer = None
    total_written = 0
    total_positives = 0
    total_negatives = 0

    n_chunks = int(np.ceil(total_pairs / chunk_size))
    print(f"Streaming feature extraction in {n_chunks} chunks (~{chunk_size:,d} pairs/chunk) ...\n")

    for c_idx in range(n_chunks):
        t0 = time.time()
        start = c_idx * chunk_size
        end = min(start + chunk_size, total_pairs)
        chunk = cands_df.iloc[start:end].copy()

        # Join S1 and S23 records
        chunk = chunk.join(s1_df.add_prefix("s1_"), on="source1_entity_id")
        chunk = chunk.join(s23_df.add_prefix("cand_"), on="candidate_entity_id")

        labels = None
        if is_train:
            chunk_pairs = list(zip(chunk["source1_entity_id"], chunk["candidate_entity_id"]))
            labels = np.array([1 if p in true_pairs_set else 0 for p in chunk_pairs], dtype=np.int8)
            total_positives += int((labels == 1).sum())
            total_negatives += int((labels == 0).sum())

        feat_dict = extract_features_subchunk(
            s1_ids=chunk["source1_entity_id"].values,
            cand_ids=chunk["candidate_entity_id"].values,
            s1_country=chunk["s1_country"].fillna("").values,
            cand_country=chunk["cand_country"].fillna("").values,
            s1_name_core=chunk["s1_name_core"].fillna("").values,
            cand_name_core=chunk["cand_name_core"].fillna("").values,
            s1_name_tok=chunk["s1_name_token_set"].fillna("").values,
            cand_name_tok=chunk["cand_name_token_set"].fillna("").values,
            s1_sfx=chunk["s1_legal_suffix"].fillna("").values,
            cand_sfx=chunk["cand_legal_suffix"].fillna("").values,
            s1_non_latin=chunk["s1_name_has_non_latin"].fillna(False).values,
            cand_non_latin=chunk["cand_name_has_non_latin"].fillna(False).values,
            s1_was_domain=chunk["s1_was_domain_format"].fillna(False).values,
            cand_was_domain=chunk["cand_was_domain_format"].fillna(False).values,
            s1_addr_core=chunk["s1_address_core"].fillna("").values,
            cand_addr_core=chunk["cand_address_core"].fillna("").values,
            s1_addr_tok=chunk["s1_address_token_set"].fillna("").values,
            cand_addr_tok=chunk["cand_address_token_set"].fillna("").values,
            s1_landmark=chunk["s1_address_landmark"].fillna("").values,
            cand_landmark=chunk["cand_address_landmark"].fillna("").values,
            s1_addr_empty=chunk["s1_address_is_empty"].fillna(True).values,
            cand_addr_empty=chunk["cand_address_is_empty"].fillna(True).values,
            hit_a=chunk["hit_strategy_a"].values,
            hit_b=chunk["hit_strategy_b"].values,
            hit_c=chunk["hit_strategy_c"].values,
            labels=labels,
        )

        feat_df = pd.DataFrame(feat_dict)
        table = pa.Table.from_pandas(feat_df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(out_parquet, table.schema, compression="snappy")

        writer.write_table(table)
        total_written += len(feat_df)
        elapsed = time.time() - t0
        speed = len(feat_df) / max(elapsed, 0.001)

        print(f"  Chunk {c_idx+1:>3d}/{n_chunks} ({len(feat_df):>7,d} pairs) written in {elapsed:5.1f}s ({speed:,.0f} pairs/s) | Total: {total_written:>10,d}")

    if writer is not None:
        writer.close()

    del s1_df, s23_df, cands_df
    gc.collect()

    print(f"\n✓ Saved features to {out_parquet} ({out_parquet.stat().st_size/1e6:.1f} MB)")
    if is_train:
        print(f"--- Training Class Statistics ---")
        print(f"  Total pairs:      {total_written:>12,d}")
        print(f"  True Positives:   {total_positives:>12,d} ({100*total_positives/total_written:.2f}%)")
        print(f"  Negative pairs:   {total_negatives:>12,d} ({100*total_negatives/total_written:.2f}%)")
        print(f"  Negative/Pos:     {total_negatives/max(total_positives,1):.1f} : 1")

    print(f"  Phase 4 ({split}) completed in {time.time()-t_start:.1f}s\n")


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
