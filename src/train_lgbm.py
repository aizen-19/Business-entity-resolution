#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
High-Performance LightGBM Training Pipeline (src/train_lgbm.py)

1. Subsamples high-signal training pairs: All true positive pairs + balanced hard negatives.
2. Extracts 22 discriminative features (fuzzy string ratios, token sort/set, Jaccard, address match, script flags).
3. Performs entity-grouped Train/Val split on source1_entity_id (no data leakage).
4. Trains LightGBM with early stopping and feature importance analysis.
5. Calibrates optimal decision threshold T* to maximize link-level F1.
6. Saves model booster and metadata.

Outputs:
  - output/models/lgbm_model.txt
  - output/models/model_metadata.json
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.model_selection import GroupShuffleSplit

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
    """Extract all 22 feature columns for a dataframe joined with S1 and Candidate normalized records."""
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
    if "label" in chunk.columns:
        feat_dict["label"] = chunk["label"].astype(np.int8).values

    return pd.DataFrame(feat_dict)


def evaluate_link_f1(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[float, float, float]:
    preds = probs >= threshold
    tp = np.sum(preds & (y_true == 1))
    fp = np.sum(preds & (y_true == 0))
    fn = np.sum((~preds) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser(description="High-Performance LightGBM Training")
    parser.add_argument("--candidates", default="output/candidates/train_candidates.parquet")
    parser.add_argument("--ground-truth", default="dataset/train/train_ground_truth.tsv")
    parser.add_argument("--norm-dir", default="output/normalized")
    parser.add_argument("--model-dir", default="output/models")
    parser.add_argument("--pos-sample", type=int, default=1500000, help="Number of positive pairs to sample")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Ratio of hard negatives to positives")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  LIGHTGBM MODEL TRAINING & LINK-F1 THRESHOLD OPTIMIZATION")
    print(f"{'='*80}\n")
    t_start = time.time()

    # 1. Load Ground Truth Pairs
    print(f"Loading ground truth labels from {args.ground_truth} ...")
    gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False)
    true_pairs_set = set()
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        m = row["matched_entity_ids"]
        if m:
            for mid in m.split(","):
                if mid.strip():
                    true_pairs_set.add((s1, mid.strip()))
    print(f"  Total true match pairs in Ground Truth: {len(true_pairs_set):,d}")

    # 2. Sample Training Pairs from train_candidates.parquet
    print(f"\nLoading candidates from {args.candidates} ...")
    cands_df = pd.read_parquet(args.candidates)
    print(f"  Total raw candidate pairs: {len(cands_df):,d}")

    print("Labeling candidates and sampling balanced high-signal dataset ...")
    cand_pairs = list(zip(cands_df["source1_entity_id"], cands_df["candidate_entity_id"]))
    labels = np.array([1 if p in true_pairs_set else 0 for p in cand_pairs], dtype=np.int8)
    cands_df["label"] = labels
    del cand_pairs, true_pairs_set
    gc.collect()

    pos_df = cands_df[cands_df["label"] == 1]
    neg_df = cands_df[cands_df["label"] == 0]
    del cands_df
    gc.collect()

    print(f"  Surviving True Positives: {len(pos_df):,d}")
    print(f"  Surviving Negatives:      {len(neg_df):,d}")

    n_pos = min(len(pos_df), args.pos_sample)
    n_neg = min(len(neg_df), n_pos * args.neg_ratio)

    pos_sample = pos_df.sample(n=n_pos, random_state=42)
    neg_sample = neg_df.sample(n=n_neg, random_state=42)
    del pos_df, neg_df
    gc.collect()

    train_cands = pd.concat([pos_sample, neg_sample], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    del pos_sample, neg_sample
    gc.collect()

    print(f"\nConstructed training sample: {len(train_cands):,d} pairs (Positives: {n_pos:,d}, Negatives: {n_neg:,d}, Ratio: {n_neg/n_pos:.1f}:1)")

    # 3. Join Normalized Source Tables
    cols_to_load = [
        "entity_id", "country", "legal_suffix", "name_core", "name_token_set",
        "name_has_non_latin", "was_domain_format", "address_core", "address_landmark",
        "address_is_empty", "address_token_set"
    ]
    print(f"Loading normalized source records ...")
    s1_df = pd.read_parquet(Path(args.norm_dir) / "train_source1.parquet", columns=cols_to_load).set_index("entity_id")
    s2_df = pd.read_parquet(Path(args.norm_dir) / "train_source2.parquet", columns=cols_to_load)
    s3_df = pd.read_parquet(Path(args.norm_dir) / "train_source3.parquet", columns=cols_to_load)
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True).set_index("entity_id")
    del s2_df, s3_df
    gc.collect()

    print("Joining feature tables ...")
    train_cands = train_cands.join(s1_df.add_prefix("s1_"), on="source1_entity_id")
    train_cands = train_cands.join(s23_df.add_prefix("cand_"), on="candidate_entity_id")
    del s1_df, s23_df
    gc.collect()

    # 4. Extract Features
    print(f"Extracting 22 discriminative feature columns for {len(train_cands):,d} pairs ...")
    t_feat = time.time()
    feats_df = extract_features_df(train_cands)
    del train_cands
    gc.collect()
    print(f"  Features extracted in {time.time()-t_feat:.1f}s")

    # 5. Entity-Grouped Train/Validation Split (GroupShuffleSplit on source1_entity_id)
    print("\nSplitting Train/Val by source1_entity_id (15% validation) ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(feats_df, groups=feats_df["source1_entity_id"]))

    train_data = feats_df.iloc[train_idx]
    val_data = feats_df.iloc[val_idx]

    X_train = train_data[FEATURE_COLS]
    y_train = train_data["label"].values
    X_val = val_data[FEATURE_COLS]
    y_val = val_data["label"].values

    print(f"  Train set: {len(X_train):,d} pairs (Positives: {(y_train==1).sum():,d})")
    print(f"  Val set:   {len(X_val):,d} pairs (Positives: {(y_val==1).sum():,d})")

    # 6. Train LightGBM
    scale_pos = float(np.sqrt((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    print(f"\nTraining LightGBM Classifier (num_leaves={args.num_leaves}, lr={args.learning_rate}, scale_pos_weight={scale_pos:.2f}) ...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=args.n_jobs,
        importance_type="gain",
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric=["auc", "binary_logloss"],
        callbacks=callbacks,
    )

    # 7. Decision Threshold Optimization
    print("\nOptimizing link decision threshold on validation set ...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_t, best_f1, best_p, best_r = 0.50, 0.0, 0.0, 0.0
    for t in np.linspace(0.10, 0.90, 81):
        p, r, f1 = evaluate_link_f1(y_val, val_probs, t)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, t, p, r

    print(f"\n{'━'*70}")
    print(f"  ★ OPTIMAL LINK DECISION THRESHOLD:  T = {best_t:.3f}")
    print(f"  ★ Validation Link Precision:        {best_p*100:6.2f}%")
    print(f"  ★ Validation Link Recall:           {best_r*100:6.2f}%")
    print(f"  ★ Validation Link-F1 Score:         {best_f1*100:6.2f}%")
    print(f"{'━'*70}\n")

    # Feature Importance
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)
    print("Top Feature Importances (Gain):")
    for rk, (_, r) in enumerate(importance_df.iterrows(), 1):
        print(f"  {rk:>2d}. {r['feature']:<28s} : {r['importance_gain']:>12,.1f}")

    # 8. Save Model and Metadata
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    booster_path = Path(args.model_dir) / "lgbm_model.txt"
    model.booster_.save_model(str(booster_path))
    print(f"\n✓ Saved LightGBM booster to {booster_path}")

    meta_path = Path(args.model_dir) / "model_metadata.json"
    metadata = {
        "best_threshold": float(best_t),
        "val_link_f1": float(best_f1),
        "val_precision": float(best_p),
        "val_recall": float(best_r),
        "best_iteration": int(model.best_iteration_),
        "feature_names": FEATURE_COLS,
        "scale_pos_weight": float(scale_pos),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata and threshold to {meta_path}")

    print(f"\nTraining pipeline completed in {time.time()-t_start:.1f}s\n")


if __name__ == "__main__":
    main()
