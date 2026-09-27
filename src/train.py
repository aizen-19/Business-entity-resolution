#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Business Entity Resolution
Phase 5: LightGBM Model Training & Link-F1 Optimization (src/train.py)

Trains a LightGBM binary classifier on extracted candidate pair features.
Evaluates using GroupKFold / Group train-val split on `source1_entity_id`
to prevent data leakage, and finds the global Link-F1 optimal decision threshold.

Outputs:
  - output/models/lgbm_model.txt (LightGBM booster)
  - output/models/model_metadata.json (optimal threshold, feature names, metrics)
  - Feature importance analysis and validation performance report
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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


def evaluate_link_f1(
    val_df: pd.DataFrame,
    pred_probs: np.ndarray,
    threshold: float,
) -> tuple[float, float, float]:
    """Compute overall link Precision, Recall, and F1 at a given probability threshold."""
    preds = pred_probs >= threshold
    labels = val_df["label"].values

    tp = np.sum(preds & (labels == 1))
    fp = np.sum(preds & (labels == 0))
    fn = np.sum((~preds) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def train_model(
    train_features_path: str = "output/features/train_features.parquet",
    model_dir: str = "output/models",
    val_ratio: float = 0.15,
    sample_rows: int = None,
    n_estimators: int = 1200,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    n_jobs: int = 10,
):
    """Train LightGBM binary classifier with entity-grouped validation."""
    print(f"\n{'='*80}")
    print(f"  PHASE 5: LIGHTGBM MODEL TRAINING & THRESHOLD OPTIMIZATION")
    print(f"{'='*80}\n")

    t_start = time.time()
    print(f"Loading training features from {train_features_path} ...")
    if sample_rows is not None:
        print(f"  (Reading first {sample_rows:,d} rows for quick exploration)")
        df = pd.read_parquet(train_features_path).head(sample_rows)
    else:
        df = pd.read_parquet(train_features_path)

    n_total = len(df)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    print(f"  Total records: {n_total:,d}")
    print(f"  Positives:     {n_pos:,d} ({100*n_pos/n_total:.2f}%)")
    print(f"  Negatives:     {n_neg:,d} ({100*n_neg/n_total:.2f}%)")
    print(f"  Class Ratio:   {n_neg/max(n_pos,1):.1f} : 1")

    # Cast boolean columns to int8 for LightGBM
    for col in FEATURE_COLS:
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.int8)

    # ── Group-Aware Train/Validation Split by source1_entity_id ──
    print(f"\nPerforming Group-aware train/val split (val_ratio={val_ratio:.2f}) on source1_entity_id ...")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df["source1_entity_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    del df

    print(f"  Train set: {len(train_df):,d} pairs (Positives: {(train_df['label']==1).sum():,d})")
    print(f"  Val set:   {len(val_df):,d} pairs (Positives: {(val_df['label']==1).sum():,d})")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"].values
    X_val = val_df[FEATURE_COLS]
    y_val = val_df["label"].values

    # ── Train LightGBM ──
    # Compute scale_pos_weight for class imbalance
    scale_weight = float(np.sqrt((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
    print(f"\nConfiguring LightGBM (scale_pos_weight={scale_weight:.2f}, num_leaves={num_leaves}, lr={learning_rate}) ...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=n_jobs,
        importance_type="gain",
    )

    print("Training LightGBM model with early stopping on validation AUC ...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=["val"],
        eval_metric=["auc", "binary_logloss"],
        callbacks=callbacks,
    )

    # ── Optimal Threshold Search on Validation Set ──
    print(f"\nSearching for optimal Link-F1 decision threshold on validation set ...")
    val_probs = model.predict_proba(X_val)[:, 1]

    best_thresh = 0.50
    best_f1 = 0.0
    best_p = 0.0
    best_r = 0.0

    thresholds = np.linspace(0.10, 0.90, 81)
    results = []

    for t in thresholds:
        p, r, f1 = evaluate_link_f1(val_df, val_probs, t)
        results.append((t, p, r, f1))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_p = p
            best_r = r

    print(f"\n{'━'*70}")
    print(f"  ★ OPTIMAL LINK DECISION THRESHOLD:  T = {best_thresh:.3f}")
    print(f"  ★ Validation Precision:             {best_p*100:6.2f}%")
    print(f"  ★ Validation Recall:                {best_r*100:6.2f}%")
    print(f"  ★ Validation Link-F1:               {best_f1*100:6.2f}%")
    print(f"{'━'*70}\n")

    # ── Feature Importance ──
    importance_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)

    print("Top Feature Importances (Gain):")
    for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
        print(f"  {rank:>2d}. {row['feature']:<28s} : {row['importance_gain']:>12,.1f}")

    # ── Save Model & Metadata ──
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    booster_path = Path(model_dir) / "lgbm_model.txt"
    model.booster_.save_model(str(booster_path))
    print(f"\n✓ Saved LightGBM booster to {booster_path}")

    meta_path = Path(model_dir) / "model_metadata.json"
    metadata = {
        "best_threshold": float(best_thresh),
        "val_link_f1": float(best_f1),
        "val_precision": float(best_p),
        "val_recall": float(best_r),
        "best_iteration": int(model.best_iteration_),
        "feature_names": FEATURE_COLS,
        "scale_pos_weight": float(scale_weight),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata and threshold to {meta_path}")

    print(f"\n  Phase 5 Training completed in {time.time()-t_start:.1f}s\n")
    return model, metadata


def main():
    parser = argparse.ArgumentParser(description="Phase 5 — Model Training & Link-F1 Optimization")
    parser.add_argument("--features", default="output/features/train_features.parquet")
    parser.add_argument("--model-dir", default="output/models")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    train_model(
        train_features_path=args.features,
        model_dir=args.model_dir,
        val_ratio=args.val_ratio,
        sample_rows=args.sample_rows,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
