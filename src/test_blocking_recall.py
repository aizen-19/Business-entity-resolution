import pandas as pd
import numpy as np
import time
from collections import defaultdict, Counter

print("Loading test partition...")
t0 = time.time()
s1 = pd.read_parquet('output/normalized/train_source1.parquet', columns=['entity_id', 'country', 'name_core', 'name_token_set', 'address_core', 'address_token_set', 'address_is_empty', 'name_has_non_latin']).head(100000)
s2 = pd.read_parquet('output/normalized/train_source2.parquet', columns=['entity_id', 'country', 'name_core', 'name_token_set', 'address_core', 'address_token_set', 'address_is_empty', 'name_has_non_latin']).head(250000)
s3 = pd.read_parquet('output/normalized/train_source3.parquet', columns=['entity_id', 'country', 'name_core', 'name_token_set', 'address_core', 'address_token_set', 'address_is_empty', 'name_has_non_latin']).head(250000)
s23 = pd.concat([s2, s3], ignore_index=True)
print(f"Loaded in {time.time()-t0:.2f}s. |S1|={len(s1):,d}, |S23|={len(s23):,d}")

# Load Ground Truth
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
s1_id_set = set(s1['entity_id'])
gt_subset = gt[gt['source1_entity_id'].isin(s1_id_set)]
true_pairs = []
s23_id_set = set(s23['entity_id'])
for _, row in gt_subset.iterrows():
    s1_id = row['source1_entity_id']
    m_str = row['matched_entity_ids']
    if m_str:
        for mid in m_str.split(','):
            mid = mid.strip()
            if mid in s23_id_set:
                true_pairs.append((s1_id, mid))

print(f"Eval ground truth pairs in subset: {len(true_pairs):,d}")

# Test Blocking Strategy
STOPWORDS = {'and', 'the', 'inc', 'corp', 'llc', 'ltd', 'pvt', 'co', 'sa', 'sarl', 'sas'}

idx = defaultdict(list)
for i, (country, n_tok, a_tok, a_empty) in enumerate(zip(s23['country'], s23['name_token_set'], s23['address_token_set'], s23['address_is_empty'])):
    # 1. All name tokens of length >= 2
    if n_tok:
        for t in n_tok.split('|'):
            if len(t) >= 2 and t not in STOPWORDS:
                idx[(country, 'name', t)].append(i)
    # 2. Address tokens (digits and words length >= 3)
    if not a_empty and a_tok:
        for t in a_tok.split('|'):
            if t not in STOPWORDS and (t.isdigit() or len(t) >= 4):
                idx[(country, 'addr', t)].append(i)

clean_idx = {k: np.array(v, dtype=np.uint32) for k, v in idx.items() if len(v) <= 5000}
print(f"Index built with {len(clean_idx):,d} keys")

s23_ids = s23['entity_id'].values

# Query S1 entities
cands_map = {}
for s1_id, country, n_tok, a_tok, a_empty in zip(s1['entity_id'], s1['country'], s1['name_token_set'], s1['address_token_set'], s1['address_is_empty']):
    matched_arrays = []
    # Name tokens
    if n_tok:
        for t in n_tok.split('|'):
            if len(t) >= 2 and (country, 'name', t) in clean_idx:
                matched_arrays.append(clean_idx[(country, 'name', t)])
    # Address tokens
    if not a_empty and a_tok:
        for t in a_tok.split('|'):
            if (t.isdigit() or len(t) >= 4) and (country, 'addr', t) in clean_idx:
                matched_arrays.append(clean_idx[(country, 'addr', t)])
    
    if matched_arrays:
        cands_concat = np.concatenate(matched_arrays)
        # Count frequency of each candidate index across matching keys
        uniq_cands, counts = np.unique(cands_concat, return_counts=True)
        if len(uniq_cands) > 60:
            top_order = np.argsort(-counts)[:60]
            uniq_cands = uniq_cands[top_order]
        cands_map[s1_id] = set(s23_ids[uniq_cands])
    else:
        cands_map[s1_id] = set()

# Calculate recall
recalled = sum(1 for s1_id, cand_id in true_pairs if cand_id in cands_map.get(s1_id, set()))
print(f"\n======================================")
print(f"RECALL: {100.0 * recalled / len(true_pairs):.2f}% ({recalled:,d} / {len(true_pairs):,d})")
print(f"======================================\n")
