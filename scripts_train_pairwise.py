import os
import random
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.pairwise_model import (
    FEATURE_COLUMNS,
    feature_vector,
    train_model,
    evaluate_model,
    save_model,
)


BASE = r"data\student_resource\dataset\train"

S1_PATH = os.path.join(BASE, "train_source1.tsv")
S2_PATH = os.path.join(BASE, "train_source2.tsv")
S3_PATH = os.path.join(BASE, "train_source3.tsv")
GT_PATH = os.path.join(BASE, "train_ground_truth.tsv")

# Keep this small because we have a deadline.
N_S1 = 5000
MAX_POSITIVES_PER_S1 = 3
N_NEGATIVES = 15000

random.seed(42)


def clean_record(row):
    return {
        "entity_id": row["entity_id"],
        "business_name": row["business_name"],
        "business_address": row["business_address"],
        "country": row["country"],
    }


print("1/6 Reading ground truth...")
gt = pd.read_csv(GT_PATH, sep="\t", nrows=N_S1)

print("2/6 Reading S1 sample...")
s1 = pd.read_csv(S1_PATH, sep="\t", nrows=N_S1)

gt_lookup = {}

for _, row in gt.iterrows():
    value = row["matched_entity_ids"]

    if pd.isna(value) or str(value).strip() == "":
        gt_lookup[row["source1_entity_id"]] = []
    else:
        gt_lookup[row["source1_entity_id"]] = [
            x.strip() for x in str(value).split(",") if x.strip()
        ]


s1_lookup = {
    row["entity_id"]: clean_record(row)
    for _, row in s1.iterrows()
}


# ---------------------------------------------------------
# Collect positive target IDs
# ---------------------------------------------------------

positive_ids = set()

for s1_id, matches in gt_lookup.items():
    for target_id in matches[:MAX_POSITIVES_PER_S1]:
        positive_ids.add(target_id)


print("3/6 Positive target IDs:", len(positive_ids))


# ---------------------------------------------------------
# Scan S2/S3 in chunks and collect:
#   - required positive records
#   - random negative target pool
# ---------------------------------------------------------

target_records = {}
negative_pool = []

for source_name, path in [("S2", S2_PATH), ("S3", S3_PATH)]:

    print(f"Scanning {source_name}...")

    for chunk in pd.read_csv(path, sep="\t", chunksize=100000):

        for _, row in chunk.iterrows():

            entity_id = row["entity_id"]

            if entity_id in positive_ids:
                target_records[entity_id] = clean_record(row)

            # Build a deterministic random pool of negatives.
            if len(negative_pool) < N_NEGATIVES * 3:
                if random.random() < 0.02:
                    negative_pool.append(clean_record(row))

        if len(target_records) == len(positive_ids) and len(negative_pool) >= N_NEGATIVES * 3:
            break


print("Positive target records found:", len(target_records))
print("Negative pool size:", len(negative_pool))


# ---------------------------------------------------------
# Build positive pairs
# ---------------------------------------------------------

print("4/6 Building positive pairs...")

X = []
y = []

positive_pair_keys = set()

for s1_id, matches in gt_lookup.items():

    if s1_id not in s1_lookup:
        continue

    source_record = s1_lookup[s1_id]

    for target_id in matches[:MAX_POSITIVES_PER_S1]:

        if target_id not in target_records:
            continue

        target_record = target_records[target_id]

        X.append(feature_vector(source_record, target_record))
        y.append(1)

        positive_pair_keys.add((s1_id, target_id))


print("Positive pairs:", len(X))


# ---------------------------------------------------------
# Build random negative pairs
# ---------------------------------------------------------

print("5/6 Building negative pairs...")

negative_count = 0

while negative_count < N_NEGATIVES:

    source_record = random.choice(list(s1_lookup.values()))
    target_record = random.choice(negative_pool)

    key = (source_record["entity_id"], target_record["entity_id"])

    if key in positive_pair_keys:
        continue

    X.append(feature_vector(source_record, target_record))
    y.append(0)

    negative_count += 1


print("Negative pairs:", negative_count)
print("Total pairs:", len(X))


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------

X = np.asarray(X, dtype=float)
y = np.asarray(y, dtype=int)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


print("Training model...")

model = train_model(X_train, y_train)

metrics = evaluate_model(
    model,
    X_val,
    y_val,
    threshold=0.5,
)

print("\nVALIDATION RESULTS")
print("------------------")
print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F0.5      : {metrics['f0.5']:.4f}")
print(f"Threshold : {metrics['threshold']:.2f}")


# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------

os.makedirs("models", exist_ok=True)

MODEL_PATH = r"models\pairwise_model.pkl"

save_model(model, MODEL_PATH)

print("\nSaved model to:")
print(MODEL_PATH)