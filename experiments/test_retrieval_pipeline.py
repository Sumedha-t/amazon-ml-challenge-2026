import pandas as pd

from src.retrieval.retrieval_pipeline import (
    build_retrieval_indexes,
    retrieve_candidates,
)


# Use a small sample so the test is fast and memory-safe
S2_PATH = "data/student_resource/dataset/train_source2.tsv"
S1_PATH = "data/student_resource/dataset/train_source1.tsv"


print("Loading small S2 sample...")
s2 = pd.read_csv(
    S2_PATH,
    sep="\t",
    nrows=100_000,
)

print(f"S2 sample rows: {len(s2):,}")

print("\nBuilding retrieval indexes...")
indexes = build_retrieval_indexes(s2)

print("Indexes built successfully.")

print("\nLoading one S1 record...")
s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    nrows=1,
)

row = next(s1.itertuples(index=False))

print("\nRetrieving candidates...")
candidates = retrieve_candidates(row, indexes)

print(f"Number of candidates: {len(candidates)}")

print("\nFirst 10 candidates:")

for candidate in candidates[:10]:
    print(candidate)

print("\n" + "=" * 60)
print("RETRIEVAL PIPELINE TEST PASSED")
print("=" * 60)