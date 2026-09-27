import pandas as pd

from src.retrieval.retrieval_pipeline import (
    build_retrieval_indexes,
    retrieve_candidates,
)


def retrieve_for_target(source1_path, target_path, target_name):
    """
    Run the finalized retrieval pipeline for one target source.

    target_name should be "S2" or "S3".
    """

    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL FOR {target_name}")
    print(f"{'=' * 60}")

    # Load target source
    print(f"Loading {target_name}...")
    target_df = pd.read_csv(
        target_path,
        sep="\t",
    )

    print(f"{target_name} rows: {len(target_df):,}")

    # Build the three retrieval indexes
    print("Building retrieval indexes...")
    indexes = build_retrieval_indexes(target_df)

    # Load Source 1
    print("Loading Source 1...")
    source1_df = pd.read_csv(
        source1_path,
        sep="\t",
    )

    print(f"S1 rows: {len(source1_df):,}")

    # Demonstration on first S1 record
    row = next(source1_df.itertuples(index=False))

    candidates = retrieve_candidates(
        row,
        indexes,
    )

    print(f"\nS1 entity: {row.entity_id}")
    print(f"Number of candidates: {len(candidates)}")

    print("\nFirst 10 candidates:")

    for candidate in candidates[:10]:
        print(candidate)

    return candidates


if __name__ == "__main__":

    S1_PATH = "data/student_resource/dataset/train_source1.tsv"

    S2_PATH = "data/student_resource/dataset/train_source2.tsv"
    S3_PATH = "data/student_resource/dataset/train_source3.tsv"

    # S2 retrieval
    retrieve_for_target(
        S1_PATH,
        S2_PATH,
        "S2",
    )

    # S3 retrieval
    retrieve_for_target(
        S1_PATH,
        S3_PATH,
        "S3",
    )