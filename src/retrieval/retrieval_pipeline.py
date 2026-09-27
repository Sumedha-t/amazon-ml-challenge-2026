from .name_blocking import (
    build_name_index,
    lookup_name_candidates,
)

from .address_blocking import (
    build_address_index,
    lookup_address_candidates,
)

from .token_blocking import (
    build_token_index,
    lookup_token_candidates,
)


def build_retrieval_indexes(target_df):
    """
    Build all retrieval indexes for one target source.

    The retrieval system uses three complementary blocking views:
    1. Normalized exact name
    2. Normalized exact address
    3. Informative name tokens

    Parameters
    ----------
    target_df : pandas.DataFrame
        Target source containing:
        entity_id, business_name, business_address, country

    Returns
    -------
    dict
        All indexes required for candidate retrieval.
    """

    name_index = build_name_index(target_df)

    address_index = build_address_index(target_df)

    token_index, token_counts = build_token_index(
        target_df,
        min_frequency=5,
        max_frequency=500,
    )

    return {
        "name_index": name_index,
        "address_index": address_index,
        "token_index": token_index,
        "token_counts": token_counts,
    }


def retrieve_candidates(source1_row, indexes):
    """
    Retrieve candidate target entities for one Source-1 record.

    Candidates from all three blocking views are UNIONED.
    A candidate is retained if it is found by at least one view.

    Parameters
    ----------
    source1_row : row-like object
        Source-1 record containing:
        entity_id, business_name, business_address, country

    indexes : dict
        Output of build_retrieval_indexes()

    Returns
    -------
    list of dict
        Each candidate contains:
        - entity_id
        - candidate_source

    candidate_source records which blocking views retrieved
    the candidate, e.g.:
        name
        address
        token
        name|token
        address|token
        name|address|token
    """

    candidates = {}

    # ---------------------------------------------------------
    # 1. Exact normalized NAME blocking
    # ---------------------------------------------------------

    name_candidates = lookup_name_candidates(
        source1_row,
        indexes["name_index"],
    )

    for entity_id in name_candidates:
        candidates.setdefault(entity_id, set()).add("name")

    # ---------------------------------------------------------
    # 2. Exact normalized ADDRESS blocking
    # ---------------------------------------------------------

    address_candidates = lookup_address_candidates(
        source1_row,
        indexes["address_index"],
    )

    for entity_id in address_candidates:
        candidates.setdefault(entity_id, set()).add("address")

    # ---------------------------------------------------------
    # 3. Informative NAME-TOKEN blocking
    # ---------------------------------------------------------

    token_candidates = lookup_token_candidates(
        source1_row,
        indexes["token_index"],
        indexes["token_counts"],
        max_tokens=2,
    )

    for entity_id in token_candidates:
        candidates.setdefault(entity_id, set()).add("token")

    # ---------------------------------------------------------
    # Convert provenance sets into deterministic strings
    # ---------------------------------------------------------

    output = []

    for entity_id, sources in candidates.items():

        source_string = "|".join(sorted(sources))

        output.append(
            {
                "entity_id": entity_id,
                "candidate_source": source_string,
            }
        )

    return output