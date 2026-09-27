from collections import defaultdict

import pandas as pd

from .normalization import compact_text


def build_name_index(df):
    """
    Build an index:

        country + compact business name
            -> candidate entity IDs

    Parameters
    ----------
    df : pandas.DataFrame
        Source-2 or source-3 dataframe.

    Returns
    -------
    dict
        Blocking key -> list of entity IDs.
    """
    index = defaultdict(list)

    for row in df.itertuples(index=False):
        name_key = compact_text(row.business_name)

        if not name_key:
            continue

        key = f"{row.country}|{name_key}"
        index[key].append(row.entity_id)

    return dict(index)


def lookup_name_candidates(row, index):
    """
    Retrieve candidates for one S1 row.
    """
    name_key = compact_text(row.business_name)

    if not name_key:
        return []

    key = f"{row.country}|{name_key}"

    return index.get(key, [])