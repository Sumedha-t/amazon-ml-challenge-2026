from collections import defaultdict

from .normalization import compact_text


def build_address_index(df):
    """
    Build an index:

        country + compact normalized address
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

        address_key = compact_text(row.business_address)

        if not address_key:
            continue

        key = f"{row.country}|{address_key}"

        index[key].append(row.entity_id)

    return dict(index)


def lookup_address_candidates(row, index):
    """
    Retrieve address-based candidates for one S1 row.
    """

    address_key = compact_text(row.business_address)

    if not address_key:
        return []

    key = f"{row.country}|{address_key}"

    return index.get(key, [])