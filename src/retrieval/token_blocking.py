from collections import defaultdict, Counter

from .normalization import tokenize_text


def build_token_frequency(df):
    """
    Count how many source records contain each token.

    Each token is counted at most once per business name.
    """

    token_counts = Counter()

    for name in df["business_name"]:

        tokens = set(tokenize_text(name))

        for token in tokens:

            if token:
                token_counts[token] += 1

    return token_counts


def build_token_index(
    df,
    min_frequency=5,
    max_frequency=500,
):
    """
    Build an inverted index for informative name tokens.

    Only tokens with frequency in the specified range
    are indexed.
    """

    token_counts = build_token_frequency(df)

    index = defaultdict(list)

    for row in df.itertuples(index=False):

        tokens = set(
            tokenize_text(row.business_name)
        )

        for token in tokens:

            frequency = token_counts.get(token, 0)

            if (
                frequency < min_frequency
                or frequency > max_frequency
            ):
                continue

            key = f"{row.country}|{token}"

            index[key].append(row.entity_id)

    return dict(index), token_counts


def lookup_token_candidates(
    row,
    index,
    token_counts,
    max_tokens=2,
):
    """
    Retrieve candidates using the rarest informative
    name tokens.
    """

    tokens = set(
        tokenize_text(row.business_name)
    )

    informative_tokens = []

    for token in tokens:

        frequency = token_counts.get(token, 0)

        if frequency <= 0:
            continue

        informative_tokens.append(
            (frequency, token)
        )

    # Rarest tokens first
    informative_tokens.sort(
        key=lambda x: x[0]
    )

    selected_tokens = [
        token
        for frequency, token
        in informative_tokens[:max_tokens]
    ]

    candidates = set()

    for token in selected_tokens:

        key = f"{row.country}|{token}"

        candidates.update(
            index.get(key, [])
        )

    return list(candidates)