import re
import unicodedata


def normalize_text(value):
    """
    Basic normalization for business names and addresses.

    - Handles missing values
    - Converts text to lowercase
    - Normalizes Unicode representation
    - Replaces punctuation with spaces
    - Collapses repeated whitespace
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Normalize Unicode representation
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Replace punctuation/special characters with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse multiple spaces
    value = re.sub(r"\s+", " ", value).strip()

    return value

from difflib import SequenceMatcher


def character_similarity(value_a, value_b):
    """
    Calculate character-level similarity between two text values.

    Returns a value between 0 and 1:
        1.0 = identical
        0.0 = completely different
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    # If both values are missing
    if not a and not b:
        return 1.0

    # If only one value is missing
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()

def token_similarity(value_a, value_b):
    """
    Calculate Jaccard similarity between the sets of tokens
    in two text values.

    Returns a value between 0 and 1.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    # Both values contain no tokens
    if not tokens_a and not tokens_b:
        return 1.0

    # One value has no tokens
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)

def edit_similarity(value_a, value_b):
    """
    Normalized Levenshtein similarity.

    Returns:
        1.0 -> identical strings
        0.0 -> maximally different strings
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    # Keep b as the shorter string to reduce memory use.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char_a != char_b)

            current.append(min(insertion, deletion, substitution))

        previous = current

    distance = previous[-1]
    max_length = max(len(a), len(b))

    return 1.0 - (distance / max_length)

def extract_numbers(value):
    """
    Extract numeric components from a text value.

    Example:
        '1795 Westchester Drive, NC 27262'
        -> {'1795', '27262'}
    """

    text = normalize_text(value)

    if not text:
        return set()

    return set(re.findall(r"\d+", text))

def address_number_overlap(value_a, value_b):
    """
    Calculate Jaccard overlap between numeric components
    of two addresses.
    """

    numbers_a = extract_numbers(value_a)
    numbers_b = extract_numbers(value_b)

    if not numbers_a and not numbers_b:
        return 1.0

    if not numbers_a or not numbers_b:
        return 0.0

    intersection = numbers_a & numbers_b
    union = numbers_a | numbers_b

    return len(intersection) / len(union)

def country_agreement(value_a, value_b):
    """
    Return 1 if both country values are present and equal.
    Return 0 otherwise.
    """

    a = normalize_text(value_a)
    b = normalize_text(value_b)

    if not a or not b:
        return 0

    return int(a == b)

def is_missing(value):
    """
    Return 1 if a value is missing/empty, otherwise 0.
    """

    if value is None:
        return 1

    if not isinstance(value, str):
        return 0

    return int(not value.strip())

def build_pair_features(record_a, record_b):
    """
    Build the complete feature vector for a pair of business records.

    Parameters
    ----------
    record_a : dict
        First business record.

    record_b : dict
        Second business record.

    Returns
    -------
    dict
        Pairwise evidence features.
    """

    name_a = record_a.get("business_name")
    name_b = record_b.get("business_name")

    address_a = record_a.get("business_address")
    address_b = record_b.get("business_address")

    country_a = record_a.get("country")
    country_b = record_b.get("country")

    features = {
        # -------------------------
        # Name evidence
        # -------------------------
        "name_char_similarity":
            character_similarity(name_a, name_b),

        "name_token_similarity":
            token_similarity(name_a, name_b),

        "name_edit_similarity":
            edit_similarity(name_a, name_b),

        # -------------------------
        # Address evidence
        # -------------------------
        "address_char_similarity":
            character_similarity(address_a, address_b),

        "address_token_similarity":
            token_similarity(address_a, address_b),

        "address_edit_similarity":
            edit_similarity(address_a, address_b),

        "address_number_overlap":
            address_number_overlap(address_a, address_b),

        # -------------------------
        # Country evidence
        # -------------------------
        "country_same":
            country_agreement(country_a, country_b),

        # -------------------------
        # Missingness
        # -------------------------
        "name_a_missing":
            is_missing(name_a),

        "name_b_missing":
            is_missing(name_b),

        "address_a_missing":
            is_missing(address_a),

        "address_b_missing":
            is_missing(address_b),
    }

    return features