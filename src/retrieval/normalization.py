import re
import unicodedata


def normalize_text(value):
    """
    Normalize business names and addresses while preserving
    Unicode scripts such as Devanagari and Kannada.
    """
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Normalize Unicode representation.
    text = unicodedata.normalize("NFKC", value)

    # Lowercase.
    text = text.lower()

    # Replace punctuation characters with spaces.
    # Keep Unicode letters, numbers, combining marks and whitespace.
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith(("L", "N", "M")) or char.isspace():
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compact_text(value):
    """
    Compact representation used for exact blocking keys.
    """
    text = normalize_text(value)
    return re.sub(r"\s+", "", text)


def tokenize_text(value):
    """
    Token representation used for token-based blocking.
    """
    text = normalize_text(value)

    if not text:
        return []

    return text.split()

def compact_text(value):
    """
    Compact representation used for exact blocking keys.
    """
    text = normalize_text(value)
    return re.sub(r"\s+", "", text)


def tokenize_text(value):
    """
    Token representation used for token-based blocking.
    """
    text = normalize_text(value)

    if not text:
        return []

    return text.split()