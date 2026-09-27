from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
from difflib import SequenceMatcher


@dataclass
class CrossSourceEvidence:
    s2_id: str
    s3_id: str
    name_similarity: float
    address_similarity: float
    number_overlap: float
    country_agreement: float
    support: float
    contradiction: float


def normalize_text(value: object) -> str:
    if value is None:
        return ""

    text = str(value).lower().strip()

    if text in {"", "nan", "none"}:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    return text.strip()


def token_jaccard(a: object, b: object) -> float:
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def number_overlap(a: object, b: object) -> float:
    nums_a = set(re.findall(r"\d+", normalize_text(a)))
    nums_b = set(re.findall(r"\d+", normalize_text(b)))

    if not nums_a or not nums_b:
        return 0.0

    return len(nums_a & nums_b) / len(nums_a | nums_b)


def country_agreement(country_a: object, country_b: object) -> float:
    a = normalize_text(country_a)
    b = normalize_text(country_b)

    if not a or not b:
        return 0.0

    return 1.0 if a == b else 0.0


def character_similarity(a: object, b: object) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def compute_cross_source_evidence(
    s2_record: dict,
    s3_record: dict,
) -> CrossSourceEvidence:

    name_char = character_similarity(
        s2_record["business_name"],
        s3_record["business_name"],
    )

    name_token = token_jaccard(
        s2_record["business_name"],
        s3_record["business_name"],
    )

    address_char = character_similarity(
        s2_record["business_address"],
        s3_record["business_address"],
    )

    address_token = token_jaccard(
        s2_record["business_address"],
        s3_record["business_address"],
    )

    number_sim = number_overlap(
        s2_record["business_address"],
        s3_record["business_address"],
    )

    country_sim = country_agreement(
        s2_record["country"],
        s3_record["country"],
    )

    name_similarity = max(name_char, name_token)
    address_similarity = max(address_char, address_token)

    # Prototype support value.
    # Do NOT treat these weights as final model parameters yet.
    support = (
        0.35 * name_similarity
        + 0.45 * address_similarity
        + 0.10 * number_sim
        + 0.10 * country_sim
    )

    country_a = normalize_text(s2_record["country"])
    country_b = normalize_text(s3_record["country"])

    contradiction = 0.0

    # Missing country is unknown, not contradictory.
    if country_a and country_b and country_a != country_b:
        contradiction = 1.0

    return CrossSourceEvidence(
        s2_id=str(s2_record["entity_id"]),
        s3_id=str(s3_record["entity_id"]),
        name_similarity=name_similarity,
        address_similarity=address_similarity,
        number_overlap=number_sim,
        country_agreement=country_sim,
        support=support,
        contradiction=contradiction,
    )


def build_local_cross_source_graph(
    s2_candidates: list[dict],
    s3_candidates: list[dict],
) -> list[CrossSourceEvidence]:
    """
    Build S2 <-> S3 edges for one S1 entity.

    Only cross-source edges are created:
        S2 <-> S3

    No S2 <-> S2 or S3 <-> S3 edges.
    """

    edges = []

    for s2_record, s3_record in product(
        s2_candidates,
        s3_candidates,
    ):
        edges.append(
            compute_cross_source_evidence(
                s2_record,
                s3_record,
            )
        )

    return edges