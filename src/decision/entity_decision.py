from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import DecisionConfig


@dataclass
class CandidateEvidence:
    candidate_id: str
    source: str

    # Output from Teammate 3
    pair_probability: float

    # Output from our graph layer
    graph_support: float = 0.0
    contradiction: float = 0.0


def adjusted_score(
    candidate: CandidateEvidence,
    config: DecisionConfig,
) -> float:
    """
    Apply bounded graph evidence to the pairwise probability.

    Pairwise probability remains the primary signal.
    Graph evidence can only make a limited adjustment.
    """

    support_bonus = min(
        candidate.graph_support * config.max_graph_bonus,
        config.max_graph_bonus,
    )

    contradiction_penalty = min(
        candidate.contradiction * config.max_contradiction_penalty,
        config.max_contradiction_penalty,
    )

    score = (
        candidate.pair_probability
        + support_bonus
        - contradiction_penalty
    )

    return max(0.0, min(1.0, score))


def decide_candidate(
    candidate: CandidateEvidence,
    config: DecisionConfig,
) -> bool:
    """
    Decide whether one candidate should be included
    in the final match set.
    """

    score = adjusted_score(candidate, config)

    # Strong pairwise evidence.
    if candidate.pair_probability >= config.high_threshold:
        return candidate.contradiction < 1.0

    # Borderline candidate:
    # graph support is required.
    if candidate.pair_probability >= config.low_threshold:
        return (
            candidate.graph_support >= config.borderline_support_threshold
            and candidate.contradiction < 1.0
            and score >= config.low_threshold
        )

    # Weak pairwise evidence is rejected.
    return False


def decide_entity_matches(
    candidates: list[CandidateEvidence],
    config: DecisionConfig | None = None,
) -> list[str]:
    """
    Convert candidate-level evidence into a variable-sized
    match set for one S1 entity.

    Returns:
        []                       -> no matches
        ["S2-..."]               -> one match
        ["S2-...", "S3-..."]     -> multiple matches
    """

    if config is None:
        config = DecisionConfig()

    matched = []

    for candidate in candidates:
        if decide_candidate(candidate, config):
            matched.append(candidate.candidate_id)

    # Safety: no duplicate IDs.
    return list(dict.fromkeys(matched))