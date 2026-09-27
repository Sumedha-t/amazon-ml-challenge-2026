from src.decision.entity_decision import (
    CandidateEvidence,
    decide_entity_matches,
)


candidates = [
    CandidateEvidence(
        candidate_id="S2-A",
        source="S2",
        pair_probability=0.97,
    ),

    CandidateEvidence(
        candidate_id="S2-B",
        source="S2",
        pair_probability=0.72,
        graph_support=0.95,
    ),

    CandidateEvidence(
        candidate_id="S3-A",
        source="S3",
        pair_probability=0.68,
        graph_support=0.90,
    ),

    CandidateEvidence(
        candidate_id="S3-B",
        source="S3",
        pair_probability=0.71,
        graph_support=0.10,
    ),

    CandidateEvidence(
        candidate_id="S2-C",
        source="S2",
        pair_probability=0.30,
    ),
]


result = decide_entity_matches(candidates)

print("Predicted matches:")
print(result)