from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionConfig:
    # Initial values only.
    # These will be tuned using validation data later.
    high_threshold: float = 0.85
    low_threshold: float = 0.50

    # Maximum influence of graph evidence.
    max_graph_bonus: float = 0.08
    max_contradiction_penalty: float = 0.15

    # Borderline candidates can be rescued only with meaningful support.
    borderline_support_threshold: float = 0.70