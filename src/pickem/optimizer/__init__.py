"""Choose the 10 Pick'Em picks (2x 3-0, 2x 0-3, 6x advance) that maximize P(>=5 correct)."""

from pickem.optimizer.card import (
    CATEGORY_NAMES,
    Card,
    CardStats,
    Outcomes,
    correct_counts,
    evaluate,
    marginals,
)
from pickem.optimizer.search import (
    OptimizeResult,
    ScoredCard,
    favorites_card,
    optimize,
    prune_candidates,
)

__all__ = [
    "CATEGORY_NAMES",
    "Card",
    "CardStats",
    "OptimizeResult",
    "Outcomes",
    "ScoredCard",
    "correct_counts",
    "evaluate",
    "favorites_card",
    "marginals",
    "optimize",
    "prune_candidates",
]
