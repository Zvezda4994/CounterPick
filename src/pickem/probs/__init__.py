"""Odds layer: odds sheet -> fair probabilities -> joint strength fit -> BO1/BO3 matrices.

Strengths come with uncertainty (``StrengthModel``), which ``simulate_with_uncertainty``
samples per simulation batch.
"""

from pickem.probs.fit import FitConfig, StrengthFit, fit_strengths
from pickem.probs.margin import remove_margin
from pickem.probs.odds_csv import MarketProbabilities, OddsSheet, fair_probabilities, load_odds_csv
from pickem.probs.strength import (
    StrengthModel,
    bo1_matrix,
    bo3_from_bo1,
    stage_from_strengths,
)
from pickem.probs.uncertain import simulate_with_uncertainty

__all__ = [
    "FitConfig",
    "MarketProbabilities",
    "OddsSheet",
    "StrengthFit",
    "StrengthModel",
    "bo1_matrix",
    "bo3_from_bo1",
    "fair_probabilities",
    "fit_strengths",
    "load_odds_csv",
    "remove_margin",
    "simulate_with_uncertainty",
    "stage_from_strengths",
]
