"""Vectorized 16-team Swiss stage simulation.

Pairing, BO1/BO3 rules and Buchholz tiebreaks must match Valve's current rulebook exactly
(see ``pickem.swiss.rules`` for the cited sections).
"""

from pickem.swiss.simulate import (
    MatchLog,
    StageInput,
    SwissResult,
    simulate_from_uniforms,
    simulate_swiss,
)

__all__ = ["MatchLog", "StageInput", "SwissResult", "simulate_from_uniforms", "simulate_swiss"]
