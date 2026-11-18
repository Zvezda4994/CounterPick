"""Playoffs: the 8-team single-elimination bracket after Stage 3, and its Pick'Em."""

from pickem.playoffs.bracket import (
    FORMATS,
    MatchProbabilities,
    PlayoffFormat,
    match_probabilities,
    outcome_probabilities,
    outcome_table,
    playoff_seeds,
    simulate_playoffs,
    team_odds,
)
from pickem.playoffs.picks import (
    Objective,
    PlayoffPick,
    best_picks,
    consistent_picks,
    favourites_pick,
    independent_picks,
    parse_picks,
    score_picks,
)

__all__ = [
    "FORMATS",
    "MatchProbabilities",
    "Objective",
    "PlayoffFormat",
    "PlayoffPick",
    "best_picks",
    "consistent_picks",
    "favourites_pick",
    "independent_picks",
    "match_probabilities",
    "outcome_probabilities",
    "outcome_table",
    "parse_picks",
    "playoff_seeds",
    "score_picks",
    "simulate_playoffs",
    "team_odds",
]
