"""HLTV fantasy scoring rules, as published on the game page (hltv.org/fantasy/676, 2026-11-21).

Per match a drafted player earns:
- rating points: 50 * (HLTV Rating 3.0 for the match - 1), rounded. HLTV does not publish this; it
  reproduces the rating points of 7 of 7 player-matches checked from IEM Beijing 2026 (game 671),
  for BO3s that went 2 or 3 maps (e.g. rating 1.74 -> 37, 0.80 -> -10, 1.04 -> 2).
- team points: +6 if the team won the match, -3 if it lost.
- role points: +5 (max bonus), +2 (small bonus) or -2 (penalty) for the role assigned to them.
- booster points: +5 if the booster assigned to them for that round succeeds, else 0.

Padding (verified on game 671): when a team advances early, each later round it does not play is
filled with the truncated mean of the rounds it did play, separately for rating, team and role
points (boosters are not padded). Eliminated teams get nothing for later rounds: in Cologne Major
2026 Stage 1 (game 629) the 0-3 teams SINNERS and Gaimin Gladiators scored -106 and -190 in total,
against -104 and -245 without padding and -174 and -405 with it.

Season points come from the final placement percentile in HLTV's public league.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

N_ROUNDS = 5
"""Fantasy rounds of a Swiss stage = the 5 Swiss rounds (game 629: 3-0 teams padded rounds 4-5)."""
LINEUP_SIZE = 5
DEFAULT_BUDGET = 1_000_000
DEFAULT_MAX_PER_TEAM = 2
"""Budget and team cap as shown in the Stage 1 draft UI: 'Max 2 players from each team'."""

TEAM_WIN, TEAM_LOSS = 6, -3
ROLE_MAX, ROLE_SMALL, ROLE_FAIL = 5, 2, -2
BOOST_POINTS = 5
RATING_SCALE = 50.0

SEASON_TIERS: tuple[tuple[float, int], ...] = (
    (0.01, 25),
    (0.05, 18),
    (0.10, 15),
    (0.20, 12),
    (0.30, 10),
    (0.40, 8),
    (0.50, 6),
    (0.60, 4),
    (0.70, 2),
    (0.80, 1),
)
"""(top fraction of the league, season points). Below the top 80%: 0."""


def rating_points(match_rating: ArrayLike) -> NDArray[np.int16]:
    """50 * (rating - 1), rounded half away from zero."""
    x = RATING_SCALE * (np.asarray(match_rating, dtype=np.float64) - 1.0)
    return np.asarray(np.sign(x) * np.floor(np.abs(x) + 0.5), dtype=np.int16)


@dataclass(frozen=True)
class Role:
    name: str
    stat: str
    """Key of the match statistic it is judged on (see ``pickem.fantasy.performance``)."""
    max_bonus: float
    """Max bonus strictly above this (strictly below if ``lower_is_better``)."""
    small_bonus: float
    """Small bonus at or above this (at or below if ``lower_is_better``); penalty otherwise."""
    lower_is_better: bool = False

    def points(self, value: ArrayLike) -> NDArray[np.int8]:
        v = np.asarray(value, dtype=np.float64)
        if self.lower_is_better:
            top, small = v < self.max_bonus, v <= self.small_bonus
        else:
            top, small = v > self.max_bonus, v >= self.small_bonus
        return np.where(top, ROLE_MAX, np.where(small, ROLE_SMALL, ROLE_FAIL)).astype(np.int8)


STAT_ROLES: tuple[Role, ...] = (
    Role("Main AWP", "awp_kpr", 0.35, 0.20),
    Role("HS Machine", "hs_pct", 60.0, 50.0),
    Role("Support", "support_pct", 25.0, 17.0),
    Role("Multi Fragger", "multikill_kpr", 0.20, 0.14),
    Role("Entry Fragger", "entry_kpr", 0.15, 0.08),
    Role("Attacker", "t_rating", 1.30, 0.90),
    Role("Defender", "ct_rating", 1.35, 1.00),
    Role("Eco Friendly", "eco_kpr", 0.07, 0.02),
    Role("Stathunter", "rating", 1.30, 1.00),
    Role("Camper", "dpr", 0.55, 0.65, lower_is_better=True),
    Role("Noob", "rating", 0.85, 1.12, lower_is_better=True),
)
LEADER = "Leader"
"""Max bonus: won a BO3 without dropping a map, or a BO1 in regulation. Small bonus: won a BO3
dropping a map, or a BO1 in overtime. Penalty: lost."""
ROLE_NAMES: tuple[str, ...] = (*(r.name for r in STAT_ROLES), LEADER)
N_ROLES = len(ROLE_NAMES)
LEADER_INDEX = N_ROLES - 1


def leader_points(
    won: NDArray[np.bool_], maps_lost: NDArray[np.integer], overtime_bo1: NDArray[np.bool_]
) -> NDArray[np.int8]:
    """``maps_lost`` counts maps the team lost in the match; ``overtime_bo1`` marks BO1s that
    went to overtime (False for BO3s)."""
    clean = won & (maps_lost == 0) & ~overtime_bo1
    return np.where(clean, ROLE_MAX, np.where(won, ROLE_SMALL, ROLE_FAIL)).astype(np.int8)


BOOSTER_NAMES: tuple[str, ...] = (
    "Hellcase",
    "Top of scoreboard",
    "Bottom of scoreboard",
    "Carry",
    "Aim bot",
    "Mister/Miss consistent",
    "Cannon fodder",
    "Best pistol round",
    "Quad",
    "Clutch",
    "Assist",
    "Avenger",
    "Bait",
    "Flash",
    "Kobe",
    "Farmer",
    "Saver",
    "Rambo",
)
"""Each booster is used once, on one player for one round, and succeeds if any map of that
player's match that round meets its condition (HLTV's breakdown shows the best map)."""
N_BOOSTERS = len(BOOSTER_NAMES)
BOOSTER_INDEX = {name: i for i, name in enumerate(BOOSTER_NAMES)}


def padded(
    points: NDArray[np.integer], played: NDArray[np.bool_], pad: NDArray[np.bool_]
) -> NDArray[np.int32]:
    """Total over rounds with padding: ``points`` and ``played`` are (..., N_ROUNDS); ``pad``
    (...) marks entries whose unplayed rounds are padded (the team advanced). Each padded round
    is worth trunc(mean of the played rounds)."""
    pts = np.where(played, points, 0).astype(np.int32)
    n_played = played.sum(axis=-1)
    total = pts.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_round = np.trunc(np.where(n_played > 0, total / np.maximum(n_played, 1), 0))
    extra = np.where(pad, per_round * (N_ROUNDS - n_played), 0)
    return np.asarray(total + extra, dtype=np.int32)


def season_points(score: ArrayLike, cutoffs: ArrayLike) -> NDArray[np.int8]:
    """Season points for ``score`` given each tier's minimum score ``cutoffs`` (..., n_tiers),
    tiers in ``SEASON_TIERS`` order (top 1% first)."""
    s = np.asarray(score, dtype=np.float64)[..., None]
    c = np.asarray(cutoffs, dtype=np.float64)
    reached = s >= c
    values = np.array([pts for _, pts in SEASON_TIERS], dtype=np.int8)
    first = np.where(reached.any(axis=-1), reached.argmax(axis=-1), -1)
    return np.where(first >= 0, values[np.maximum(first, 0)], 0).astype(np.int8)
