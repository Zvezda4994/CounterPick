"""Scoring a lineup: five players, a role each, and a booster plan.

Roles are held for the whole stage and must differ (12 roles, 5 players; the team page greys out
a role already given to a teammate: "role already assigned", checked 2026-11-21); the plan maximizes
their expected points as an assignment problem. Boosters are planned before the stage: each of
the 18 goes to a different (player, round) slot, chosen to maximize expected booster points
(a booster on a round the player does not play earns nothing). Both can be changed between
rounds in the game, so these are pre-lock plans, re-planned as results come in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from pickem.fantasy.performance import PlayerSims
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import BOOST_POINTS, LINEUP_SIZE, N_BOOSTERS, N_ROUNDS


@dataclass(frozen=True)
class Plan:
    players: tuple[int, ...]
    roles: tuple[int, ...]
    """Role index (``ROLE_NAMES``) per player, same order as ``players``."""
    boosters: tuple[tuple[int, int, int], ...]
    """(booster index, position in ``players``, round index 0-4)."""
    expected: float
    """Expected total points."""


def check_lineup(
    pool: PlayerPool, players: Sequence[int], budget: int, max_per_team: int
) -> list[str]:
    """Problems with a lineup under the draft rules (empty if legal)."""
    problems = []
    if len(players) != LINEUP_SIZE or len(set(players)) != LINEUP_SIZE:
        problems.append(f"a lineup is {LINEUP_SIZE} different players")
    cost = int(pool.price[list(players)].sum())
    if cost > budget:
        problems.append(f"costs ${cost:,}, over the ${budget:,} budget")
    counts = np.bincount(pool.team[list(players)], minlength=int(np.max(pool.team)) + 1)
    over = sorted({pool.team_names[p] for p in players if counts[pool.team[p]] > max_per_team})
    if over:
        problems.append(f"more than {max_per_team} players from {', '.join(over)}")
    return problems


class Evaluator:
    """Expected values and per-sim scores of lineups on one set of player sims."""

    def __init__(self, sims: PlayerSims) -> None:
        self.sims = sims

    @cached_property
    def base_mean(self) -> NDArray[np.float64]:
        result: NDArray[np.float64] = self.sims.base.mean(axis=0)
        return result

    @cached_property
    def role_mean(self) -> NDArray[np.float64]:
        """(P, 12)."""
        result: NDArray[np.float64] = self.sims.roles.mean(axis=0)
        return result

    @cached_property
    def booster_prob(self) -> NDArray[np.float64]:
        """(P, 5, 18)."""
        return self.sims.booster_probabilities()

    @cached_property
    def player_value(self) -> NDArray[np.float64]:
        """Additive stand-in for a player's worth: base + best role + a fair share of boosters
        (the best booster each round, scaled to 18 boosters over 25 slots)."""
        share = N_BOOSTERS / (LINEUP_SIZE * N_ROUNDS)
        boost = BOOST_POINTS * share * self.booster_prob.max(axis=2).sum(axis=1)
        result: NDArray[np.float64] = self.base_mean + self.role_mean.max(axis=1) + boost
        return result

    def plan(self, players: Sequence[int]) -> Plan:
        idx = list(players)
        role_values = self.role_mean[idx]  # (5, 12)
        rows, cols = linear_sum_assignment(-role_values)
        roles = [0] * len(idx)
        for r, c in zip(rows, cols, strict=True):
            roles[int(r)] = int(c)
        slot_values = self.booster_prob[idx].reshape(len(idx) * N_ROUNDS, N_BOOSTERS).T
        b_rows, b_cols = linear_sum_assignment(-slot_values)
        boosters = tuple(
            sorted(
                (
                    (int(b), int(s) // N_ROUNDS, int(s) % N_ROUNDS)
                    for b, s in zip(b_rows, b_cols, strict=True)
                ),
                key=lambda x: (x[2], x[1]),
            )
        )
        expected = (
            float(self.base_mean[idx].sum())
            + float(role_values[rows, cols].sum())
            + BOOST_POINTS * float(slot_values[b_rows, b_cols].sum())
        )
        return Plan(tuple(idx), tuple(roles), boosters, expected)

    def scores(self, plan: Plan) -> NDArray[np.int32]:
        """(n,) total points of the plan in every sim."""
        idx = list(plan.players)
        total = self.sims.base[:, idx].sum(axis=1)
        total = total + self.sims.roles[:, idx, list(plan.roles)].sum(axis=1)
        for booster, pos, rnd in plan.boosters:
            bit = (self.sims.boosts[:, idx[pos], rnd] >> np.uint32(booster)) & 1
            total = total + BOOST_POINTS * bit.astype(np.int32)
        return np.asarray(total, dtype=np.int32)

    def booster_points(self, plan: Plan) -> tuple[float, float]:
        """(mean, SD) of the booster points, treating boosters as independent."""
        p = np.array(
            [self.booster_prob[plan.players[pos], rnd, b] for b, pos, rnd in plan.boosters]
        )
        return BOOST_POINTS * float(p.sum()), BOOST_POINTS * float(np.sqrt((p * (1 - p)).sum()))
