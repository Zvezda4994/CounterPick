"""Lineup search.

Max expected points: every legal lineup is ranked by an additive stand-in (each player's
expected base points + best role + a share of boosters), the top ``k`` are found exactly by
branch and bound, and those are re-planned and re-ranked on their true expected totals (roles
must differ and boosters compete for slots, which the stand-in ignores).

High upside: steepest-ascent single-player swaps on E[season points] against the simulated
field, from the max-expected lineup and several random legal starts.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pickem.fantasy.lineup import Evaluator, Plan
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import LINEUP_SIZE, SEASON_TIERS, season_points


def top_additive(
    pool: PlayerPool, value: NDArray[np.float64], budget: int, max_per_team: int, k: int
) -> list[tuple[float, tuple[int, ...]]]:
    """The ``k`` legal lineups with the highest sum of ``value``, best first."""
    order = [int(i) for i in np.argsort(-value, kind="stable")]
    vals = [float(value[i]) for i in order]
    prices = [int(pool.price[i]) for i in order]
    teams = [int(pool.team[i]) for i in order]
    cheapest = int(np.min(pool.price))
    best: list[tuple[float, tuple[int, ...]]] = []  # min-heap of (value, lineup)
    counts: dict[int, int] = {}

    def bound(start: int, slots: int) -> float:
        return sum(vals[start : start + slots])

    def visit(start: int, chosen: list[int], total: float, spent: int) -> None:
        slots = LINEUP_SIZE - len(chosen)
        if slots == 0:
            item = (total, tuple(sorted(order[i] for i in chosen)))
            if len(best) < k:
                heapq.heappush(best, item)
            elif total > best[0][0]:
                heapq.heapreplace(best, item)
            return
        for i in range(start, len(order) - slots + 1):
            if len(best) == k and total + bound(i, slots) <= best[0][0]:
                return
            if spent + prices[i] + (slots - 1) * cheapest > budget:
                continue
            if counts.get(teams[i], 0) >= max_per_team:
                continue
            counts[teams[i]] = counts.get(teams[i], 0) + 1
            chosen.append(i)
            visit(i + 1, chosen, total + vals[i], spent + prices[i])
            chosen.pop()
            counts[teams[i]] -= 1

    visit(0, [], 0.0, 0)
    return sorted(best, reverse=True)


def max_expected(
    evaluator: Evaluator, pool: PlayerPool, budget: int, max_per_team: int, k: int = 200
) -> list[Plan]:
    """Plans of the top additive lineups, re-ranked by true expected points (best first)."""
    candidates = top_additive(pool, evaluator.player_value, budget, max_per_team, k)
    if not candidates:
        raise ValueError("no legal lineup fits the budget and team cap")
    plans = [evaluator.plan(lineup) for _, lineup in candidates]
    return sorted(plans, key=lambda p: -p.expected)


@dataclass(frozen=True)
class UpsideStats:
    expected: float
    sd: float
    season: float
    """E[season points]."""
    tiers: tuple[float, ...]
    """P(inside each tier of ``SEASON_TIERS``)."""


def upside(scores: NDArray[np.int32], cutoffs: NDArray[np.float64]) -> UpsideStats:
    reached = scores[:, None] >= cutoffs
    return UpsideStats(
        expected=float(scores.mean()),
        sd=float(scores.std()),
        season=float(season_points(scores, cutoffs).mean()),
        tiers=tuple(float(x) for x in reached.mean(axis=0)),
    )


def objective_function(name: str) -> Callable[[UpsideStats], float]:
    """``season`` (E[season points]) or ``topN`` (P(top N%)) for N in the season tiers."""
    if name == "season":
        return lambda s: s.season
    tiers = {f"top{round(100 * frac)}": j for j, (frac, _) in enumerate(SEASON_TIERS)}
    if name not in tiers:
        raise ValueError(f"objective must be season or one of {', '.join(tiers)}")
    j = tiers[name]
    return lambda s: s.tiers[j]


def _legal(pool: PlayerPool, players: Sequence[int], budget: int, max_per_team: int) -> bool:
    idx = list(players)
    if int(pool.price[idx].sum()) > budget:
        return False
    return int(np.max(np.bincount(pool.team[idx]))) <= max_per_team


def _random_lineup(
    pool: PlayerPool, budget: int, max_per_team: int, rng: np.random.Generator
) -> tuple[int, ...]:
    for _ in range(10_000):
        pick = rng.choice(pool.size, LINEUP_SIZE, replace=False)
        if _legal(pool, [int(i) for i in pick], budget, max_per_team):
            return tuple(sorted(int(i) for i in pick))
    raise ValueError("could not find a random legal lineup")


@dataclass(frozen=True)
class UpsideResult:
    plan: Plan
    stats: UpsideStats
    evaluated: int


def max_upside(
    evaluator: Evaluator,
    pool: PlayerPool,
    cutoffs: NDArray[np.float64],
    budget: int,
    max_per_team: int,
    rng: np.random.Generator,
    *,
    objective: str = "season",
    starts: Sequence[Sequence[int]] = (),
    restarts: int = 6,
) -> UpsideResult:
    """Steepest-ascent swaps on the objective from ``starts`` plus ``restarts`` random lineups."""
    score_of = objective_function(objective)
    cache: dict[tuple[int, ...], tuple[float, Plan, UpsideStats]] = {}

    def evaluate(lineup: tuple[int, ...]) -> tuple[float, Plan, UpsideStats]:
        if lineup not in cache:
            plan = evaluator.plan(lineup)
            stats = upside(evaluator.scores(plan), cutoffs)
            cache[lineup] = (score_of(stats) + 1e-9 * stats.expected, plan, stats)
        return cache[lineup]

    begin = [tuple(sorted(int(i) for i in s)) for s in starts]
    begin += [_random_lineup(pool, budget, max_per_team, rng) for _ in range(restarts)]
    best: tuple[float, Plan, UpsideStats] | None = None
    for lineup in begin:
        current = evaluate(lineup)
        while True:
            players = current[1].players
            neighbours = []
            for pos in range(LINEUP_SIZE):
                for new in range(pool.size):
                    if new in players:
                        continue
                    cand = tuple(sorted((*players[:pos], new, *players[pos + 1 :])))
                    if _legal(pool, cand, budget, max_per_team):
                        neighbours.append(cand)
            step = max((evaluate(c) for c in neighbours), key=lambda x: x[0], default=None)
            if step is None or step[0] <= current[0]:
                break
            current = step
        if best is None or current[0] > best[0]:
            best = current
    assert best is not None
    return UpsideResult(best[1], best[2], len(cache))
