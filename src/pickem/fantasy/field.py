"""The field: lineups the rest of the HLTV league is likely to draft, scored in the same sims.

Ownership (the share of league teams drafting each player) is taken from ``picked_pct`` where
known (HLTV shows the 8 most-picked players before lock and every player after). The rest comes
from a log-linear model fitted on IEM Beijing 2026 (game 671, 80 players, r = 0.71):
log share = 1.383 - 0.795 (price - $200k) / $10k + 8.957 (rating - 1.05) - 1.955 log(team rank),
rescaled so that the shares add up to 5 players per team.

Field lineups are drawn player by player in proportion to adjusted weights, under the same
budget and team cap, and the weights are refitted until the sampled pick rates match the
targets. Each field lineup plays its roles and boosters as well as ``Evaluator.plan`` would (a
strong assumption for the whole league, but the top of the leaderboard is what decides season
points). Booster points are added as independent noise with the plan's mean and SD.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pickem.fantasy.lineup import Evaluator
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import LINEUP_SIZE, SEASON_TIERS

OWN_INTERCEPT, OWN_PRICE, OWN_RATING, OWN_LOG_RANK = 1.383, -0.795, 8.957, -1.955
DEFAULT_RANK = 30.0
MAX_SHARE = 0.95


def ownership(pool: PlayerPool) -> NDArray[np.float64]:
    """Expected share of league teams drafting each player; adds up to 5."""
    known = ~np.isnan(pool.picked)
    rank = np.where(np.isnan(pool.hltv_rank), DEFAULT_RANK, pool.hltv_rank)
    model = np.exp(
        OWN_INTERCEPT
        + OWN_PRICE * (pool.price - 200_000) / 10_000
        + OWN_RATING * (pool.stats["rating"] - 1.05)
        + OWN_LOG_RANK * np.log(np.maximum(rank, 1.0))
    )
    remaining = LINEUP_SIZE - float(pool.picked[known].sum())
    if remaining <= 0:
        raise ValueError("the known pick rates already add up to 5 players per team or more")
    if known.all():
        shares = pool.picked.copy()
    else:
        shares = np.where(known, pool.picked, model * remaining / model[~known].sum())
    result: NDArray[np.float64] = np.clip(shares, 1e-4, MAX_SHARE)
    return result


def _draw(
    pool: PlayerPool,
    weights: NDArray[np.float64],
    n: int,
    budget: int,
    max_per_team: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    n_players = pool.size
    chosen = np.zeros((n, n_players), dtype=np.bool_)
    per_team = np.zeros((n, int(np.max(pool.team)) + 1), dtype=np.int64)
    spent = np.zeros(n, dtype=np.int64)
    by_price = np.sort(pool.price)
    rows = np.arange(n)
    out = np.empty((n, LINEUP_SIZE), dtype=np.intp)
    for k in range(LINEUP_SIZE):
        # Keep enough money to fill the remaining slots even if the k cheapest are taken.
        reserve = int(by_price[k : k + LINEUP_SIZE - k - 1].sum())
        afford = budget - spent - reserve
        ok = (
            ~chosen
            & (per_team[:, pool.team] < max_per_team)
            & (pool.price[None, :] <= afford[:, None])
        )
        p = np.where(ok, weights[None, :], 0.0)
        total = p.sum(axis=1, keepdims=True)
        if not np.all(total > 0):
            raise ValueError("no legal lineup fits the budget and team cap")
        cum = np.cumsum(p / total, axis=1)
        pick = np.minimum((cum < rng.random((n, 1))).sum(axis=1), n_players - 1)
        out[:, k] = pick
        chosen[rows, pick] = True
        per_team[rows, pool.team[pick]] += 1
        spent += pool.price[pick]
    return out


def sample_field(
    pool: PlayerPool,
    shares: NDArray[np.float64],
    n: int,
    budget: int,
    max_per_team: int,
    rng: np.random.Generator,
    iterations: int = 12,
) -> NDArray[np.intp]:
    """(n, 5) field lineups whose pick rates approximate ``shares``."""
    weights = shares.copy()
    lineups = _draw(pool, weights, n, budget, max_per_team, rng)
    for _ in range(iterations):
        realized = np.bincount(lineups.ravel(), minlength=pool.size) / n
        weights = weights * (shares / np.maximum(realized, 0.5 / n)) ** 0.8
        lineups = _draw(pool, weights, n, budget, max_per_team, rng)
    return lineups


def field_scores(
    evaluator: Evaluator, lineups: NDArray[np.intp], rng: np.random.Generator
) -> NDArray[np.float64]:
    """(n_sims, n_lineups) total points of every field lineup in every sim."""
    sims = evaluator.sims
    plans = [evaluator.plan(row) for row in lineups]
    roles = np.array([p.roles for p in plans], dtype=np.intp)
    boost = np.array([evaluator.booster_points(p) for p in plans])
    n, m = sims.n_sims, len(lineups)
    out = np.empty((n, m), dtype=np.float64)
    for start in range(0, n, 500):
        s = slice(start, min(start + 500, n))
        base = sims.base[s][:, lineups].sum(axis=2)
        role = sims.roles[s][:, lineups, roles].sum(axis=2)
        noise = rng.standard_normal((base.shape[0], m))
        out[s] = base + role + boost[:, 0] + boost[:, 1] * noise
    return out


def tier_cutoffs(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """(n_sims, n_tiers) lowest field score still inside each tier (top 1% first)."""
    m = scores.shape[1]
    cut = np.empty((scores.shape[0], len(SEASON_TIERS)), dtype=np.float64)
    for j, (frac, _) in enumerate(SEASON_TIERS):
        k = max(1, math.ceil(frac * m))
        cut[:, j] = -np.partition(-scores, k - 1, axis=1)[:, k - 1]
    return cut
