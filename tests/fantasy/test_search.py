import itertools
from dataclasses import replace

import numpy as np

from pickem.fantasy.field import tier_cutoffs
from pickem.fantasy.lineup import Evaluator
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.search import _legal, max_expected, max_upside, top_additive, upside


def _small_pool(pool: PlayerPool, teams: int) -> PlayerPool:
    keep = np.flatnonzero(pool.team < teams)
    return replace(
        pool,
        names=tuple(pool.names[i] for i in keep),
        team_names=tuple(pool.team_names[i] for i in keep),
        team=pool.team[keep],
        price=pool.price[keep],
        stats={k: v[keep] for k, v in pool.stats.items()},
        hltv_rank=pool.hltv_rank[keep],
        picked=pool.picked[keep],
    )


def test_top_additive_matches_brute_force(pool: PlayerPool) -> None:
    small = _small_pool(pool, 4)  # 20 players, 15,504 lineups
    value = np.random.default_rng(5).normal(50, 15, small.size)
    budget, cap = 1_000_000, 2
    legal = [
        (float(value[list(c)].sum()), c)
        for c in itertools.combinations(range(small.size), 5)
        if _legal(small, c, budget, cap)
    ]
    legal.sort(key=lambda x: -x[0])
    got = top_additive(small, value, budget, cap, 25)
    assert [c for _, c in got] == [c for _, c in legal[:25]]
    assert np.allclose([v for v, _ in got], [v for v, _ in legal[:25]])


def test_max_expected_beats_every_candidate(evaluator: Evaluator, pool: PlayerPool) -> None:
    plans = max_expected(evaluator, pool, 1_000_000, 2, k=50)
    assert plans[0].expected == max(p.expected for p in plans)
    assert _legal(pool, plans[0].players, 1_000_000, 2)


def test_tier_cutoffs() -> None:
    scores = np.arange(1000, dtype=float)[None, :]
    cut = tier_cutoffs(scores)
    assert cut[0, 0] == 990  # top 1% of 1000 = the best 10
    assert cut[0, -1] == 200  # top 80%


def test_max_upside_improves_on_its_start(evaluator: Evaluator, pool: PlayerPool) -> None:
    rng = np.random.default_rng(1)
    field = np.vstack([rng.normal(300, 60, 400) for _ in range(evaluator.sims.n_sims)])
    cutoffs = tier_cutoffs(field)
    start = max_expected(evaluator, pool, 1_000_000, 2, k=20)[0]
    start_stats = upside(evaluator.scores(start), cutoffs)
    result = max_upside(
        evaluator, pool, cutoffs, 1_000_000, 2, rng, starts=[start.players], restarts=1
    )
    assert result.stats.season >= start_stats.season
    assert _legal(pool, result.plan.players, 1_000_000, 2)
