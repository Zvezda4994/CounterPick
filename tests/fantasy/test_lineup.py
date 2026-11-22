import numpy as np

from pickem.fantasy.lineup import Evaluator, check_lineup
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import N_ROUNDS


def test_check_lineup(pool: PlayerPool) -> None:
    team0 = [int(i) for i in np.flatnonzero(pool.team == 0)]
    others = [int(i) for i in np.flatnonzero(pool.team == 15)]
    assert check_lineup(pool, [*team0[:2], *others[:3]], 10**7, 3) == []
    assert any("from Team 01" in p for p in check_lineup(pool, team0, 10**7, 2))
    assert any("budget" in p for p in check_lineup(pool, [*team0[:2], *others[:3]], 500_000, 3))
    assert check_lineup(pool, [0, 0, 1, 2, 3], 10**7, 5)


def test_plan_is_legal_and_its_expectation_is_exact(evaluator: Evaluator) -> None:
    players = (0, 6, 12, 18, 24)
    plan = evaluator.plan(players)
    assert len(set(plan.roles)) == 5
    assert len({b for b, _, _ in plan.boosters}) == len(plan.boosters) == 18
    slots = {(pos, rnd) for _, pos, rnd in plan.boosters}
    assert len(slots) == 18 and all(0 <= r < N_ROUNDS for _, r in slots)
    scores = evaluator.scores(plan)
    assert np.isclose(scores.mean(), plan.expected)


def test_main_awp_pays_only_for_awpers(evaluator: Evaluator, pool: PlayerPool) -> None:
    awpers = pool.stats["awp_kpr"] > 0.3
    main_awp = evaluator.role_mean[:, 0]
    assert np.min(main_awp[awpers]) > 0 > np.max(main_awp[~awpers])
