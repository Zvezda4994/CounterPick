import time

import numpy as np
import pytest

from pickem.optimizer import (
    Card,
    OptimizeResult,
    Outcomes,
    evaluate,
    favorites_card,
    optimize,
    prune_candidates,
)
from pickem.optimizer.card import ADVANCE, OH_THREE, THREE_OH
from pickem.swiss import StageInput, SwissResult, simulate_swiss
from tests.optimizer import brute_force

TEAMS = tuple(f"T{i:02d}" for i in range(16))

Oracle = list[tuple[brute_force.Key, brute_force.CardTuple]]
SmallCase = tuple[SwissResult, Outcomes, OptimizeResult, Oracle]


def simulate(n: int, *, spread: float, seed: int) -> tuple[SwissResult, Outcomes]:
    rng = np.random.default_rng(seed)
    strength = rng.normal(scale=spread, size=16)
    gap = strength[:, None] - strength[None, :]
    stage = StageInput(
        TEAMS,
        tuple(int(s) for s in rng.permutation(16) + 1),
        1.0 / (1.0 + np.exp(-gap)),
        1.0 / (1.0 + np.exp(-1.6 * gap)),
    )
    result = simulate_swiss(stage, n, rng)
    return result, Outcomes.from_result(result)


# --- exhaustive check on a small case -----------------------------------------------------


@pytest.fixture(scope="module", params=[("lopsided", 2.0, 21), ("even", 0.5, 22)])
def small_case(request: pytest.FixtureRequest) -> SmallCase:
    _, spread, seed = request.param
    result, outcomes = simulate(200, spread=spread, seed=seed)
    start = time.perf_counter()
    oracle = brute_force.brute_force_top(result.wins, result.losses, top=5)
    elapsed = time.perf_counter() - start
    found = optimize(outcomes, rng=np.random.default_rng(seed))
    print(
        f"\n[{request.param[0]}] brute force over 10,090,080 cards: {elapsed:.1f}s; "
        f"oracle top keys {[k for k, _ in oracle]}; "
        f"optimizer top keys {[s.stats.key for s in found.top]}"
    )
    return result, outcomes, found, oracle


def test_search_finds_the_brute_force_optimum(
    small_case: SmallCase,
) -> None:
    _, _, found, oracle = small_case
    assert found.best.stats.key == oracle[0][0]


def test_top_five_match_the_brute_force_top_five(
    small_case: SmallCase,
) -> None:
    _, _, found, oracle = small_case
    assert [s.stats.key for s in found.top] == [key for key, _ in oracle]


def test_reported_stats_agree_with_the_oracle(
    small_case: SmallCase,
) -> None:
    result, _, found, _ = small_case
    for scored in (*found.top, found.favorites):
        assert scored.stats.key == brute_force.score(result.wins, result.losses, scored.card.groups)


# --- search behaviour on a realistic sim count --------------------------------------------


@pytest.fixture(scope="module")
def medium_case() -> tuple[Outcomes, OptimizeResult]:
    _, outcomes = simulate(20_000, spread=1.0, seed=31)
    return outcomes, optimize(outcomes, rng=np.random.default_rng(5))


def neighbours(card: Card, candidates: tuple[tuple[int, ...], ...]) -> list[Card]:
    """Every replace and exchange move, enumerated independently of the search code."""
    groups = [list(g) for g in card.groups]
    out = []
    for c in range(3):
        for t in groups[c]:
            for j in candidates[c]:
                if j in card.teams:
                    continue
                new = [list(g) for g in groups]
                new[c] = [j if u == t else u for u in new[c]]
                out.append(Card.from_groups(new))
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            for a in groups[c1]:
                for b in groups[c2]:
                    if a in candidates[c2] and b in candidates[c1]:
                        new = [list(g) for g in groups]
                        new[c1] = [b if u == a else u for u in new[c1]]
                        new[c2] = [a if u == b else u for u in new[c2]]
                        out.append(Card.from_groups(new))
    return out


def test_best_card_is_a_local_optimum(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    outcomes, found = medium_case
    best = found.best
    moves = neighbours(best.card, found.candidates)
    assert len(moves) > 50
    for card in moves:
        assert evaluate(card, outcomes).key <= best.stats.key


def test_top_cards_are_distinct_sorted_and_scored_correctly(
    medium_case: tuple[Outcomes, OptimizeResult],
) -> None:
    outcomes, found = medium_case
    assert len(found.top) == 5
    assert len({s.card for s in found.top}) == 5
    keys = [s.stats.key for s in found.top]
    assert keys == sorted(keys, reverse=True)
    for scored in found.top:
        assert evaluate(scored.card, outcomes).key == scored.stats.key
    assert found.cards_evaluated >= len(found.top)


def test_picks_come_from_pruned_candidates(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    outcomes, found = medium_case
    p = outcomes.probabilities
    assert found.candidates == prune_candidates(outcomes, 6)
    assert len(found.candidates[THREE_OH]) == len(found.candidates[OH_THREE]) == 6
    assert found.candidates[ADVANCE] == tuple(range(16))
    assert min(p[THREE_OH][list(found.candidates[THREE_OH])]) >= np.sort(p[THREE_OH])[-6]
    for scored in found.top:
        assert set(scored.card.three_oh) <= set(found.candidates[THREE_OH])
        assert set(scored.card.oh_three) <= set(found.candidates[OH_THREE])


def test_beats_or_ties_favorites_in_sample(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    _, found = medium_case
    assert found.best.stats.key >= found.favorites.stats.key
    assert found.best.stats.p5 >= found.favorites.stats.p5


def test_same_rng_seed_gives_same_result(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    outcomes, found = medium_case
    again = optimize(outcomes, rng=np.random.default_rng(5))
    assert [s.card for s in again.top] == [s.card for s in found.top]


def test_more_restarts_never_hurt(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    outcomes, found = medium_case
    no_restarts = optimize(outcomes, restarts=0)
    assert found.best.stats.key >= no_restarts.best.stats.key


def test_parameter_validation(medium_case: tuple[Outcomes, OptimizeResult]) -> None:
    outcomes, _ = medium_case
    with pytest.raises(ValueError, match="top"):
        optimize(outcomes, top=0)
    with pytest.raises(ValueError, match="prune"):
        optimize(outcomes, prune=3)
    with pytest.raises(ValueError, match="restarts"):
        optimize(outcomes, restarts=-1)


# --- favorites baseline -------------------------------------------------------------------


def outcomes_with_probabilities(p: dict[int, dict[int, float]], n: int = 100) -> Outcomes:
    """Synthetic outcomes: p[category][team] = hit rate (other teams never hit)."""
    hit = np.zeros((3, n, 16), dtype=np.float32)
    for c, by_team in p.items():
        for t, rate in by_team.items():
            hit[c, : round(rate * n), t] = 1.0
    return Outcomes(teams=TEAMS, hit=hit)


def test_favorites_take_categories_in_order_without_reusing_teams() -> None:
    outcomes = outcomes_with_probabilities(
        {
            THREE_OH: {0: 0.5, 1: 0.4, 2: 0.3},
            OH_THREE: {0: 0.6, 15: 0.5, 14: 0.4, 13: 0.3},  # team 0 already used for 3-0
            ADVANCE: {1: 0.9, 15: 0.8, 3: 0.7, 4: 0.6, 5: 0.5, 6: 0.4, 7: 0.3, 8: 0.2, 9: 0.1},
        }
    )
    card = favorites_card(outcomes)
    assert card.three_oh == (0, 1)
    assert card.oh_three == (14, 15)
    assert card.advance == (3, 4, 5, 6, 7, 8)


def test_favorites_break_ties_by_lower_index() -> None:
    outcomes = outcomes_with_probabilities({})  # every probability is 0
    card = favorites_card(outcomes)
    assert card == Card(three_oh=(0, 1), oh_three=(2, 3), advance=(4, 5, 6, 7, 8, 9))
