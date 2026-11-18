import numpy as np
import pytest

from pickem.optimizer import Card, CardStats, Outcomes, correct_counts, evaluate, marginals
from pickem.optimizer.card import key_of
from pickem.swiss import StageInput, SwissResult, simulate_swiss
from tests.optimizer import brute_force

TEAMS = tuple(f"T{i:02d}" for i in range(16))

# One hand-written final standing: index -> (wins, losses).
RECORDS = {
    0: (3, 0), 1: (3, 0),
    2: (3, 1), 3: (3, 1), 4: (3, 1),
    5: (3, 2), 6: (3, 2), 7: (3, 2),
    8: (2, 3), 9: (2, 3), 10: (2, 3),
    11: (1, 3), 12: (1, 3), 13: (1, 3),
    14: (0, 3), 15: (0, 3),
}  # fmt: skip


def result_from_records(*sims: dict[int, tuple[int, int]]) -> SwissResult:
    wins = np.array([[s[t][0] for t in range(16)] for s in sims], dtype=np.int8)
    losses = np.array([[s[t][1] for t in range(16)] for s in sims], dtype=np.int8)
    zeros = np.zeros_like(wins)
    return SwissResult(
        teams=TEAMS,
        wins=wins,
        losses=losses,
        difficulty=zeros,
        final_seed=zeros,
        rematches=np.zeros(len(sims), dtype=np.int8),
    )


def sim_outcomes(n: int, seed: int) -> tuple[SwissResult, Outcomes]:
    rng = np.random.default_rng(seed)
    strength = rng.normal(size=16)
    gap = strength[:, None] - strength[None, :]
    p = 1.0 / (1.0 + np.exp(-gap))
    result = simulate_swiss(StageInput(TEAMS, tuple(range(1, 17)), p, p), n, rng)
    return result, Outcomes.from_result(result)


def test_each_category_scores_only_its_own_records() -> None:
    outcomes = Outcomes.from_result(result_from_records(RECORDS))
    card = Card(
        three_oh=(0, 2),  # 3-0 hit; 3-1 is a miss for a 3-0 pick
        advance=(1, 3, 4, 5, 8, 11),  # 3-0 team is a miss; 3-1, 3-1, 3-2 hit; 2-3, 1-3 miss
        oh_three=(14, 13),  # 0-3 hit; 1-3 is a miss
    )
    assert correct_counts(card, outcomes).tolist() == [1 + 3 + 1]


def test_perfect_and_empty_cards() -> None:
    outcomes = Outcomes.from_result(result_from_records(RECORDS))
    perfect = Card(three_oh=(0, 1), advance=(2, 3, 4, 5, 6, 7), oh_three=(14, 15))
    backwards = Card(three_oh=(14, 15), advance=(0, 1, 8, 9, 10, 11), oh_three=(2, 3))
    assert correct_counts(perfect, outcomes).tolist() == [10]
    assert correct_counts(backwards, outcomes).tolist() == [0]


@pytest.mark.parametrize(
    ("three_oh", "advance", "oh_three", "message"),
    [
        ((0,), (1, 2, 3, 4, 5, 6), (7, 8), "3-0 needs 2"),
        ((0, 1), (2, 3, 4, 5, 6), (7, 8), "Advance needs 6"),
        ((0, 1), (2, 3, 4, 5, 6, 7), (8, 9, 10), "0-3 needs 2"),
        ((0, 1), (1, 3, 4, 5, 6, 7), (8, 9), "only one category"),
        ((0, 0), (2, 3, 4, 5, 6, 7), (8, 9), "only one category"),
        ((-1, 1), (2, 3, 4, 5, 6, 7), (8, 9), "non-negative"),
    ],
)
def test_invalid_cards_rejected(
    three_oh: tuple[int, ...], advance: tuple[int, ...], oh_three: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Card(three_oh=three_oh, advance=advance, oh_three=oh_three)


def test_card_groups_are_sorted_so_equal_cards_compare_equal() -> None:
    a = Card(three_oh=(1, 0), advance=(7, 6, 5, 4, 3, 2), oh_three=(9, 8))
    b = Card(three_oh=(0, 1), advance=(2, 3, 4, 5, 6, 7), oh_three=(8, 9))
    assert a == b and hash(a) == hash(b)


def test_marginals_sum_to_slot_counts() -> None:
    _, outcomes = sim_outcomes(5_000, seed=1)
    table = marginals(outcomes)
    assert table["P(3-0)"].sum() == pytest.approx(2)
    assert table["P(3-1/3-2)"].sum() == pytest.approx(6)
    assert table["P(0-3)"].sum() == pytest.approx(2)
    assert table["P(qualify)"].sum() == pytest.approx(8)
    assert list(table.index) == list(TEAMS)


def test_counts_match_independent_scoring_and_stats_are_consistent() -> None:
    result, outcomes = sim_outcomes(3_000, seed=2)
    rng = np.random.default_rng(0)
    for _ in range(25):
        perm = [int(t) for t in rng.permutation(16)]
        card = Card(three_oh=tuple(perm[:2]), advance=tuple(perm[2:8]), oh_three=tuple(perm[8:10]))
        counts = correct_counts(card, outcomes)
        stats = evaluate(card, outcomes)
        oracle = brute_force.score(result.wins, result.losses, card.groups)
        assert key_of(counts) == stats.key == oracle
        assert stats.n_sims == 3_000
        assert stats.p5 == pytest.approx((counts >= 5).mean())
        assert stats.p6 == pytest.approx((counts >= 6).mean())
        assert stats.expected == pytest.approx(counts.mean())
        assert stats.se5 == pytest.approx(np.sqrt(stats.p5 * (1 - stats.p5) / 3_000))


def test_card_stats_from_counts() -> None:
    stats = CardStats.from_counts(np.array([5, 6, 4, 10, 0], dtype=np.int8))
    assert stats.distribution.tolist() == [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]
    assert (stats.p5, stats.p6, stats.expected) == (0.6, 0.4, 5.0)
    assert stats.key == (3, 2, 25)


def test_card_must_fit_the_outcomes() -> None:
    _, outcomes = sim_outcomes(10, seed=3)
    card = Card(three_oh=(0, 1), advance=(2, 3, 4, 5, 6, 7), oh_three=(8, 16))
    with pytest.raises(ValueError, match="outside"):
        correct_counts(card, outcomes)
