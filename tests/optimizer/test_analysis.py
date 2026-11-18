import numpy as np
import pytest

from pickem.optimizer import Card, Outcomes, evaluate, optimize
from pickem.optimizer.analysis import (
    approach_label,
    difference_path,
    overlap,
    record_distribution,
)
from pickem.swiss import StageInput, simulate_swiss

NAMES = tuple(f"T{i:02d}" for i in range(16))
CHALK = Card(three_oh=(0, 1), oh_three=(14, 15), advance=(2, 3, 4, 5, 6, 7))


@pytest.fixture(scope="module")
def stage() -> tuple[Outcomes, np.ndarray]:
    strength = np.linspace(0.9, -0.9, 16)
    gap = strength[:, None] - strength[None, :]
    p = 1.0 / (1.0 + np.exp(-gap))
    result = simulate_swiss(
        StageInput(NAMES, tuple(range(1, 17)), p, p * p * (3 - 2 * p)),
        20_000,
        np.random.default_rng(4),
    )
    return Outcomes.from_result(result), record_distribution(result)


def test_overlap() -> None:
    card = Card(three_oh=(0, 9), oh_three=(14, 15), advance=(1, 3, 4, 5, 6, 7))
    assert overlap(card, CHALK) == (8, 1)  # team 1 is on both cards, in another slot
    assert overlap(CHALK, CHALK) == (10, 0)


def test_difference_costs_add_up_to_the_gap(stage: tuple[Outcomes, np.ndarray]) -> None:
    outcomes, _ = stage
    optimal = optimize(outcomes, restarts=3).best.card
    card = Card(three_oh=(7, 0), oh_three=(15, 10), advance=(6, 1, 3, 2, 8, 4))
    path = difference_path(card, optimal, outcomes, NAMES)
    gap = evaluate(optimal, outcomes).p5 - evaluate(card, outcomes).p5
    assert sum(d.gain for d in path) == pytest.approx(gap, abs=1e-12)
    assert 1 <= len(path) <= 10
    assert all("→" in d.change for d in path)
    assert difference_path(optimal, optimal, outcomes, NAMES) == []


def test_record_distribution_rows_sum_to_one(stage: tuple[Outcomes, np.ndarray]) -> None:
    _, records = stage
    assert records.shape == (16, 6)
    assert records.sum(axis=1) == pytest.approx(np.ones(16))


def probabilities() -> np.ndarray:
    """3-0 and advance favour low indices, 0-3 favours high ones."""
    p = np.zeros((3, 16))
    p[0] = np.linspace(0.3, 0.01, 16)  # 3-0
    p[1] = np.linspace(0.5, 0.1, 16)  # advance
    p[2] = np.linspace(0.01, 0.3, 16)  # 0-3
    return p


OPTIMAL = Card(three_oh=(0, 2), oh_three=(14, 15), advance=(1, 3, 4, 5, 6, 7))


@pytest.mark.parametrize(
    ("card", "label"),
    [
        (OPTIMAL, "optimizer pick"),
        (CHALK, "seed chalk"),
        (Card(three_oh=(0, 1), oh_three=(14, 15), advance=(2, 3, 4, 5, 6, 8)), "chalk"),
        (
            Card(three_oh=(0, 9), oh_three=(14, 15), advance=(2, 3, 4, 5, 6, 7)),
            "one dark horse to 3-0",
        ),
        (
            Card(three_oh=(0, 9), oh_three=(14, 15), advance=(1, 2, 3, 4, 5, 6)),
            "one dark horse to 3-0; 3-0 favourite in Advance",
        ),
        (
            Card(three_oh=(8, 9), oh_three=(10, 11), advance=(2, 3, 4, 5, 6, 7)),
            "two dark horses to 3-0; two surprise 0-3s; contrarian advances",
        ),
        (
            Card(three_oh=(0, 1), oh_three=(14, 15), advance=(2, 3, 10, 11, 12, 13)),
            "contrarian advances",
        ),
        (
            Card(three_oh=(8, 9), oh_three=(14, 15), advance=(0, 1, 2, 3, 4, 5)),
            "two dark horses to 3-0; 3-0 favourites in Advance",
        ),
    ],
)
def test_approach_labels(card: Card, label: str) -> None:
    assert approach_label(card, probabilities(), OPTIMAL, CHALK)[0] == label
