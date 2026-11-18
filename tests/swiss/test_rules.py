from itertools import pairwise

import pytest

from pickem.swiss.rules import (
    ROUND1_PAIRING,
    ROUND45_PRIORITY_TABLE,
    Pairing,
    high_low_pairings,
    is_bo3,
    record_groups,
    round_layouts,
)
from tests.swiss.reference import TABLE


def _as_set(pairing: Pairing) -> frozenset[frozenset[int]]:
    return frozenset(frozenset(p) for p in pairing)


def test_round1_is_seed_k_vs_k_plus_8() -> None:
    assert [(a + 1, b + 1) for a, b in ROUND1_PAIRING] == [(k, k + 8) for k in range(1, 9)]


def test_priority_table_is_every_matching_of_six_exactly_once() -> None:
    assert len(ROUND45_PRIORITY_TABLE) == 15  # 5 * 3 * 1 perfect matchings of 6 teams
    for row in ROUND45_PRIORITY_TABLE:
        assert sorted(t for pair in row for t in pair) == [1, 2, 3, 4, 5, 6]
        assert all(a < b for a, b in row)
        assert [a for a, _ in row] == sorted(a for a, _ in row)  # pairs in seed order
    assert len({_as_set(row) for row in ROUND45_PRIORITY_TABLE}) == 15


def test_priority_table_matches_independent_transcription() -> None:
    assert [[(a - 1, b - 1) for a, b in row] for row in ROUND45_PRIORITY_TABLE] == TABLE


def test_high_low_order_for_four() -> None:
    assert high_low_pairings(4) == (
        ((0, 3), (1, 2)),
        ((0, 2), (1, 3)),
        ((0, 1), (2, 3)),
    )


@pytest.mark.parametrize(("size", "count"), [(2, 1), (4, 3), (6, 15), (8, 105)])
def test_high_low_lists_every_matching_once(size: int, count: int) -> None:
    candidates = high_low_pairings(size)
    assert len(candidates) == count
    assert len({_as_set(c) for c in candidates}) == count
    assert candidates[0] == tuple((k, size - 1 - k) for k in range(size // 2))
    for c in candidates:
        assert all(a < b for a, b in c)
        assert [a for a, _ in c] == sorted(a for a, _ in c)


def test_high_low_is_lexicographic_on_opponents() -> None:
    """The top seed's opponent moves up from the bottom only after every completion of the
    rest of the group has been tried (the lookahead reading of L170)."""
    opponents_of_top = [c[0][1] for c in high_low_pairings(8)]
    assert opponents_of_top == sorted(opponents_of_top, reverse=True)
    assert [len(list(g)) for g in _runs(opponents_of_top)] == [15] * 7


def _runs(values: list[int]) -> list[list[int]]:
    runs: list[list[int]] = [[values[0]]]
    for prev, cur in pairwise(values):
        if cur == prev:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    return runs


@pytest.mark.parametrize(
    ("record", "bo3"),
    [
        ((0, 0), False),
        ((1, 0), False),
        ((0, 1), False),
        ((1, 1), False),
        ((2, 0), True),
        ((0, 2), True),
        ((2, 1), True),
        ((1, 2), True),
        ((2, 2), True),
    ],
)
def test_bo3_only_when_a_team_can_advance_or_be_eliminated(
    record: tuple[int, int], bo3: bool
) -> None:
    assert is_bo3(record) is bo3


def test_record_groups_have_fixed_sizes() -> None:
    def sizes(rounds: int) -> list[tuple[str, int]]:
        return [(f"{w}-{lo}", size) for (w, lo), _, size in record_groups(rounds)]

    assert sizes(0) == [("0-0", 16)]
    assert sizes(1) == [("1-0", 8), ("0-1", 8)]
    assert sizes(2) == [("2-0", 4), ("1-1", 8), ("0-2", 4)]
    assert sizes(3) == [("3-0", 2), ("2-1", 6), ("1-2", 6), ("0-3", 2)]
    assert sizes(4) == [("3-0", 2), ("3-1", 3), ("2-2", 6), ("1-3", 3), ("0-3", 2)]
    assert sizes(5) == [
        ("3-0", 2),
        ("3-1", 3),
        ("3-2", 3),
        ("2-3", 3),
        ("1-3", 3),
        ("0-3", 2),
    ]


def test_round_layouts() -> None:
    layouts = round_layouts()
    assert [layout.n_matches for layout in layouts] == [8, 8, 8, 6, 3]
    assert [layout.best_of_by_slot for layout in layouts] == [
        (1,) * 8,
        (1,) * 8,
        (3, 3, 1, 1, 1, 1, 3, 3),
        (3,) * 6,
        (3,) * 3,
    ]
    assert [len(g.candidates) for layout in layouts for g in layout.groups] == [
        1,  # round 1: fixed
        105,
        105,  # round 2: 1-0 and 0-1 groups of 8
        3,
        105,
        3,  # round 3: 2-0, 1-1, 0-2
        15,
        15,  # round 4: table
        15,  # round 5: table
    ]
