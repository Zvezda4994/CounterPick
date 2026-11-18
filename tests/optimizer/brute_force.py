"""Exhaustive Pick'Em oracle: scores every legal card. Shares no code with pickem.optimizer.

A card is a 3-0 pair, a 0-3 pair and 6 advance teams, all distinct:
C(16,2) * C(14,2) * C(12,6) = 10,090,080 cards. The loop runs over the 1,820 four-team sets
that fill the 3-0 and 0-3 slots (6 ways to split each); every advance set drawn from the
other 12 teams is scored at once from precomputed per-set hit sums.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

Key = tuple[int, int, int]
CardTuple = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]  # (3-0, advance, 0-3)


def hit_matrices(
    wins: NDArray[np.int8], losses: NDArray[np.int8]
) -> tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]]:
    """(teams, n) indicators: a 3-0 pick hits, an advance pick hits, a 0-3 pick hits."""
    three_oh = ((wins == 3) & (losses == 0)).T.astype(np.int8)
    advance = ((wins == 3) & ((losses == 1) | (losses == 2))).T.astype(np.int8)
    oh_three = ((wins == 0) & (losses == 3)).T.astype(np.int8)
    return three_oh, advance, oh_three


def score(wins: NDArray[np.int8], losses: NDArray[np.int8], card: CardTuple) -> Key:
    three_oh, advance, oh_three = hit_matrices(wins, losses)
    correct = (
        three_oh[list(card[0])].sum(axis=0)
        + advance[list(card[1])].sum(axis=0)
        + oh_three[list(card[2])].sum(axis=0)
    )
    return (int((correct >= 5).sum()), int((correct >= 6).sum()), int(correct.sum()))


def brute_force_top(
    wins: NDArray[np.int8], losses: NDArray[np.int8], top: int = 5
) -> list[tuple[Key, CardTuple]]:
    """The ``top`` best cards over all 10,090,080, ranked by (#>=5, #>=6, total correct)."""
    three_oh, advance, oh_three = hit_matrices(wins, losses)
    n_teams, n = advance.shape
    big = 10 * n + 1  # strictly above any total, so the composite ranks lexicographically

    sixes = np.array(list(combinations(range(n_teams), 6)))  # (8008, 6)
    six_masks = (1 << sixes).sum(axis=1)
    six_sums = advance[sixes].sum(axis=1, dtype=np.int8)  # (8008, n)
    six_totals = six_sums.sum(axis=1, dtype=np.int64)

    found: list[tuple[Key, CardTuple]] = []
    for four in combinations(range(n_teams), 4):
        mask = sum(1 << t for t in four)
        valid = np.flatnonzero((six_masks & mask) == 0)  # the 924 advance sets
        sums = six_sums[valid]
        splits = list(combinations(four, 2))  # which two of the four go 3-0
        rests = [tuple(t for t in four if t not in pair) for pair in splits]
        partial = np.stack(
            [
                three_oh[list(pair)].sum(axis=0) + oh_three[list(rest)].sum(axis=0)
                for pair, rest in zip(splits, rests, strict=True)
            ]
        ).astype(np.int8)  # (6, n)
        total = sums[None, :, :] + partial[:, None, :]  # (6, 924, n)
        n5 = np.count_nonzero(total >= 5, axis=2).astype(np.int64)
        n6 = np.count_nonzero(total >= 6, axis=2).astype(np.int64)
        tot = six_totals[valid][None, :] + partial.sum(axis=1, dtype=np.int64)[:, None]
        composite = (n5 * big + n6) * big + tot
        flat = composite.ravel()
        best = np.argpartition(flat, -top)[-top:] if flat.size > top else np.arange(flat.size)
        for idx in best:
            s, m = divmod(int(idx), len(valid))
            card = (splits[s], tuple(int(t) for t in sixes[valid[m]]), rests[s])
            found.append(((int(n5[s, m]), int(n6[s, m]), int(tot[s, m])), card))
    found.sort(key=lambda item: item[0], reverse=True)
    return found[:top]
