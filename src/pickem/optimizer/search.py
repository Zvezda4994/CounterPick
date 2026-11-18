"""Search for the Pick'Em card that maximizes P(>= 5 correct) over simulated stages.

1. Prune: 3-0 and 0-3 candidates are the ``prune`` teams with the highest marginal probability
   for that category; every team is an advance candidate.
2. Greedy: add one pick at a time, maximizing E[min(correct, 5)] (P(>= 5) itself is flat at 0
   until five picks are placed), ties broken by E[correct].
3. Local search: steepest ascent on the exact key (#sims >= 5, #sims >= 6, total correct) over
   replace moves (swap a pick for an unused candidate in the same category) and exchange
   moves (two picks in different categories trade categories).
4. Restarts from the greedy card, the favorites card and random cards. Every card scored on
   the way is kept, and the best ``top`` distinct cards are reported.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pickem.optimizer.card import (
    ADVANCE,
    CATEGORIES,
    CATEGORY_SIZES,
    N_PICKS,
    OH_THREE,
    TARGET,
    THREE_OH,
    Card,
    CardStats,
    Key,
    Outcomes,
    correct_counts,
    evaluate,
    key_of,
)

Groups = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]

_EXCHANGES = ((THREE_OH, ADVANCE), (THREE_OH, OH_THREE), (ADVANCE, OH_THREE))


@dataclass(frozen=True)
class ScoredCard:
    card: Card
    stats: CardStats


@dataclass(frozen=True)
class OptimizeResult:
    top: tuple[ScoredCard, ...]
    """Best distinct cards found, best first (stats on the sims that were searched)."""
    favorites: ScoredCard
    candidates: Groups
    """Teams allowed in each category after pruning (THREE_OH, ADVANCE, OH_THREE)."""
    cards_evaluated: int

    @property
    def best(self) -> ScoredCard:
        return self.top[0]


def _ranked(p: NDArray[np.float64]) -> list[int]:
    """Team indices by probability, highest first; ties go to the lower index."""
    return sorted(range(len(p)), key=lambda t: (-p[t], t))


def favorites_card(outcomes: Outcomes) -> Card:
    """Naive baseline: top-2 P(3-0), then top-2 P(0-3), then top-6 P(3-1/3-2) of the rest."""
    p = outcomes.probabilities
    groups: list[list[int]] = [[], [], []]
    used: set[int] = set()
    for c in (THREE_OH, OH_THREE, ADVANCE):
        for t in _ranked(p[c]):
            if len(groups[c]) == CATEGORY_SIZES[c]:
                break
            if t not in used:
                groups[c].append(t)
                used.add(t)
    return Card.from_groups(groups)


def prune_candidates(outcomes: Outcomes, prune: int) -> Groups:
    p = outcomes.probabilities
    return (
        tuple(sorted(_ranked(p[THREE_OH])[:prune])),
        tuple(range(outcomes.n_teams)),
        tuple(sorted(_ranked(p[OH_THREE])[:prune])),
    )


def optimize(
    outcomes: Outcomes,
    *,
    top: int = 5,
    prune: int = 6,
    restarts: int = 20,
    rng: np.random.Generator | None = None,
) -> OptimizeResult:
    if top < 1:
        raise ValueError("top must be at least 1")
    if not 4 <= prune <= outcomes.n_teams:
        raise ValueError(f"prune must be between 4 and {outcomes.n_teams}")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    rng = np.random.default_rng(0) if rng is None else rng

    search = _Search(outcomes, prune_candidates(outcomes, prune))
    favorites = favorites_card(outcomes)
    starts = [search.greedy(), favorites]
    starts += [search.random_card(rng) for _ in range(restarts)]
    for start in starts:
        search.climb(start)

    ranked = sorted(search.pool.items(), key=lambda item: (_desc(item[1]), item[0].groups))
    return OptimizeResult(
        top=tuple(ScoredCard(card, evaluate(card, outcomes)) for card, _ in ranked[:top]),
        favorites=ScoredCard(favorites, evaluate(favorites, outcomes)),
        candidates=search.candidates,
        cards_evaluated=len(search.pool),
    )


def _desc(key: Key) -> Key:
    return (-key[0], -key[1], -key[2])


class _Search:
    def __init__(self, outcomes: Outcomes, candidates: Groups) -> None:
        self.outcomes = outcomes
        self.candidates = candidates
        self.allowed = tuple(frozenset(c) for c in candidates)
        self.x = outcomes.hit  # float32, for exact BLAS counts
        self.xi = outcomes.hit_int8
        self.n = outcomes.n_sims
        self.col_totals = self.xi.sum(axis=1, dtype=np.int64)  # (3, teams)
        self.pool: dict[Card, Key] = {}

    def greedy(self) -> Card:
        groups: list[list[int]] = [[], [], []]
        used: set[int] = set()
        counts = np.zeros(self.n, dtype=np.int8)
        for _ in range(N_PICKS):
            below = (counts < TARGET).astype(np.float32)
            best: tuple[tuple[int, int], int, int] | None = None
            for c in CATEGORIES:
                if len(groups[c]) == CATEGORY_SIZES[c]:
                    continue
                # Adding pick (c, j) raises E[min(correct, 5)] by P(correct <= 4 and hit).
                gain = np.rint(below @ self.x[c]).astype(np.int64)
                for j in self.candidates[c]:
                    if j in used or not self._can_finish(groups, used, c, j):
                        continue
                    score = (int(gain[j]), int(self.col_totals[c, j]))
                    if best is None or score > best[0]:
                        best = (score, c, j)
            assert best is not None
            _, c, j = best
            groups[c].append(j)
            used.add(j)
            counts += self.xi[c, :, j]
        return Card.from_groups(groups)

    def _can_finish(self, groups: list[list[int]], used: set[int], c: int, j: int) -> bool:
        """Whether the card can still be completed after adding pick (c, j).

        Hall's condition: the pruned 3-0 and 0-3 slots need enough unused candidates, alone
        and together, and advance slots take any unused team.
        """
        taken = used | {j}
        left = [CATEGORY_SIZES[k] - len(groups[k]) - (k == c) for k in CATEGORIES]
        free_30 = self.allowed[THREE_OH] - taken
        free_03 = self.allowed[OH_THREE] - taken
        return (
            len(free_30) >= left[THREE_OH]
            and len(free_03) >= left[OH_THREE]
            and len(free_30 | free_03) >= left[THREE_OH] + left[OH_THREE]
            and len(self.allowed[ADVANCE] - taken) >= sum(left)
        )

    def random_card(self, rng: np.random.Generator) -> Card:
        groups: list[list[int]] = [[], [], []]
        used: set[int] = set()
        for c in (THREE_OH, OH_THREE, ADVANCE):
            pool = [t for t in self.candidates[c] if t not in used]
            picked = rng.choice(pool, size=CATEGORY_SIZES[c], replace=False)
            groups[c] = [int(t) for t in picked]
            used.update(groups[c])
        return Card.from_groups(groups)

    def climb(self, card: Card) -> Card:
        """Steepest ascent from ``card``; returns the local optimum."""
        counts = correct_counts(card, self.outcomes)
        key = key_of(counts)
        self.pool[card] = key
        while True:
            best_key, best_card = key, None
            for neighbour_key, neighbour in self._neighbours(card, counts):
                self.pool[neighbour] = neighbour_key
                if neighbour_key > best_key:
                    best_key, best_card = neighbour_key, neighbour
            if best_card is None:
                return card
            card, key = best_card, best_key
            counts = correct_counts(card, self.outcomes)

    def _neighbours(self, card: Card, counts: NDArray[np.int8]) -> Iterator[tuple[Key, Card]]:
        groups = card.groups
        used = card.teams

        # Replace: for a fixed outgoing pick, score every incoming candidate at once. Hits are
        # 0/1, so #(base + x_j >= 5) = #(base >= 5) + (base == 4) . x_j.
        for c in CATEGORIES:
            free = [j for j in self.candidates[c] if j not in used]
            if not free:
                continue
            free_idx = np.asarray(free)
            for t in groups[c]:
                base = counts - self.xi[c, :, t]
                at4 = (base == TARGET - 1).astype(np.float32) @ self.x[c]
                at5 = (base == TARGET).astype(np.float32) @ self.x[c]
                n5 = np.count_nonzero(base >= TARGET) + np.rint(at4[free_idx]).astype(np.int64)
                n6 = np.count_nonzero(base > TARGET) + np.rint(at5[free_idx]).astype(np.int64)
                total = int(base.sum(dtype=np.int64)) + self.col_totals[c, free_idx]
                rest = tuple(u for u in groups[c] if u != t)
                for k, j in enumerate(free):
                    swapped = list(groups)
                    swapped[c] = (*rest, j)
                    yield (int(n5[k]), int(n6[k]), int(total[k])), Card.from_groups(swapped)

        # Exchange: picks a (category c1) and b (category c2) trade categories.
        for c1, c2 in _EXCHANGES:
            for a in groups[c1]:
                if a not in self.allowed[c2]:
                    continue
                gain_a = self.xi[c2, :, a] - self.xi[c1, :, a]
                for b in groups[c2]:
                    if b not in self.allowed[c1]:
                        continue
                    new_counts = counts + gain_a + (self.xi[c1, :, b] - self.xi[c2, :, b])
                    swapped = list(groups)
                    swapped[c1] = (*(u for u in groups[c1] if u != a), b)
                    swapped[c2] = (*(u for u in groups[c2] if u != b), a)
                    yield key_of(new_counts), Card.from_groups(swapped)
