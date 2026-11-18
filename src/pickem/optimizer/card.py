"""Pick'Em cards and how they score against simulated Swiss stages.

Scoring (project spec; Valve's rules repo has no Pick'Em section, so re-check the in-game
rules before each Major):
- 2 teams picked to go 3-0 score only on a 3-0 record.
- 2 teams picked to go 0-3 score only on a 0-3 record.
- 6 teams picked to advance score only on 3-1 or 3-2; a team that goes 3-0 is a miss.
- A team can appear in only one category.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pickem.swiss import SwissResult

THREE_OH, ADVANCE, OH_THREE = 0, 1, 2
CATEGORIES = (THREE_OH, ADVANCE, OH_THREE)
CATEGORY_NAMES = ("3-0", "Advance", "0-3")
CATEGORY_SIZES = (2, 6, 2)
N_PICKS = sum(CATEGORY_SIZES)
TARGET = 5
"""Correct picks needed for the stage reward."""

Key = tuple[int, int, int]
"""Exact ranking key over a set of sims: (#sims >= 5 correct, #sims >= 6, total correct)."""

_MAX_EXACT_SIMS = 2**24  # float32 sums of 0/1 values are exact up to here


@dataclass(frozen=True)
class Card:
    """Ten picks as team indices into ``Outcomes.teams``. Each group is stored sorted."""

    three_oh: tuple[int, ...]
    advance: tuple[int, ...]
    oh_three: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = tuple(
            tuple(sorted(int(t) for t in g)) for g in (self.three_oh, self.advance, self.oh_three)
        )
        for name, group, size in zip(CATEGORY_NAMES, groups, CATEGORY_SIZES, strict=True):
            if len(group) != size:
                raise ValueError(f"{name} needs {size} teams, got {len(group)}")
        teams = [t for g in groups for t in g]
        if len(set(teams)) != N_PICKS:
            raise ValueError("a team can be picked in only one category")
        if min(teams) < 0:
            raise ValueError("team indices must be non-negative")
        object.__setattr__(self, "three_oh", groups[THREE_OH])
        object.__setattr__(self, "advance", groups[ADVANCE])
        object.__setattr__(self, "oh_three", groups[OH_THREE])

    @classmethod
    def from_groups(cls, groups: Sequence[Iterable[int]]) -> Card:
        """Build from groups indexed by category (THREE_OH, ADVANCE, OH_THREE)."""
        return cls(
            three_oh=tuple(groups[THREE_OH]),
            advance=tuple(groups[ADVANCE]),
            oh_three=tuple(groups[OH_THREE]),
        )

    @property
    def groups(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Picks indexed by category (THREE_OH, ADVANCE, OH_THREE)."""
        return (self.three_oh, self.advance, self.oh_three)

    @property
    def picks(self) -> tuple[tuple[int, int], ...]:
        """(category, team) for all ten picks."""
        return tuple((c, t) for c, group in enumerate(self.groups) for t in group)

    @property
    def teams(self) -> frozenset[int]:
        return frozenset(t for g in self.groups for t in g)

    def describe(self, names: Sequence[str]) -> dict[str, list[str]]:
        return {CATEGORY_NAMES[c]: [names[t] for t in group] for c, group in enumerate(self.groups)}


@dataclass(frozen=True)
class Outcomes:
    """Per-sim hit indicators for each pick category, built once from a ``SwissResult``."""

    teams: tuple[str, ...]
    hit: NDArray[np.float32]
    """(3, n, teams): hit[c, i, t] = 1 if team t's record in sim i scores a category-c pick."""

    @classmethod
    def from_result(cls, result: SwissResult) -> Outcomes:
        if result.n_sims > _MAX_EXACT_SIMS:
            raise ValueError(f"at most {_MAX_EXACT_SIMS} sims per Outcomes")
        wins, losses = result.wins, result.losses
        hit = np.stack(
            [
                (wins == 3) & (losses == 0),  # 3-0
                (wins == 3) & (losses > 0),  # 3-1 or 3-2
                (wins == 0) & (losses == 3),  # 0-3
            ]
        ).astype(np.float32)
        hit.setflags(write=False)
        return cls(teams=result.teams, hit=hit)

    @property
    def n_sims(self) -> int:
        return int(self.hit.shape[1])

    @property
    def n_teams(self) -> int:
        return int(self.hit.shape[2])

    @cached_property
    def hit_int8(self) -> NDArray[np.int8]:
        return self.hit.astype(np.int8)

    @cached_property
    def probabilities(self) -> NDArray[np.float64]:
        """(3, teams): P(a category-c pick on team t is correct)."""
        result: NDArray[np.float64] = self.hit.mean(axis=1, dtype=np.float64)
        return result


def correct_counts(card: Card, outcomes: Outcomes) -> NDArray[np.int8]:
    """(n,) number of correct picks in each sim."""
    if max(card.teams) >= outcomes.n_teams:
        raise ValueError("card uses a team index outside the outcomes")
    categories, teams = zip(*card.picks, strict=True)
    counts: NDArray[np.int8] = outcomes.hit_int8[list(categories), :, list(teams)].sum(
        axis=0, dtype=np.int8
    )
    return counts


def key_of(counts: NDArray[np.int8]) -> Key:
    return (
        int(np.count_nonzero(counts >= TARGET)),
        int(np.count_nonzero(counts >= TARGET + 1)),
        int(counts.sum(dtype=np.int64)),
    )


@dataclass(frozen=True)
class CardStats:
    distribution: NDArray[np.int64]
    """(11,) number of sims with exactly k correct picks."""

    @classmethod
    def from_counts(cls, counts: NDArray[np.int8]) -> CardStats:
        return cls(distribution=np.bincount(counts, minlength=N_PICKS + 1).astype(np.int64))

    @property
    def n_sims(self) -> int:
        return int(self.distribution.sum())

    def p_at_least(self, k: int) -> float:
        return float(self.distribution[k:].sum() / self.n_sims)

    @property
    def p5(self) -> float:
        return self.p_at_least(TARGET)

    @property
    def p6(self) -> float:
        return self.p_at_least(TARGET + 1)

    @property
    def se5(self) -> float:
        """Monte Carlo standard error of ``p5``."""
        return float(np.sqrt(self.p5 * (1.0 - self.p5) / self.n_sims))

    @property
    def expected(self) -> float:
        return float(np.arange(N_PICKS + 1) @ self.distribution / self.n_sims)

    @property
    def key(self) -> Key:
        d = self.distribution
        return (int(d[TARGET:].sum()), int(d[TARGET + 1 :].sum()), int(np.arange(11) @ d))


def evaluate(card: Card, outcomes: Outcomes) -> CardStats:
    return CardStats.from_counts(correct_counts(card, outcomes))


def marginals(outcomes: Outcomes) -> pd.DataFrame:
    """Per-team probability that each kind of pick on that team is correct."""
    p = outcomes.probabilities
    return pd.DataFrame(
        {
            "P(3-0)": p[THREE_OH],
            "P(3-1/3-2)": p[ADVANCE],
            "P(0-3)": p[OH_THREE],
            "P(qualify)": p[THREE_OH] + p[ADVANCE],
        },
        index=pd.Index(outcomes.teams, name="team"),
    )
