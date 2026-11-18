"""Per-card analysis for the leaderboard: overlap, differences from the optimal card, approach
labels, and (after the stage) swing matches and how likely the actual outcome was."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pickem.optimizer.card import (
    ADVANCE,
    CATEGORIES,
    CATEGORY_NAMES,
    CATEGORY_SIZES,
    OH_THREE,
    TARGET,
    THREE_OH,
    Card,
    Outcomes,
    correct_counts,
    evaluate,
)
from pickem.swiss import StageInput, SwissResult, simulate_swiss
from pickem.swiss.replay import MatchResult

RECORDS = ("3-0", "3-1", "3-2", "2-3", "1-3", "0-3")


def overlap(card: Card, optimal: Card) -> tuple[int, int]:
    """(picks identical to the optimal card, teams on both cards but in a different category)."""
    identical = sum(1 for c in CATEGORIES for t in card.groups[c] if t in optimal.groups[c])
    return identical, len(card.teams & optimal.teams) - identical


@dataclass(frozen=True)
class Difference:
    change: str
    gain: float
    """P(>=5) change from making this fix, applied after the fixes listed before it."""


def _moves_toward(card: Card, target: Card, names: Sequence[str]) -> Iterator[tuple[str, Card]]:
    """Single legal moves that put one more of ``target``'s picks in place."""
    groups = [list(g) for g in card.groups]
    for c in CATEGORIES:
        label = CATEGORY_NAMES[c]
        for wrong in (t for t in groups[c] if t not in target.groups[c]):
            for needed in (t for t in target.groups[c] if t not in groups[c]):
                new = [list(g) for g in groups]
                new[c] = [needed if u == wrong else u for u in new[c]]
                other = next((k for k in CATEGORIES if k != c and needed in groups[k]), None)
                if other is None:
                    change = f"{label}: {names[wrong]} → {names[needed]}"
                else:
                    new[other] = [wrong if u == needed else u for u in new[other]]
                    change = (
                        f"{label}: {names[wrong]} → {names[needed]} ({names[needed]} leaves "
                        f"{CATEGORY_NAMES[other]}, {names[wrong]} takes its place)"
                    )
                yield change, Card.from_groups(new)


def difference_path(
    card: Card, optimal: Card, outcomes: Outcomes, names: Sequence[str]
) -> list[Difference]:
    """Turn ``card`` into ``optimal`` one legal move at a time, always taking the move that
    leaves the best card. The gains add up exactly to P(>=5)(optimal) - P(>=5)(card)."""
    steps: list[Difference] = []
    current, current_p5 = card, evaluate(card, outcomes).p5
    while current != optimal:
        options = [
            (evaluate(new, outcomes).key, change, new)
            for change, new in _moves_toward(current, optimal, names)
        ]
        key, change, new = max(options, key=lambda option: option[0])
        new_p5 = key[0] / outcomes.n_sims
        steps.append(Difference(change=change, gain=new_p5 - current_p5))
        current, current_p5 = new, new_p5
    return steps


def _ranked(p: NDArray[np.float64]) -> list[int]:
    return sorted(range(len(p)), key=lambda t: (-p[t], t))


def approach_label(
    card: Card, probabilities: NDArray[np.float64], optimal: Card, chalk: Card
) -> tuple[str, tuple[str, ...]]:
    """A short, rule-based description of how the card was built.

    Teams are ranked by the model's chance that a pick on them is correct. A 3-0 pick is a
    favourite if its team is in the top 2 for 3-0; a 0-3 pick if it is in the top 2 for 0-3
    among teams not picked to go 3-0; an advance pick if it is in the top 6 for advance among
    teams not used in the card's 3-0 or 0-3 slots (strong teams on a 3-0 slot would otherwise
    crowd every card's advance picks out of the top 6).
    """
    if card == optimal:
        return "optimizer pick", ("optimizer pick",)
    if card == chalk:
        return "seed chalk", ("seed chalk",)

    def top(category: int, exclude: set[int]) -> set[int]:
        ranked = [t for t in _ranked(probabilities[category]) if t not in exclude]
        return set(ranked[: CATEGORY_SIZES[category]])

    fav_30 = top(THREE_OH, set())
    fav_03 = top(OH_THREE, set(card.three_oh))
    fav_adv = top(ADVANCE, set(card.three_oh) | set(card.oh_three))
    parts: list[str] = []
    long_30 = sum(t not in fav_30 for t in card.three_oh)
    if long_30:
        parts.append("one dark horse to 3-0" if long_30 == 1 else "two dark horses to 3-0")
    long_03 = sum(t not in fav_03 for t in card.oh_three)
    if long_03:
        parts.append("one surprise 0-3" if long_03 == 1 else "two surprise 0-3s")
    if sum(t in fav_adv for t in card.advance) <= 4:
        parts.append("contrarian advances")
    stranded = fav_30 & set(card.advance)
    if stranded:
        parts.append(
            "3-0 favourite in Advance" if len(stranded) == 1 else "3-0 favourites in Advance"
        )
    return ("; ".join(parts) if parts else "chalk"), tuple(parts)


def record_distribution(result: SwissResult) -> NDArray[np.float64]:
    """(teams, 6): share of sims ending at each record in ``RECORDS``."""
    table = np.zeros((len(result.teams), len(RECORDS)))
    for k, record in enumerate(RECORDS):
        w, lo = (int(x) for x in record.split("-"))
        table[:, k] = ((result.wins == w) & (result.losses == lo)).mean(axis=0)
    return table


def single_outcome(teams: tuple[str, ...], wins: Sequence[int], losses: Sequence[int]) -> Outcomes:
    """One finished stage as ``Outcomes`` (one sim), so cards score with ``correct_counts``."""
    w = np.array([wins], dtype=np.int8)
    lo = np.array([losses], dtype=np.int8)
    zeros = np.zeros_like(w)
    return Outcomes.from_result(
        SwissResult(teams=teams, wins=w, losses=lo, difficulty=zeros, final_seed=zeros,
                    rematches=np.zeros(1, dtype=np.int8))
    )  # fmt: skip


@dataclass(frozen=True)
class Swing:
    round: int
    team1: str
    team2: str
    winner: str
    p_clear_if_flipped: float
    certain: bool


def conditional_outcomes(
    stage: StageInput,
    rounds: Sequence[Sequence[MatchResult]],
    flipped: MatchResult | None,
    n_sims: int,
    seed: int,
) -> tuple[Outcomes, bool]:
    """Replay the stage with every real result fixed (``flipped`` reversed). Pairings that never
    happened in reality use the stage's probabilities. Returns the outcomes and whether every sim
    came out the same (no unknown pairing was needed)."""
    index = {t: i for i, t in enumerate(stage.teams)}
    p1, p3 = stage.p_bo1.copy(), stage.p_bo3.copy()
    for matches in rounds:
        for m in matches:
            winner, loser = (m.loser, m.winner) if m == flipped else (m.winner, m.loser)
            w, lo = index[winner], index[loser]
            for p in (p1, p3):
                p[w, lo], p[lo, w] = 1.0, 0.0
    result = simulate_swiss(
        StageInput(stage.teams, stage.seeds, p1, p3), n_sims, np.random.default_rng(seed)
    )
    certain = bool(
        (result.wins == result.wins[0]).all() and (result.losses == result.losses[0]).all()
    )
    return Outcomes.from_result(result), certain


def swing_matches(
    stage: StageInput,
    rounds: Sequence[Sequence[MatchResult]],
    cards: Sequence[Card],
    cleared: Sequence[bool],
    n_sims: int = 4000,
    seed: int = 2,
) -> list[list[Swing]]:
    """Per card, the real results that, reversed, flip whether the card clears in most replays."""
    swings: list[list[Swing]] = [[] for _ in cards]
    for r, matches in enumerate(rounds, start=1):
        for m in matches:
            outcomes, certain = conditional_outcomes(stage, rounds, m, n_sims, seed)
            for k, card in enumerate(cards):
                p_clear = float((correct_counts(card, outcomes) >= TARGET).mean())
                if abs(p_clear - float(cleared[k])) > 0.5:
                    swings[k].append(Swing(r, m.team1, m.team2, m.winner, p_clear, certain))
    return swings
