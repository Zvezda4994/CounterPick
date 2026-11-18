"""Score a hand-picked card against the optimizer's best and find which picks cost the most."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from pickem.optimizer.card import (
    ADVANCE,
    CATEGORIES,
    CATEGORY_NAMES,
    OH_THREE,
    TARGET,
    THREE_OH,
    Card,
    CardStats,
    Outcomes,
    correct_counts,
    evaluate,
)

_LABELS = {
    "3-0": THREE_OH,
    "30": THREE_OH,
    "0-3": OH_THREE,
    "03": OH_THREE,
    "advance": ADVANCE,
    "adv": ADVANCE,
}


def _compact(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def resolve_team(name: str, teams: Sequence[str]) -> int:
    """Exact name (case ignored), else unique initials ("NiP"), else unique prefix, else unique
    prefix ignoring spaces and punctuation ("100T" -> "100 Thieves")."""
    key = name.strip().casefold()
    if not key:
        raise ValueError("empty team name")
    exact = [i for i, t in enumerate(teams) if t.casefold() == key]
    if exact:
        return exact[0]
    initials = [
        i for i, t in enumerate(teams) if "".join(w[0] for w in t.split()).casefold() == key
    ]
    prefix = [i for i, t in enumerate(teams) if t.casefold().startswith(key)]
    compact = [
        i for i, t in enumerate(teams) if _compact(key) and _compact(t).startswith(_compact(key))
    ]
    for found in (initials, prefix, compact):
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise ValueError(f"{name.strip()!r} could be {', '.join(teams[i] for i in found)}")
    raise ValueError(f"unknown team {name.strip()!r}; the teams are {', '.join(teams)}")


def parse_card(text: str, teams: Sequence[str]) -> Card:
    """``"3-0: A, B; 0-3: C, D; advance: E, F, G, H, I, J"`` (categories in any order)."""
    groups: dict[int, list[int]] = {}
    for part in text.split(";"):
        if not part.strip():
            continue
        label, sep, names = part.partition(":")
        if not sep:
            raise ValueError(f"expected 'category: team, team' in {part.strip()!r}")
        category = _LABELS.get(label.strip().casefold())
        if category is None:
            raise ValueError(f"unknown category {label.strip()!r}; use 3-0, 0-3 or advance")
        if category in groups:
            raise ValueError(f"{CATEGORY_NAMES[category]} is given twice")
        groups[category] = [resolve_team(n, teams) for n in names.split(",") if n.strip()]
    missing = [CATEGORY_NAMES[c] for c in (THREE_OH, OH_THREE, ADVANCE) if c not in groups]
    if missing:
        raise ValueError(f"the card is missing {', '.join(missing)}")
    return Card.from_groups([groups[THREE_OH], groups[ADVANCE], groups[OH_THREE]])


@dataclass(frozen=True)
class Comparison:
    yours: CardStats
    best: CardStats
    diff_p5: float
    """P(>=5) of your card minus the best card's, on the same sims."""
    diff_se: float
    """Paired standard error of ``diff_p5``."""


def compare_cards(card: Card, best: Card, outcomes: Outcomes) -> Comparison:
    hit_yours = (correct_counts(card, outcomes) >= TARGET).astype(np.float64)
    hit_best = (correct_counts(best, outcomes) >= TARGET).astype(np.float64)
    diff = hit_yours - hit_best
    return Comparison(
        yours=evaluate(card, outcomes),
        best=evaluate(best, outcomes),
        diff_p5=float(diff.mean()),
        diff_se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
    )


@dataclass(frozen=True)
class PickCost:
    category: int
    team: int
    p_correct: float
    """Chance this pick alone is correct (evaluation sims)."""
    change: str | None
    """Best single change to this pick, or None if no change helps."""
    fixed: Card | None
    gain: float
    """P(>=5) gained by that change on the evaluation sims (0 if none)."""


def _changes(
    card: Card, category: int, team: int, names: Sequence[str]
) -> Iterator[tuple[str, Card]]:
    """Every card that differs from ``card`` only in the pick (category, team)."""
    groups = [list(g) for g in card.groups]
    used = card.teams
    label = CATEGORY_NAMES[category]
    for other in range(len(names)):
        if other in used:
            continue
        new = [list(g) for g in groups]
        new[category] = [other if u == team else u for u in new[category]]
        yield f"{label} {names[team]} -> {names[other]}", Card.from_groups(new)
    for target in CATEGORIES:
        if target == category:
            continue
        for partner in groups[target]:
            new = [list(g) for g in groups]
            new[category] = [partner if u == team else u for u in new[category]]
            new[target] = [team if u == partner else u for u in new[target]]
            yield (
                f"{names[team]} to {CATEGORY_NAMES[target]}, {names[partner]} to {label}",
                Card.from_groups(new),
            )


def pick_costs(
    card: Card, search: Outcomes, evaluation: Outcomes, names: Sequence[str]
) -> list[PickCost]:
    """For each pick, the best single change: chosen on ``search``, measured on ``evaluation``.

    Choosing on one set of sims and measuring on another keeps the gains honest. Costliest first.
    """
    base_key = evaluate(card, search).key
    base_p5 = evaluate(card, evaluation).p5
    costs = []
    for category, team in card.picks:
        best: tuple[tuple[int, int, int], str, Card] | None = None
        for change, new in _changes(card, category, team, names):
            key = evaluate(new, search).key
            if best is None or key > best[0]:
                best = (key, change, new)
        p_correct = float(evaluation.probabilities[category][team])
        if best is not None and best[0] > base_key:
            gain = evaluate(best[2], evaluation).p5 - base_p5
            costs.append(PickCost(category, team, p_correct, best[1], best[2], gain))
        else:
            costs.append(PickCost(category, team, p_correct, None, None, 0.0))
    return sorted(costs, key=lambda c: -c.gain)
