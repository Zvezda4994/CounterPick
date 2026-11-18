"""Playoff Pick'Em: every legal set of picks, scored exactly against all bracket outcomes.

A pick is a winner for each quarterfinal, each semifinal and the final (bracket indices, i.e.
seed - 1). Two rule sets, since the in-game rule has to be confirmed:

- bracket-consistent: each semifinal pick is one of your two quarterfinal picks feeding it, and
  the final pick is one of your two semifinal picks (16 * 4 * 2 = 128 picks);
- independent: each round is picked on its own, a semifinal pick being any team of that half
  and the final pick any team (16 * 16 * 8 = 2,048 picks, a superset of the consistent ones).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pickem.optimizer.compare import resolve_team
from pickem.playoffs.bracket import HALVES, N_TEAMS, QF_SEEDS, outcome_table

Mode = Literal["consistent", "independent"]
MODES: tuple[Mode, ...] = ("consistent", "independent")


@dataclass(frozen=True, order=True)
class PlayoffPick:
    qf: tuple[int, int, int, int]
    sf: tuple[int, int]
    gf: int

    @property
    def consistent(self) -> bool:
        return self.sf[0] in self.qf[:2] and self.sf[1] in self.qf[2:] and self.gf in self.sf

    def describe(self, teams: Sequence[str]) -> dict[str, list[str]]:
        return {
            "QF": [teams[t] for t in self.qf],
            "SF": [teams[t] for t in self.sf],
            "GF": [teams[self.gf]],
        }


def _qf_options() -> list[tuple[int, int]]:
    return [(a - 1, b - 1) for a, b in QF_SEEDS]


@cache
def consistent_picks() -> tuple[PlayoffPick, ...]:
    picks = []
    for a, b, c, d in product(*_qf_options()):
        for top, bottom in product((a, b), (c, d)):
            for gf in (top, bottom):
                picks.append(PlayoffPick((a, b, c, d), (top, bottom), gf))
    return tuple(picks)


@cache
def independent_picks() -> tuple[PlayoffPick, ...]:
    return tuple(
        PlayoffPick((a, b, c, d), (top, bottom), gf)
        for a, b, c, d in product(*_qf_options())
        for top, bottom in product(*HALVES)
        for gf in range(N_TEAMS)
    )


def picks_for(mode: Mode) -> tuple[PlayoffPick, ...]:
    return consistent_picks() if mode == "consistent" else independent_picks()


@dataclass(frozen=True)
class Objective:
    """Pick'Em goal: at least ``min_qf`` QF, ``min_sf`` SF and ``min_gf`` GF picks correct."""

    min_qf: int = 2
    min_sf: int = 1
    min_gf: int = 1

    def __post_init__(self) -> None:
        if not (0 <= self.min_qf <= 4 and 0 <= self.min_sf <= 2 and 0 <= self.min_gf <= 1):
            raise ValueError("thresholds must be within 0-4 (QF), 0-2 (SF) and 0-1 (GF)")

    def describe(self) -> str:
        final = " AND GF correct" if self.min_gf else ""
        return f">={self.min_qf} QF correct AND >={self.min_sf} SF correct{final}"


@dataclass(frozen=True)
class Correct:
    """(picks, 256) correct picks per round in each outcome."""

    qf: NDArray[np.int64]
    sf: NDArray[np.int64]
    gf: NDArray[np.int64]


def correct_counts(picks: Sequence[PlayoffPick]) -> Correct:
    table = outcome_table()
    qf = np.array([p.qf for p in picks])
    sf = np.array([p.sf for p in picks])
    gf = np.array([p.gf for p in picks])
    return Correct(
        qf=(qf[:, None, :] == table.qf_winners[None, :, :]).sum(axis=2),
        sf=(sf[:, None, :] == table.sf_winners[None, :, :]).sum(axis=2),
        gf=(gf[:, None] == table.champion[None, :]).astype(np.int64),
    )


def hits(correct: Correct, objective: Objective) -> NDArray[np.bool_]:
    """(picks, 256): whether each pick meets the objective in each outcome."""
    result: NDArray[np.bool_] = (
        (correct.qf >= objective.min_qf)
        & (correct.sf >= objective.min_sf)
        & (correct.gf >= objective.min_gf)
    )
    return result


@dataclass(frozen=True)
class PickScore:
    pick: PlayoffPick
    p_objective: float
    expected_qf: float
    expected_sf: float
    p_gf: float
    se: float = 0.0
    """Monte Carlo error of ``p_objective`` from the strength draws (0 with fixed probabilities)."""

    @property
    def expected_total(self) -> float:
        return self.expected_qf + self.expected_sf + self.p_gf


def score_picks(
    picks: Sequence[PlayoffPick],
    outcome_probs: NDArray[np.float64],
    objective: Objective,
    per_draw: NDArray[np.float64] | None = None,
) -> list[PickScore]:
    """Exact scores against averaged outcome probabilities (256,). With ``per_draw`` (draws, 256)
    the standard error over draws is reported too."""
    correct = correct_counts(picks)
    met = hits(correct, objective).astype(np.float64)
    p_obj = met @ outcome_probs
    se = np.zeros(len(picks))
    if per_draw is not None and per_draw.shape[0] > 1:
        values = per_draw @ met.T  # (draws, picks)
        se = values.std(axis=0, ddof=1) / np.sqrt(per_draw.shape[0])
    return [
        PickScore(
            pick,
            float(p_obj[i]),
            float(correct.qf[i] @ outcome_probs),
            float(correct.sf[i] @ outcome_probs),
            float(correct.gf[i] @ outcome_probs),
            float(se[i]),
        )
        for i, pick in enumerate(picks)
    ]


def rank(scores: Sequence[PickScore]) -> list[PickScore]:
    """Best first: P(objective), then expected correct picks, then seed order (deterministic)."""
    return sorted(
        scores, key=lambda s: (-round(s.p_objective, 12), -round(s.expected_total, 12), s.pick)
    )


def best_picks(
    mode: Mode,
    search_probs: NDArray[np.float64],
    report_probs: NDArray[np.float64],
    objective: Objective,
    report_per_draw: NDArray[np.float64] | None = None,
    top: int = 5,
) -> list[PickScore]:
    """Brute force: rank every pick on ``search_probs``; report the best on ``report_probs``."""
    picks = picks_for(mode)
    chosen = [s.pick for s in rank(score_picks(picks, search_probs, objective))[:top]]
    return score_picks(chosen, report_probs, objective, report_per_draw)


def favourites_pick(mode: Mode, outcome_probs: NDArray[np.float64]) -> PlayoffPick:
    """Each slot's most likely winner; in consistent mode, only among the teams you advanced."""
    table = outcome_table()
    teams = np.arange(N_TEAMS)

    def odds(winners: NDArray[np.intp]) -> NDArray[np.float64]:
        result: NDArray[np.float64] = outcome_probs @ (winners[:, None] == teams[None, :])
        return result

    def likeliest(options: Sequence[int], p: NDArray[np.float64]) -> int:
        return max(options, key=lambda t: (p[t], -t))

    qf = tuple(
        likeliest((a - 1, b - 1), odds(table.qf_winners[:, k])) for k, (a, b) in enumerate(QF_SEEDS)
    )
    sf_odds = [odds(table.sf_winners[:, j]) for j in range(2)]
    champion = odds(table.champion)
    if mode == "consistent":
        sf = (likeliest(qf[:2], sf_odds[0]), likeliest(qf[2:], sf_odds[1]))
        gf = likeliest(sf, champion)
    else:
        sf = (likeliest(HALVES[0], sf_odds[0]), likeliest(HALVES[1], sf_odds[1]))
        gf = likeliest(range(N_TEAMS), champion)
    return PlayoffPick((qf[0], qf[1], qf[2], qf[3]), (sf[0], sf[1]), gf)


def parse_picks(text: str, teams: Sequence[str]) -> PlayoffPick:
    """``"qf: A, B, C, D; sf: A, C; gf: A"``; each team is placed in its own match or half."""
    parts: dict[str, list[int]] = {}
    for chunk in text.split(";"):
        if not chunk.strip():
            continue
        label, sep, names = chunk.partition(":")
        key = label.strip().casefold()
        if not sep or key not in ("qf", "sf", "gf"):
            raise ValueError(f"expected 'qf: …', 'sf: …' or 'gf: …', got {chunk.strip()!r}")
        if key in parts:
            raise ValueError(f"{key.upper()} is given twice")
        parts[key] = [resolve_team(n, teams) for n in names.split(",") if n.strip()]
    missing = [k.upper() for k in ("qf", "sf", "gf") if k not in parts]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}")

    qf: list[int | None] = [None] * 4
    for t in parts["qf"]:
        k = next(i for i, (a, b) in enumerate(QF_SEEDS) if t in (a - 1, b - 1))
        taken = qf[k]
        if taken is not None:
            raise ValueError(
                f"two QF picks for the same quarterfinal ({teams[taken]} and {teams[t]})"
            )
        qf[k] = t
    if len(parts["qf"]) != 4 or None in qf:
        raise ValueError("pick exactly one winner for each of the 4 quarterfinals")
    sf: list[int | None] = [None, None]
    for t in parts["sf"]:
        half = 0 if t in HALVES[0] else 1
        taken = sf[half]
        if taken is not None:
            raise ValueError(f"two SF picks from the same half ({teams[taken]} and {teams[t]})")
        sf[half] = t
    if len(parts["sf"]) != 2 or None in sf:
        raise ValueError("pick one semifinal winner from each half")
    if len(parts["gf"]) != 1:
        raise ValueError("pick exactly one grand final winner")
    q = [t for t in qf if t is not None]
    s = [t for t in sf if t is not None]
    return PlayoffPick((q[0], q[1], q[2], q[3]), (s[0], s[1]), parts["gf"][0])
