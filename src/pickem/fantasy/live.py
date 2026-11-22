"""Re-planning roles and boosters once the stage is under way.

Results so far are read from a results sheet (``round,team1,team2,winner``, as for
pickem-next-stage, but only the rounds played so far need to be filled in). The stage is then
simulated with those results fixed, and for the next round:
- roles: the assignment (distinct roles) with the most expected role points that round;
- boosters: the remaining boosters are assigned to (player, round) slots from this round on to
  maximize expected booster points, and only this round's part of the plan is acted on; later
  rounds are re-planned when they come.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from pickem.fantasy.performance import PlayerSims
from pickem.fantasy.rules import BOOSTER_NAMES, N_BOOSTERS, N_ROUNDS
from pickem.swiss.replay import MATCHES_PER_ROUND, MatchResult


def played_matches(path: str | Path, teams: Sequence[str], before_round: int) -> list[MatchResult]:
    """Results of rounds 1 .. ``before_round - 1``; each of those rounds must be complete."""
    path = Path(path)
    try:
        df = pd.read_csv(path, comment="#", skipinitialspace=True, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"{path}: cannot read CSV ({exc})") from exc
    df.columns = [c.strip() for c in df.columns]
    missing = {"round", "team1", "team2", "winner"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")
    by_round: dict[int, list[MatchResult]] = {r: [] for r in range(1, before_round)}
    for line, row in enumerate(df.itertuples(index=False), start=2):
        values = {c: str(getattr(row, c)).strip() for c in ("round", "team1", "team2", "winner")}
        if not values["winner"]:
            continue
        try:
            r = int(values["round"])
        except ValueError as exc:
            raise ValueError(f"{path} row {line}: round must be 1..{N_ROUNDS}") from exc
        if r >= before_round:
            continue
        for team in (values["team1"], values["team2"]):
            if team not in teams:
                raise ValueError(f"{path} row {line}: unknown team {team!r}")
        try:
            by_round[r].append(MatchResult(values["team1"], values["team2"], values["winner"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path} row {line}: {exc}") from exc
    for r, matches in by_round.items():
        if len(matches) != MATCHES_PER_ROUND[r - 1]:
            raise ValueError(
                f"{path}: round {r} needs {MATCHES_PER_ROUND[r - 1]} results, got {len(matches)}"
            )
    return [m for r in sorted(by_round) for m in by_round[r]]


def booster_indices(names: str) -> set[int]:
    """Comma-separated booster names (case-insensitive) -> indices."""
    lookup = {n.casefold(): i for i, n in enumerate(BOOSTER_NAMES)}
    out = set()
    for raw in names.split(","):
        name = raw.strip().casefold()
        if not name:
            continue
        if name not in lookup:
            raise ValueError(
                f"unknown booster {raw.strip()!r}; boosters: {', '.join(BOOSTER_NAMES)}"
            )
        out.add(lookup[name])
    return out


@dataclass(frozen=True)
class RoundPlan:
    round_index: int
    roles: tuple[int, ...]
    """Role index per lineup position."""
    role_expected: tuple[float, ...]
    boosters_now: tuple[tuple[int, int, float], ...]
    """(booster, lineup position, P(success)) to set for this round."""
    boosters_later: tuple[tuple[int, int, int, float], ...]
    """(booster, position, round index, P(success)): the tentative rest of the plan."""


def plan_round(
    sims: PlayerSims, players: Sequence[int], round_index: int, used: set[int]
) -> RoundPlan:
    if sims.role_rounds is None:
        raise ValueError("simulate_players(..., keep_rounds=True) is needed to plan a round")
    if not 0 <= round_index < N_ROUNDS:
        raise ValueError(f"round must be 1..{N_ROUNDS}")
    idx = list(players)
    role_values = sims.role_rounds[:, idx, round_index, :].mean(axis=0)
    rows, cols = linear_sum_assignment(-role_values)
    roles = [0] * len(idx)
    expected = [0.0] * len(idx)
    for r, c in zip(rows, cols, strict=True):
        roles[int(r)] = int(c)
        expected[int(r)] = float(role_values[r, c])

    available = [b for b in range(N_BOOSTERS) if b not in used]
    later_rounds = N_ROUNDS - round_index
    probs = sims.booster_probabilities()[idx][:, round_index:, :][:, :, available]
    values = probs.reshape(len(idx) * later_rounds, len(available)).T
    now: list[tuple[int, int, float]] = []
    later: list[tuple[int, int, int, float]] = []
    if available:
        b_rows, b_cols = linear_sum_assignment(-values)
        for b, slot in zip(b_rows, b_cols, strict=True):
            pos, offset = divmod(int(slot), later_rounds)
            booster, p = available[int(b)], float(values[b, slot])
            if offset == 0:
                now.append((booster, pos, p))
            else:
                later.append((booster, pos, round_index + offset, p))
    return RoundPlan(
        round_index,
        tuple(roles),
        tuple(expected),
        tuple(sorted(now, key=lambda x: x[1])),
        tuple(sorted(later, key=lambda x: (x[2], x[1]))),
    )


def teams_alive(sims: PlayerSims, players: Sequence[int], round_index: int) -> np.ndarray:
    """P(each lineup player plays ``round_index``)."""
    result: np.ndarray = sims.played[:, list(players), round_index].mean(axis=0)
    return result
