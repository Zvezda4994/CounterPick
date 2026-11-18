"""The odds sheet: one row per team of a 16-team Swiss stage, filled in by hand.

Columns (lines starting with ``#`` are ignored)::

    seed,team,vrs_points,r1_opponent,r1_odds,qualify_odds,not_qualify_odds,win_major_odds

- ``seed``, ``team``, ``vrs_points`` are required for every row.
- ``r1_opponent`` is optional and only checked: round 1 is seed k against seed k + 8.
- All odds are decimal (e.g. 1.85) and may be left blank:
  - ``r1_odds``: this team wins its round-1 BO1. Give both teams of a pair or neither.
  - ``qualify_odds``: this team qualifies from the stage (3 wins). Either every team has it,
    or each team that has it also has ``not_qualify_odds`` (the "No" side).
  - ``win_major_odds``: this team wins the Major.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pickem.probs.margin import Method, overround, remove_margin

N_TEAMS = 16
COLUMNS = (
    "seed",
    "team",
    "vrs_points",
    "r1_opponent",
    "r1_odds",
    "qualify_odds",
    "not_qualify_odds",
    "win_major_odds",
)
REQUIRED = ("seed", "team", "vrs_points")
ODDS_COLUMNS = ("r1_odds", "qualify_odds", "not_qualify_odds", "win_major_odds")
OVERROUND_WARN = (0.0, 0.15)
QUALIFIERS = 8


def r1_opponent(index: int) -> int:
    """Round 1 pairs seed k with seed k + 8 (indices in seed order)."""
    return (index + N_TEAMS // 2) % N_TEAMS


@dataclass(frozen=True)
class OddsSheet:
    """The sheet's contents in seed order (``teams[i]`` is seed ``i + 1``); blank odds are NaN."""

    teams: tuple[str, ...]
    vrs_points: NDArray[np.float64]
    r1_odds: NDArray[np.float64]
    qualify_odds: NDArray[np.float64]
    not_qualify_odds: NDArray[np.float64]
    win_major_odds: NDArray[np.float64]

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(1, N_TEAMS + 1))


@dataclass(frozen=True)
class MarketProbabilities:
    """Fair probabilities per team (NaN where the sheet had no price), in seed order."""

    r1: NDArray[np.float64]
    """P(this team wins its round-1 match)."""
    qualify: NDArray[np.float64]
    """P(this team qualifies from the stage)."""
    win_share: NDArray[np.float64]
    """This team's share of P(win the Major) among the teams in the sheet that have a price."""
    overrounds: dict[str, list[float]] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def n_r1_pairs(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.r1[: N_TEAMS // 2])))

    @property
    def n_qualify(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.qualify)))

    @property
    def n_win(self) -> int:
        return int(np.count_nonzero(~np.isnan(self.win_share)))


def load_odds_csv(path: str | Path) -> OddsSheet:
    """Read and validate an odds sheet; raises ``ValueError`` with a readable message."""
    path = Path(path)
    try:
        df = pd.read_csv(path, comment="#", skipinitialspace=True, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"{path}: cannot read CSV ({exc})") from exc
    df.columns = [c.strip() for c in df.columns]
    unknown = [c for c in df.columns if c not in COLUMNS]
    missing = [c for c in REQUIRED if c not in df.columns]
    if unknown:
        raise ValueError(f"{path}: unknown columns {unknown}; expected {list(COLUMNS)}")
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df.map(lambda v: v.strip() if isinstance(v, str) else v)
    if len(df) != N_TEAMS:
        raise ValueError(f"{path}: expected {N_TEAMS} team rows, got {len(df)}")

    try:
        seeds = [int(s) for s in df["seed"]]
    except ValueError as exc:
        raise ValueError(f"{path}: seed must be a whole number ({exc})") from exc
    if sorted(seeds) != list(range(1, N_TEAMS + 1)):
        raise ValueError(f"{path}: seeds must be 1..{N_TEAMS}, each once")
    df = df.assign(seed=seeds).sort_values("seed").reset_index(drop=True)

    teams = tuple(df["team"])
    if any(not t for t in teams) or len(set(teams)) != N_TEAMS:
        raise ValueError(f"{path}: team names must be non-empty and distinct")
    vrs = _numbers(df["vrs_points"], "vrs_points", path, allow_blank=False)

    for i, given in enumerate(df["r1_opponent"]):
        expected = teams[r1_opponent(i)]
        if given and given != expected:
            raise ValueError(
                f"{path}: {teams[i]} (seed {i + 1}) plays {expected} (seed "
                f"{r1_opponent(i) + 1}) in round 1, not {given}"
            )

    odds = {c: _numbers(df[c], c, path, allow_blank=True) for c in ODDS_COLUMNS}
    for c, values in odds.items():
        bad = [teams[i] for i in np.flatnonzero(values <= 1.0)]
        if bad:
            raise ValueError(f"{path}: {c} must be decimal odds above 1 (check {', '.join(bad)})")

    r1 = odds["r1_odds"]
    for i in range(N_TEAMS // 2):
        j = r1_opponent(i)
        if np.isnan(r1[i]) != np.isnan(r1[j]):
            raise ValueError(
                f"{path}: give r1_odds for both {teams[i]} and {teams[j]}, or for neither"
            )

    yes, no = odds["qualify_odds"], odds["not_qualify_odds"]
    if np.any(~np.isnan(no) & np.isnan(yes)):
        raise ValueError(f"{path}: not_qualify_odds given without qualify_odds")
    has_yes = ~np.isnan(yes)
    if has_yes.any() and not has_yes.all() and np.any(has_yes & np.isnan(no)):
        raise ValueError(
            f"{path}: qualify_odds must be given for all {N_TEAMS} teams, or each team's "
            "qualify_odds must come with its not_qualify_odds"
        )
    if np.any(~np.isnan(no)) and np.any(has_yes & np.isnan(no)):
        raise ValueError(
            f"{path}: if any team has not_qualify_odds, every team with qualify_odds needs it"
        )

    return OddsSheet(
        teams=teams,
        vrs_points=vrs,
        r1_odds=r1,
        qualify_odds=yes,
        not_qualify_odds=no,
        win_major_odds=odds["win_major_odds"],
    )


def _numbers(column: pd.Series, name: str, path: Path, *, allow_blank: bool) -> NDArray[np.float64]:
    out = np.full(len(column), np.nan)
    for i, value in enumerate(column):
        if value == "":
            if not allow_blank:
                raise ValueError(f"{path}: {name} is blank in row {i + 1}")
            continue
        try:
            out[i] = float(value)
        except ValueError as exc:
            raise ValueError(f"{path}: {name} {value!r} is not a number") from exc
    if not np.all(np.isfinite(out[~np.isnan(out)])):
        raise ValueError(f"{path}: {name} must be finite")
    return out


def fair_probabilities(sheet: OddsSheet, method: Method = "power") -> MarketProbabilities:
    """Remove the margin market by market."""
    warnings: list[str] = []
    books: dict[str, list[float]] = {"r1": [], "qualify": [], "win_major": []}

    def check(market: str, label: str, value: float) -> None:
        books[market].append(value)
        low, high = OVERROUND_WARN
        if not low <= value <= high:
            warnings.append(f"{label}: overround {100 * value:+.1f}% looks wrong (check the odds)")

    r1 = np.full(N_TEAMS, np.nan)
    for i in range(N_TEAMS // 2):
        j = r1_opponent(i)
        if np.isnan(sheet.r1_odds[i]):
            continue
        pair = np.array([sheet.r1_odds[i], sheet.r1_odds[j]])
        check("r1", f"round 1 {sheet.teams[i]} v {sheet.teams[j]}", overround(pair))
        r1[i], r1[j] = remove_margin(pair, 1.0, method)

    qualify = np.full(N_TEAMS, np.nan)
    has_yes = ~np.isnan(sheet.qualify_odds)
    has_no = ~np.isnan(sheet.not_qualify_odds)
    if has_no.any():
        for t in (int(k) for k in np.flatnonzero(has_yes)):
            pair = np.array([sheet.qualify_odds[t], sheet.not_qualify_odds[t]])
            check("qualify", f"qualify {sheet.teams[t]}", overround(pair))
            qualify[t] = remove_margin(pair, 1.0, method)[0]
    elif has_yes.all():
        check("qualify", "qualify book", overround(sheet.qualify_odds, QUALIFIERS))
        qualify = remove_margin(sheet.qualify_odds, QUALIFIERS, method)

    win_share = np.full(N_TEAMS, np.nan)
    has_win = ~np.isnan(sheet.win_major_odds)
    if has_win.sum() >= 2:
        # Only part of the Major's field is in the sheet, so use shares within the sheet: a
        # proportional margin cancels, and the fit's temperature absorbs a power-type margin.
        pi = 1.0 / sheet.win_major_odds[has_win]
        win_share[has_win] = pi / pi.sum()

    return MarketProbabilities(
        r1=r1,
        qualify=qualify,
        win_share=win_share,
        overrounds=books,
        warnings=tuple(warnings),
    )
