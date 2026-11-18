"""Results sheet for a finished Swiss stage: one row per match.

Columns (lines starting with ``#`` are ignored; an optional ``maps`` column is allowed)::

    round,team1,team2,winner

Rows with team1, team2 and winner all blank are skipped, so a template can carry empty rows for
rounds that have not been played yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from pickem.swiss.replay import MatchResult, check_rounds
from pickem.swiss.rules import N_ROUNDS

COLUMNS = ("round", "team1", "team2", "winner")


def load_results_csv(
    path: str | Path, teams: Sequence[str] | None = None
) -> tuple[tuple[MatchResult, ...], ...]:
    """Read a results sheet into rounds of matches; ``teams`` (if given) must cover every name."""
    path = Path(path)
    try:
        df = pd.read_csv(path, comment="#", skipinitialspace=True, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"{path}: cannot read CSV ({exc})") from exc
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    unknown = [c for c in df.columns if c not in (*COLUMNS, "maps")]
    if missing or unknown:
        raise ValueError(
            f"{path}: columns must be {', '.join(COLUMNS)} (missing {missing}, unknown {unknown})"
        )

    rounds: list[list[MatchResult]] = [[] for _ in range(N_ROUNDS)]
    for line, row in enumerate(df.itertuples(index=False), start=2):
        values = {c: str(getattr(row, c)).strip() for c in df.columns}
        if not (values["team1"] or values["team2"] or values["winner"]):
            continue
        where = f"{path} row {line}"
        try:
            r = int(values["round"])
        except ValueError as exc:
            raise ValueError(f"{where}: round must be 1..{N_ROUNDS}") from exc
        if not 1 <= r <= N_ROUNDS:
            raise ValueError(f"{where}: round must be 1..{N_ROUNDS}")
        if not (values["team1"] and values["team2"]):
            raise ValueError(f"{where}: both teams are needed")
        if not values["winner"]:
            raise ValueError(f"{where}: {values['team1']} v {values['team2']} has no winner yet")
        maps = values.get("maps", "")
        try:
            rounds[r - 1].append(
                MatchResult(
                    values["team1"], values["team2"], values["winner"], int(maps) if maps else None
                )
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc

    result = tuple(tuple(r) for r in rounds)
    try:
        check_rounds(result, teams if teams is not None else _all_teams(result))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return result


def _all_teams(rounds: Sequence[Sequence[MatchResult]]) -> list[str]:
    return sorted({t for matches in rounds for m in matches for t in (m.team1, m.team2)})
