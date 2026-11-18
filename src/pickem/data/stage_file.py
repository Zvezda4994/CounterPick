"""Swiss stage input files.

JSON with these fields::

    {
      "name": "PGL Singapore Major 2026 - Stage 1",  # optional, defaults to the file name
      "teams": ["Team A", "Team B", ...],            # 16 names
      "seeds": [1, 2, ...],                          # optional, defaults to list order
      "p_bo1": [[null, 0.62, ...], ...],             # P(row team beats column team), BO1
      "p_bo3": [[null, 0.68, ...], ...]              # same for BO3
    }

Diagonal entries may be ``null``. Everything else is validated by ``StageInput``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pickem.swiss import StageInput


@dataclass(frozen=True)
class StageFile:
    name: str
    stage: StageInput


def load_stage(path: str | Path) -> StageFile:
    """Read a stage file; raises ``ValueError`` with a readable message if it is invalid."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    missing = [key for key in ("teams", "p_bo1", "p_bo3") if key not in data]
    if missing:
        raise ValueError(f"{path}: missing {', '.join(missing)}")

    teams = data["teams"]
    if not isinstance(teams, list) or not all(isinstance(t, str) for t in teams):
        raise ValueError(f"{path}: teams must be a list of names")
    seeds = data.get("seeds", list(range(1, len(teams) + 1)))
    if not isinstance(seeds, list) or not all(
        isinstance(s, int) and not isinstance(s, bool) for s in seeds
    ):
        raise ValueError(f"{path}: seeds must be a list of integers")
    try:
        stage = StageInput(
            teams=tuple(teams),
            seeds=tuple(seeds),
            p_bo1=_matrix(data["p_bo1"], "p_bo1", len(teams)),
            p_bo3=_matrix(data["p_bo3"], "p_bo3", len(teams)),
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return StageFile(name=str(data.get("name", path.stem)), stage=stage)


def _matrix(value: Any, key: str, n: int) -> NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError(f"{key} must be a {n}x{n} list of rows")
    out = np.empty((n, n), dtype=np.float64)
    for i, row in enumerate(value):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"{key} row {i} must have {n} entries")
        for j, v in enumerate(row):
            if v is None:
                if i != j:
                    raise ValueError(f"{key}[{i}][{j}] is null; only the diagonal may be")
                out[i, j] = 0.5
            elif isinstance(v, int | float) and not isinstance(v, bool):
                out[i, j] = float(v)
            else:
                raise ValueError(f"{key}[{i}][{j}] is not a number")
    return out
