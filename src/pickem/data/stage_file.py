"""Swiss stage input files.

JSON with these fields::

    {
      "name": "PGL Singapore Major 2026 - Stage 1",  # optional, defaults to the file name
      "teams": ["Team A", "Team B", ...],            # 16 names
      "seeds": [1, 2, ...],                          # optional, defaults to list order
      "p_bo1": [[null, 0.62, ...], ...],             # P(row team beats column team), BO1
      "p_bo3": [[null, 0.68, ...], ...],             # same for BO3
      "strength_model": {                            # optional (written by pickem-odds)
        "strengths": [...],                          # per team, same order as "teams"
        "covariance": [[...], ...],
        "spread": 1.0,                               # default multiplier on the SDs (0 = fixed)
        "sims_per_draw": 1000                        # sims that share one draw of strengths
      }
    }

Diagonal entries may be ``null``. Everything else is validated by ``StageInput`` and
``StrengthModel``. Any other top-level keys (such as a fit report) are kept but ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pickem.probs.strength import StrengthModel
from pickem.swiss import StageInput


@dataclass(frozen=True)
class StageFile:
    name: str
    stage: StageInput
    strength_model: StrengthModel | None = None


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
        model = _strength_model(data.get("strength_model"), len(teams))
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return StageFile(name=str(data.get("name", path.stem)), stage=stage, strength_model=model)


def save_stage(
    path: str | Path, stage_file: StageFile, extra: dict[str, Any] | None = None
) -> None:
    """Write a stage file that ``load_stage`` reads back unchanged; ``extra`` keys are added."""
    stage = stage_file.stage

    def rows(p: NDArray[np.float64]) -> list[list[float | None]]:
        return [
            [None if i == j else float(p[i, j]) for j in range(len(stage.teams))]
            for i in range(len(stage.teams))
        ]

    data: dict[str, Any] = {
        "name": stage_file.name,
        "teams": list(stage.teams),
        "seeds": list(stage.seeds),
        "p_bo1": rows(stage.p_bo1),
        "p_bo3": rows(stage.p_bo3),
    }
    model = stage_file.strength_model
    if model is not None:
        data["strength_model"] = {
            "strengths": [float(v) for v in model.strengths],
            "covariance": [[float(v) for v in row] for row in model.covariance],
            "spread": model.spread,
            "sims_per_draw": model.sims_per_draw,
        }
    data.update(extra or {})
    Path(path).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def _strength_model(value: Any, n: int) -> StrengthModel | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not {"strengths", "covariance"} <= set(value):
        raise ValueError("strength_model needs strengths and covariance")
    strengths = value["strengths"]
    if not isinstance(strengths, list) or len(strengths) != n:
        raise ValueError(f"strength_model.strengths must list {n} numbers")
    covariance = value["covariance"]
    if (
        not isinstance(covariance, list)
        or len(covariance) != n
        or any(not isinstance(r, list) or len(r) != n for r in covariance)
    ):
        raise ValueError(f"strength_model.covariance must be {n}x{n}")
    try:
        return StrengthModel(
            strengths=np.array(strengths, dtype=np.float64),
            covariance=np.array(covariance, dtype=np.float64),
            spread=float(value.get("spread", 1.0)),
            sims_per_draw=int(value.get("sims_per_draw", 1000)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strength_model: {exc}") from exc


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
