import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pickem.data.stage_file import StageFile, load_stage, save_stage
from pickem.probs import StrengthModel, stage_from_strengths

EXAMPLE = Path(__file__).parents[2] / "examples" / "example_stage.json"


def write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "stage.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def example_data() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    return data


def test_example_file_loads() -> None:
    stage_file = load_stage(EXAMPLE)
    stage = stage_file.stage
    assert stage_file.name.startswith("Example stage")
    assert stage.teams[0] == "Team 01" and len(stage.teams) == 16
    assert stage.seeds == tuple(range(1, 17))
    assert stage.p_bo1[0, 1] == pytest.approx(0.52)
    assert stage.p_bo1[15, 0] == pytest.approx(0.2)
    assert stage.p_bo3[0, 15] == pytest.approx(0.875)
    assert stage.p_bo1[3, 3] == 0.5  # null diagonal


def test_seeds_and_name_are_optional(tmp_path: Path) -> None:
    data = example_data()
    del data["seeds"], data["name"]
    stage_file = load_stage(write(tmp_path, data))
    assert stage_file.name == "stage"
    assert stage_file.stage.seeds == tuple(range(1, 17))


def test_custom_seed_order(tmp_path: Path) -> None:
    data = example_data()
    data["seeds"] = list(range(16, 0, -1))
    assert load_stage(write(tmp_path, data)).stage.seeds[0] == 16


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("p_bo3"), "missing p_bo3"),
        (lambda d: d.update(teams="Team 01"), "list of names"),
        (lambda d: d.update(teams=d["teams"][:15]), "15x15"),
        (lambda d: d.update(seeds=["1"] * 16), "list of integers"),
        (lambda d: d.update(seeds=[1] * 16), "permutation"),
        (lambda d: d["p_bo1"][0].__setitem__(1, None), "only the diagonal"),
        (lambda d: d["p_bo1"][0].__setitem__(1, "high"), "not a number"),
        (lambda d: d["p_bo1"][0].__setitem__(1, 0.9), "must equal 1"),
        (lambda d: d["p_bo3"].pop(), "16x16"),
        (lambda d: d["p_bo3"][2].pop(), "row 2"),
    ],
)
def test_invalid_files_fail_clearly(tmp_path: Path, change: Any, message: str) -> None:
    data = example_data()
    change(data)
    with pytest.raises(ValueError, match=message):
        load_stage(write(tmp_path, data))


def test_not_json_or_not_an_object(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_stage(bad)
    with pytest.raises(ValueError, match="JSON object"):
        load_stage(write(tmp_path, [1, 2, 3]))


def test_files_without_a_strength_model_still_load() -> None:
    assert load_stage(EXAMPLE).strength_model is None


def test_save_and_load_round_trip_with_strength_model(tmp_path: Path) -> None:
    teams = tuple(f"T{i:02d}" for i in range(16))
    strengths = np.linspace(0.5, -0.5, 16)
    covariance = np.full((16, 16), 0.002) + np.eye(16) * 0.02
    original = StageFile(
        name="round trip",
        stage=stage_from_strengths(teams, tuple(range(16, 0, -1)), strengths),
        strength_model=StrengthModel(strengths, covariance, spread=0.7, sims_per_draw=250),
    )
    path = tmp_path / "stage.json"
    save_stage(path, original, extra={"fit": {"note": "kept but ignored"}})
    loaded = load_stage(path)
    assert loaded.name == "round trip"
    assert loaded.stage.teams == teams and loaded.stage.seeds == tuple(range(16, 0, -1))
    assert (loaded.stage.p_bo1 == original.stage.p_bo1).all()
    assert (loaded.stage.p_bo3 == original.stage.p_bo3).all()
    model = loaded.strength_model
    assert model is not None
    assert (model.strengths == strengths).all() and (model.covariance == covariance).all()
    assert (model.spread, model.sims_per_draw) == (0.7, 250)
    assert json.loads(path.read_text(encoding="utf-8"))["fit"] == {"note": "kept but ignored"}


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"strengths": [0.0] * 16}, "needs strengths and covariance"),
        ({"strengths": [0.0] * 15, "covariance": [[0.0] * 16] * 16}, "16 numbers"),
        ({"strengths": [0.0] * 16, "covariance": [[0.0] * 16] * 15}, "16x16"),
        ({"strengths": [0.0] * 16, "covariance": [[0.0] * 16] * 16, "spread": -1}, "spread"),
    ],
)
def test_invalid_strength_model(tmp_path: Path, model: dict[str, Any], message: str) -> None:
    data = example_data()
    data["strength_model"] = model
    with pytest.raises(ValueError, match=message):
        load_stage(write(tmp_path, data))
