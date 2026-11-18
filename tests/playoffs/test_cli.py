from pathlib import Path

import numpy as np
import pytest

from pickem.data.stage_file import StageFile, save_stage
from pickem.playoffs.cli import main
from pickem.probs import StrengthModel, stage_from_strengths
from tests.probs.test_next_stage import NAME, STAGES, results_csv

TEAMS = tuple(f"Team {i:02d}" for i in range(1, 17))
BRACKET = ",".join(TEAMS[:8])
FAST = ["--draws", "300"]


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    strengths = np.linspace(0.8, -0.8, 16)
    path = tmp_path / "stage3.json"
    save_stage(
        path,
        StageFile(
            "test",
            stage_from_strengths(TEAMS, tuple(range(1, 17)), strengths),
            StrengthModel(strengths, np.eye(16) * 0.04, spread=1.0),
        ),
    )
    return path


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_report_with_what_if_bracket(stage: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, "--stage", stage, "--bracket", BRACKET, *FAST)
    assert code == 0, err
    assert "Half A: (1) Team 01 v (8) Team 08   |   (4) Team 04 v (5) Team 05" in out
    assert "Half B: (2) Team 02 v (7) Team 07   |   (3) Team 03 v (6) Team 06" in out
    assert "grand final BO5" in out and "P(3rd)" in out
    assert "Best bracket-consistent picks (top 5 of 128)" in out
    assert "Best independent per-round picks (top 5 of 2,048)" in out
    assert out.count("Favourites (") == 2
    assert "Independent rules would add" in out
    assert "reported on 300 independent ones" in out


def test_rulebook_format_and_fixed_probabilities(
    stage: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = run(
        capsys, "--stage", stage, "--bracket", BRACKET, "--format", "rulebook", "--spread", "0",
        "--mode", "consistent",
    )  # fmt: skip
    assert code == 0
    assert "every match BO3, no third-place match" in out and "P(SF loss)" in out
    assert "fixed probabilities" in out and "+/-" not in out
    assert "independent per-round" not in out


def test_scoring_your_picks(stage: Path, capsys: pytest.CaptureFixture[str]) -> None:
    picks = "qf: Team 01, Team 04, Team 02, Team 03; sf: Team 01, Team 02; gf: Team 01"
    code, out, _ = run(capsys, "--stage", stage, "--bracket", BRACKET, *FAST, "--picks", picks)
    assert code == 0
    assert "Your picks (valid under both rules)" in out
    assert "consistent picks" in out and "independent picks" in out
    loose = "qf: Team 01, Team 04, Team 02, Team 03; sf: Team 05, Team 06; gf: Team 08"
    code, out, _ = run(capsys, "--stage", stage, "--bracket", BRACKET, *FAST, "--picks", loose)
    assert "valid only with independent picks" in out
    assert "consistent picks" not in out.split("Your picks")[1]


def test_seeding_from_austin_stage3_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    teams = tuple(NAME[a] for a in STAGES[3]["seeds"])
    path = tmp_path / "austin3.json"
    save_stage(
        path, StageFile("Austin 3", stage_from_strengths(teams, tuple(range(1, 17)), np.zeros(16)))
    )
    code, out, err = run(
        capsys, "--stage", path, "--results", results_csv(tmp_path, 3), "--mode", "consistent"
    )
    assert code == 0, err
    assert "Half A: (1) Spirit v (8) MOUZ   |   (4) Natus Vincere v (5) Vitality" in out
    assert "Half B: (2) FURIA v (7) paiN   |   (3) FaZe v (6) The MongolZ" in out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--bracket", ",".join(TEAMS[:7])], "8 different teams"),
        (["--bracket", "Team 01,Vitality"], "unknown team"),
        (["--bracket", BRACKET, "--picks", "qf: Team 01"], "missing"),
        (["--bracket", BRACKET, "--min-sf", "3"], "thresholds"),
        (["--bracket", BRACKET, "--draws", "1"], "--draws"),
    ],
)
def test_errors(
    stage: Path, capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    code, _, err = run(capsys, "--stage", stage, *args)
    assert code == 2 and message in err
