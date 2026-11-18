from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pickem.data.stage_file import load_stage
from pickem.optimizer.cli import main as optimize_main
from pickem.probs.cli import main as odds_main

EXAMPLES = Path(__file__).parents[2] / "examples"
FAST_FIT = ["--fit-sims", "4000"]
FAST_OPT = ["--sims", "3000", "--eval-sims", "2000", "--restarts", "2"]


def run(
    capsys: pytest.CaptureFixture[str],
    fn: Callable[[Sequence[str] | None], int],
    *args: str | Path,
) -> tuple[int, str, str]:
    code = fn([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def example_stage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    out = tmp_path / "stage.json"
    code, _, err = run(
        capsys,
        odds_main,
        EXAMPLES / "example_odds.csv",
        "--out",
        out,
        "--sims-per-draw",
        "500",
        *FAST_FIT,
    )
    assert code == 0, err
    return out


def test_odds_to_stage_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "stage.json"
    code, report, err = run(
        capsys, odds_main, EXAMPLES / "example_odds.csv", "--out", out, *FAST_FIT
    )
    assert code == 0, err
    assert "Markets used: round 1 8/8 pairs, qualify 16/16 teams, win Major 16/16 teams" in report
    assert "VRS slope" in report and "win-Major temperature" in report
    assert report.count("Team ") >= 16
    stage_file = load_stage(out)
    assert stage_file.name == "example_odds"
    assert stage_file.stage.teams[0] == "Team 01"
    assert stage_file.strength_model is not None
    assert stage_file.strength_model.spread == 1.0
    assert '"fit"' in out.read_text(encoding="utf-8")


def test_vrs_only_template(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "stage1.json"
    code, report, _ = run(
        capsys,
        odds_main,
        EXAMPLES / "singapore_2026_stage1_odds.csv",
        "--out",
        out,
        "--name",
        "Singapore Stage 1 (VRS only)",
    )
    assert code == 0
    assert "No odds given" in report
    assert load_stage(out).name == "Singapore Stage 1 (VRS only)"


def test_odds_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("seed,team\n1,A\n", encoding="utf-8")
    code, _, err = run(capsys, odds_main, bad, "--out", tmp_path / "x.json")
    assert code == 2 and "missing columns" in err
    code, _, err = run(
        capsys,
        odds_main,
        EXAMPLES / "example_odds.csv",
        "--out",
        tmp_path / "x.json",
        "--prior-sd",
        "0",
    )
    assert code == 2 and "prior_sd" in err


def test_optimize_with_uncertainty_and_comparison(
    example_stage: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = run(capsys, optimize_main, example_stage, *FAST_OPT, "--compare-spread")
    assert code == 0, err
    assert "Strength uncertainty: spread 1.00, new team strengths every 500 sims" in out
    assert "does it change the best card?" in out
    assert "best card @ spread 0" in out and "best card @ spread 1.00" in out
    assert ("Same best card" in out) or ("The best cards differ" in out)


def test_optimize_spread_overrides(example_stage: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, optimize_main, example_stage, *FAST_OPT, "--spread", "0")
    assert code == 0 and "Strength uncertainty: none (fixed probabilities)" in out
    code, out, _ = run(
        capsys, optimize_main, example_stage, *FAST_OPT, "--spread", "2", "--sims-per-draw", "100"
    )
    assert code == 0 and "spread 2.00, new team strengths every 100 sims" in out


def test_optimize_rejects_uncertainty_without_a_model(capsys: pytest.CaptureFixture[str]) -> None:
    fixed = EXAMPLES / "example_stage.json"
    code, _, err = run(capsys, optimize_main, fixed, *FAST_OPT, "--compare-spread")
    assert code == 2 and "--compare-spread" in err
    code, _, err = run(capsys, optimize_main, fixed, *FAST_OPT, "--spread", "1")
    assert code == 2 and "no strength_model" in err
    code, _, err = run(
        capsys, optimize_main, fixed, *FAST_OPT, "--compare-spread", "--eval-sims", "0"
    )
    assert code == 2
