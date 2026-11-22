from pathlib import Path

import pytest

from pickem.data.stage_file import StageFile
from pickem.fantasy.cli import main
from tests.fantasy.conftest import EXAMPLE_STAGE, write_players


def test_cli_reports_both_lineups_and_scores_yours(
    tmp_path: Path, stage_file: StageFile, capsys: pytest.CaptureFixture[str]
) -> None:
    players = write_players(tmp_path / "players.csv", stage_file.stage.teams)
    args = [str(EXAMPLE_STAGE), "--players", str(players), "--sims", "400", "--field", "300"]
    code = main([*args, "--restarts", "1", "--lineup", "P004, P014, P024, P034, P044"])
    out = capsys.readouterr().out
    assert code == 0
    for section in ("Max expected points", "High upside (objective: season;", "Your lineup"):
        assert section in out
    assert "E[season points]" in out and "Round 1 boosters" in out
    assert "Warning" not in out


def test_cli_rejects_illegal_lineups(
    tmp_path: Path, stage_file: StageFile, capsys: pytest.CaptureFixture[str]
) -> None:
    players = write_players(tmp_path / "players.csv", stage_file.stage.teams)
    args = [str(EXAMPLE_STAGE), "--players", str(players), "--sims", "50"]
    assert main([*args, "--lineup", "P000, P001, P002, P033, P044"]) == 2
    assert "more than 2 players from Team 01" in capsys.readouterr().err
    assert main([*args, "--objective", "top3"]) == 2


def _round1_results(path: Path, teams: tuple[str, ...]) -> Path:
    rows = ["round,team1,team2,winner"]
    rows += [f"1,{teams[k]},{teams[k + 8]},{teams[k]}" for k in range(8)]
    rows += ["2,,,"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_cli_plans_a_round_from_results(
    tmp_path: Path, stage_file: StageFile, capsys: pytest.CaptureFixture[str]
) -> None:
    teams = stage_file.stage.teams
    players = write_players(tmp_path / "players.csv", teams)
    results = _round1_results(tmp_path / "results.csv", teams)
    args = [str(EXAMPLE_STAGE), "--players", str(players), "--sims", "300"]
    lineup = ["--lineup", "P004, P014, P024, P034, P044"]
    code = main(
        [
            *args,
            *lineup,
            "--round",
            "2",
            "--results",
            str(results),
            "--used-boosters",
            "Hellcase, carry",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "round 2 plan" in out and "8 results fixed" in out and "2 booster(s) already used" in out
    assert "Hellcase" not in out.split("Booster")[1]
    assert main([*args, *lineup, "--round", "3", "--results", str(results)]) == 2
    assert "round 2 needs 8 results" in capsys.readouterr().err
