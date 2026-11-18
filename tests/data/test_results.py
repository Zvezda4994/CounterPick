from pathlib import Path

import pytest

from pickem.data.results import load_results_csv
from pickem.probs.odds_csv import load_odds_csv

EXAMPLES = Path(__file__).parents[2] / "examples"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_singapore_template_lists_round1_and_waits_for_winners() -> None:
    template = EXAMPLES / "singapore_2026_stage1_results.csv"
    teams = load_odds_csv(EXAMPLES / "singapore_2026_stage1_odds.csv").teams
    lines = [line for line in template.read_text(encoding="utf-8").splitlines() if line[:2] == "1,"]
    assert [tuple(line.split(",")[1:3]) for line in lines] == [
        (teams[i], teams[i + 8]) for i in range(8)
    ]
    with pytest.raises(ValueError, match="Ninjas in Pyjamas v M80 has no winner yet"):
        load_results_csv(template, teams)


def test_blank_rows_and_maps_column(tmp_path: Path) -> None:
    rows = ["round,team1,team2,winner,maps", "1,A,B,A,1", "2,,,,"]
    with pytest.raises(ValueError, match="round 1 needs 8 results, got 1"):
        load_results_csv(write(tmp_path, "\n".join(rows)))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("round,team1,team2\n1,A,B\n", "columns must be"),
        ("round,team1,team2,winner\n6,A,B,A\n", "round must be 1..5"),
        ("round,team1,team2,winner\nx,A,B,A\n", "round must be 1..5"),
        ("round,team1,team2,winner\n1,A,,A\n", "both teams"),
        ("round,team1,team2,winner\n1,A,B,C\n", "did not play"),
        ("round,team1,team2,winner\n1,A,A,A\n", "cannot play itself"),
        ("round,team1,team2,winner\n1,A,B,A\n1,A,C,C\n", "plays twice"),
    ],
)
def test_invalid_results(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_results_csv(write(tmp_path, body))


def test_unknown_team_against_the_sheet(tmp_path: Path) -> None:
    teams = [f"T{i}" for i in range(16)]
    with pytest.raises(ValueError, match="unknown team 'X'"):
        load_results_csv(write(tmp_path, "round,team1,team2,winner\n1,T0,X,T0\n"), teams)
