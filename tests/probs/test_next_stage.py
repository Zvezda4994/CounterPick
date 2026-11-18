"""Austin 2025: building Stage 2 (and 3) from the previous stage's results must give the real seeds
and round-1 pairings."""

import json
from pathlib import Path
from typing import Any

import pytest

from pickem.probs.cli import main as odds_main
from pickem.probs.next_stage import main, parse_invited, read_invited_file
from pickem.probs.odds_csv import OddsSheet, load_odds_csv

FIXTURES = Path(__file__).parents[1] / "fixtures"
EXAMPLES = Path(__file__).parents[2] / "examples"
STAGES: dict[int, dict[str, Any]] = {
    s["stage"]: s
    for s in json.loads((FIXTURES / "austin_2025.json").read_text(encoding="utf-8"))["stages"]
}
SHEET1 = FIXTURES / "austin_2025_stage1_sheet.csv"
VRS = FIXTURES / "vrs_global_2025_05_05_excerpt.md"
WIKI = {s: FIXTURES / "liquipedia" / f"austin_2025_stage_{s}.wiki" for s in (1, 2)}

NAME = {
    "col": "Complexity", "heroic": "HEROIC", "b8": "B8", "betboom": "BetBoom", "tyloo": "TYLOO",
    "lvg": "Lynn Vision", "wildcard": "Wildcard", "fq": "FlyQuest", "og": "OG",
    "chinggis warriors": "Chinggis Warriors", "imp": "Imperial", "nemiga": "Nemiga", "nrg": "NRG",
    "legacy": "Legacy", "metizport": "Metizport", "fluxo": "Fluxo", "falcons": "Falcons",
    "faze": "FaZe", "3dmax": "3DMAX", "vp": "Virtus.pro", "pain": "paiN", "furia": "FURIA",
    "mibr": "MIBR", "m80": "M80", "vit": "Vitality", "mouz": "MOUZ", "spirit": "Spirit", "g2": "G2",
    "mongolz": "The MongolZ", "aurora": "Aurora", "navi": "Natus Vincere", "tl": "Liquid",
}  # fmt: skip
INVITED = {
    2: ["Falcons", "FaZe", "3DMAX", "Virtus.pro", "paiN", "FURIA", "MIBR", "M80"],
    3: ["Vitality", "MOUZ", "Spirit", "G2", "The MongolZ", "Aurora", "Natus Vincere", "Liquid"],
}


def results_csv(tmp_path: Path, stage: int, edit: Any = None) -> Path:
    rows = [
        [r, NAME[m["teams"][0]], NAME[m["teams"][1]], NAME[m["winner"]]]
        for r, matches in enumerate(STAGES[stage]["rounds"], start=1)
        for m in matches
    ]
    if edit is not None:
        rows = edit(rows)
    path = tmp_path / f"stage{stage}_results.csv"
    lines = ["round,team1,team2,winner"] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def real_seeds(stage: int) -> list[str]:
    return [NAME[alias] for alias in STAGES[stage]["seeds"]]


def real_round1(stage: int) -> set[frozenset[str]]:
    return {frozenset(NAME[t] for t in m["teams"]) for m in STAGES[stage]["rounds"][0]}


def sheet_round1(sheet: OddsSheet) -> set[frozenset[str]]:
    return {frozenset((sheet.teams[i], sheet.teams[i + 8])) for i in range(8)}


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def build(
    capsys: pytest.CaptureFixture[str],
    out: Path,
    stage: int,
    previous: Path,
    *source: str | Path,
    invited: list[str] | None = None,
) -> tuple[Path, str]:
    code, report, err = run(
        capsys,
        "--previous",
        previous,
        *source,
        "--vrs",
        VRS,
        "--invited",
        ",".join(INVITED[stage] if invited is None else invited),
        "--out",
        out,
    )
    assert code == 0, err
    return out, report


def data_lines(path: Path) -> list[str]:
    return [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]


@pytest.fixture
def from_wiki(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = {"BLAST/Major/2025/Austin/Stage_1": WIKI[1], "BLAST/Major/2025/Austin/Stage_2": WIKI[2]}
    monkeypatch.setattr(
        "pickem.probs.next_stage.fetch_wikitext",
        lambda page: pages[page].read_text(encoding="utf-8"),
    )


def test_stage1_results_reproduce_the_real_stage2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, report = build(
        capsys, tmp_path / "s2.csv", 2, SHEET1, "--results", results_csv(tmp_path, 1)
    )
    sheet = load_odds_csv(out)
    assert list(sheet.teams) == real_seeds(2)
    assert sheet_round1(sheet) == real_round1(2)
    assert sheet.vrs_points[0] == 1927  # Falcons, from the VRS file
    assert sheet.vrs_points[8] == 1227  # B8, carried over from the Stage 1 sheet
    assert "Round 1: Falcons v B8; FaZe v HEROIC" in report
    assert " 1  B8" in report and "3-0  Difficulty -1  advances" in report
    assert "invited, VRS standing 3" in report


def test_liquipedia_wikitext_gives_the_same_sheet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], from_wiki: None
) -> None:
    typed, _ = build(capsys, tmp_path / "a.csv", 2, SHEET1, "--results", results_csv(tmp_path, 1))
    read, _ = build(
        capsys, tmp_path / "b.csv", 2, SHEET1, "--liquipedia", "BLAST/Major/2025/Austin/Stage_1"
    )
    assert data_lines(typed) == data_lines(read)


def test_chained_stage2_to_stage3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], from_wiki: None
) -> None:
    stage2, _ = build(
        capsys, tmp_path / "s2.csv", 2, SHEET1, "--liquipedia", "BLAST/Major/2025/Austin/Stage_1"
    )
    stage3, _ = build(
        capsys, tmp_path / "s3.csv", 3, stage2, "--liquipedia", "BLAST/Major/2025/Austin/Stage_2"
    )
    sheet = load_odds_csv(stage3)
    assert list(sheet.teams) == real_seeds(3)
    assert sheet_round1(sheet) == real_round1(3)


def test_output_runs_through_pickem_odds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet, _ = build(capsys, tmp_path / "s2.csv", 2, SHEET1, "--results", results_csv(tmp_path, 1))
    assert odds_main([str(sheet), "--out", str(tmp_path / "s2.json"), "--fit-sims", "2000"]) == 0
    text = sheet.read_text(encoding="utf-8")
    assert text.startswith("# Odds sheet built by pickem-next-stage")
    assert "standings as of 2025-05-05" in text


def flip_first_winner(rows: list[list[Any]]) -> list[list[Any]]:
    first = rows[0]
    first[3] = first[2] if first[3] == first[1] else first[1]
    return rows


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (flip_first_winner, "round 2 does not follow the Swiss rules"),
        (lambda rows: rows[:-1], "round 5 needs 3 results"),
        (lambda rows: [[1, "Nobody", *rows[0][2:]], *rows[1:]], "Nobody"),
    ],
)
def test_bad_results_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], edit: Any, message: str
) -> None:
    code, _, err = run(
        capsys,
        "--previous", SHEET1,
        "--results", results_csv(tmp_path, 1, edit),
        "--vrs", VRS,
        "--invited", ",".join(INVITED[2]),
        "--out", tmp_path / "s2.csv",
    )  # fmt: skip
    assert code == 2 and message in err


@pytest.mark.parametrize(
    ("invited", "message"),
    [
        (INVITED[2][:7], "need 8 invited teams"),
        ([*INVITED[2][:7], "Astralis"], "not in the VRS table"),
        ([*INVITED[2][:7], "B8"], "advanced from the previous stage"),
        ([*INVITED[2][:7], "FaZe"], "listed twice"),
        ([*INVITED[2][:7], "Mirror=FaZe"], "same VRS entry"),
    ],
)
def test_bad_invited_lists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], invited: list[str], message: str
) -> None:
    code, _, err = run(
        capsys,
        "--previous", SHEET1,
        "--results", results_csv(tmp_path, 1),
        "--vrs", VRS,
        "--invited", ",".join(invited),
        "--out", tmp_path / "s2.csv",
    )  # fmt: skip
    assert code == 2 and message in err


def test_renamed_invited_team(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    invited = [*INVITED[2][:3], "VP=Virtus.pro", *INVITED[2][4:]]
    out, _ = build(
        capsys, tmp_path / "s2.csv", 2, SHEET1, "--results", results_csv(tmp_path, 1),
        invited=invited,
    )  # fmt: skip
    assert load_odds_csv(out).teams[3] == "VP"


def test_never_overwrites_without_force(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "s2.csv"
    out.write_text("my filled-in odds\n", encoding="utf-8")
    args: list[str | Path] = [
        "--previous", SHEET1,
        "--results", results_csv(tmp_path, 1),
        "--vrs", VRS,
        "--invited", ",".join(INVITED[2]),
        "--out", out,
    ]  # fmt: skip
    code, _, err = run(capsys, *args)
    assert code == 2 and "already exists" in err
    assert out.read_text(encoding="utf-8") == "my filled-in odds\n"
    code, _, _ = run(capsys, *args, "--force")
    assert code == 0 and load_odds_csv(out).teams[0] == "Falcons"


def test_invited_files_and_parsing() -> None:
    assert read_invited_file(EXAMPLES / "singapore_2026_stage2_invited.txt") == [
        "FUT", "Nemiga", "G2", "1win", "Natus Vincere", "Astralis", "The MongolZ", "MIBR",
    ]  # fmt: skip
    assert len(read_invited_file(EXAMPLES / "singapore_2026_stage3_invited.txt")) == 8
    assert parse_invited(["GOSH=BC.Game", " FaZe "]) == [("GOSH", "BC.Game"), ("FaZe", "FaZe")]
    with pytest.raises(ValueError, match="empty"):
        parse_invited(["=BC.Game"])
