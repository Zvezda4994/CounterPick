"""End to end on Austin 2025 Stage 1: freeze, score, add the real results, render."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pickem.data.stage_file import StageFile, load_stage, save_stage
from pickem.optimizer.analysis import conditional_outcomes
from pickem.optimizer.card import Card, correct_counts
from pickem.optimizer.cards_file import CardEntry, append_card, load_cards
from pickem.optimizer.leaderboard import Paths, chalk_card, main, optimizer_card
from pickem.probs import FitConfig, fit_strengths, load_odds_csv, stage_from_strengths
from pickem.probs.next_stage import replay_from_results
from tests.probs.test_next_stage import results_csv

FIXTURES = Path(__file__).parents[1] / "fixtures"
SHEET = FIXTURES / "austin_2025_stage1_sheet.csv"
WIKI = FIXTURES / "liquipedia" / "austin_2025_stage_1.wiki"
STAGE = "austin_stage1"
FAST = ["--optimizer-sims", "3000"]

# Austin 2025 Stage 1 actual: 3-0 B8, HEROIC; 3-1 OG, BetBoom, Nemiga; 3-2 Lynn Vision, Legacy,
# TYLOO; 2-3 FlyQuest, NRG, Wildcard; 1-3 Imperial, Chinggis Warriors, Complexity; 0-3 Metizport,
# Fluxo.
PERFECT = (
    "B8",
    "HEROIC",
    "Metizport",
    "Fluxo",
    "OG",
    "BetBoom",
    "Nemiga",
    "Lynn Vision",
    "Legacy",
    "TYLOO",
)
ALL_WRONG = (
    "Fluxo",
    "Metizport",
    "B8",
    "HEROIC",
    "Complexity",
    "FlyQuest",
    "NRG",
    "Wildcard",
    "Imperial",
    "Chinggis Warriors",
)
# Exactly 5 right; Lynn Vision's advance pick hangs on their round-5 win over NRG.
FIVE = (
    "B8",
    "HEROIC",
    "Metizport",
    "Fluxo",
    "Lynn Vision",
    "Complexity",
    "FlyQuest",
    "Wildcard",
    "Imperial",
    "Chinggis Warriors",
)


def card_of(picks: tuple[str, ...], teams: tuple[str, ...]) -> Card:
    i = [teams.index(t) for t in picks]
    return Card(three_oh=tuple(i[:2]), oh_three=tuple(i[2:4]), advance=tuple(i[4:]))


def run(capsys: pytest.CaptureFixture[str], *args: str | Path) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """A frozen Austin Stage 1 with five cards on file, not yet scored."""
    sheet = load_odds_csv(SHEET)
    fit = fit_strengths(sheet, FitConfig(fit_sims=2000))  # VRS-only
    stage = StageFile(
        "Austin 2025 Stage 1",
        stage_from_strengths(fit.teams, fit.seeds, fit.strengths),
        fit.strength_model(spread=1.0, sims_per_draw=500),
    )
    source = tmp_path / "data" / "stage1.json"
    source.parent.mkdir()
    save_stage(source, stage)
    code, _, err = run(
        capsys, "freeze", "--name", STAGE, "--root", tmp_path, "--stage", source, "--odds", SHEET,
        "--add-optimizer", "--add-chalk", *FAST,
    )  # fmt: skip
    assert code == 0, err
    teams = stage.stage.teams
    cards = Paths(tmp_path, STAGE).cards
    for name, picks in (("Perfect", PERFECT), ("All wrong", ALL_WRONG), ("Five", FIVE)):
        append_card(cards, CardEntry(name, "friend", "", card_of(picks, teams)), teams)
    return tmp_path


def score(capsys: pytest.CaptureFixture[str], root: Path, *extra: str) -> tuple[int, str, str]:
    return run(capsys, "score", "--name", STAGE, "--root", root, "--sims", "4000", *FAST, *extra)


def board(root: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(Paths(root, STAGE).board.read_text(encoding="utf-8"))
    return data


def by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in data["cards"]}


def test_freeze(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = Paths(root, STAGE)
    snapshot = load_stage(paths.snapshot)
    frozen = json.loads(paths.snapshot.read_text(encoding="utf-8"))["frozen"]
    assert set(frozen) == {"at", "source", "source_sha256", "odds", "code_commit"}
    assert paths.odds.read_text(encoding="utf-8") == SHEET.read_text(encoding="utf-8")
    entries = {e.name: e for e in load_cards(paths.cards, snapshot.stage.teams)}
    assert entries["Optimizer #1"].card == optimizer_card(snapshot, 3000)
    assert entries["Seed chalk"].card == chalk_card(snapshot)
    teams = snapshot.stage.teams
    assert [teams[t] for t in entries["Seed chalk"].card.three_oh] == ["Complexity", "HEROIC"]
    code, _, err = run(capsys, "freeze", "--name", STAGE, "--root", root, "--stage", paths.snapshot)
    assert code == 2 and "never replaced" in err


def test_score_is_frozen_and_incremental(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = score(capsys, root)
    assert code == 0, err
    first = board(root)
    cards = by_name(first)
    assert set(cards) == {"Optimizer #1", "Seed chalk", "Perfect", "All wrong", "Five"}
    assert cards["Optimizer #1"]["pre"]["label"] == "optimizer pick"
    assert cards["Optimizer #1"]["pre"]["overlap"]["identical"] == 10
    assert cards["Seed chalk"]["pre"]["label"] == "seed chalk"
    optimal_p5 = cards["Optimizer #1"]["pre"]["p5"]
    for card in first["cards"]:
        gaps = sum(d["gain"] for d in card["pre"]["differences"])
        assert gaps == pytest.approx(optimal_p5 - card["pre"]["p5"], abs=1e-12)
        assert card["post"] is None

    teams = load_stage(Paths(root, STAGE).snapshot).stage.teams
    late = card_of(("TYLOO", "Legacy", "Metizport", "Fluxo", *PERFECT[4:8], "B8", "HEROIC"), teams)
    append_card(Paths(root, STAGE).cards, CardEntry("Late", "analyst", "https://x", late), teams)
    assert score(capsys, root)[0] == 0
    second = by_name(board(root))
    for name, card in by_name(first).items():
        assert second[name] == card  # never rewritten
    assert "Late" in second

    assert score(capsys, root, "--sims", "5000")[2].count("keep it the same") == 1


def test_changed_picks_and_edited_snapshot(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert score(capsys, root)[0] == 0
    paths = Paths(root, STAGE)
    text = paths.cards.read_text(encoding="utf-8")
    assert "Five,friend,,HEROIC,B8," in text  # stored in seed order
    paths.cards.write_text(
        text.replace("Five,friend,,HEROIC,B8,", "Five,friend,,HEROIC,OG,"), encoding="utf-8"
    )
    code, _, err = score(capsys, root)
    assert code == 2 and "Five's picks changed" in err
    assert score(capsys, root, "--rescore", "five")[0] == 0
    assert by_name(board(root))["Five"]["picks"]["3-0"] == ["HEROIC", "OG"]

    paths.snapshot.write_text(paths.snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
    code, _, err = score(capsys, root)
    assert code == 2 and "has changed since it was scored" in err


def results_rows(tmp_path: Path) -> Path:
    return results_csv(tmp_path, 1)


def test_results_and_swings(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert score(capsys, root)[0] == 0
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)

    code, _, err = run(
        capsys, "results", "--name", STAGE, "--root", root, "--results", results_rows(tmp_path),
        "--swing-sims", "300",
    )  # fmt: skip
    assert code == 0, err
    cards = by_name(board(root))
    assert cards["Seed chalk"]["post"]["correct"] == 6 and cards["Seed chalk"]["post"]["cleared"]
    assert cards["Perfect"]["post"]["correct"] == 10
    assert cards["All wrong"]["post"]["correct"] == 0 and not cards["All wrong"]["post"]["cleared"]
    five = cards["Five"]["post"]
    assert five["correct"] == 5 and five["cleared"]
    lvg = [s for s in five["swings"] if {s["team1"], s["team2"]} == {"NRG", "Lynn Vision"}]
    assert lvg and lvg[0]["round"] == 5 and lvg[0]["certain"] and lvg[0]["p_clear_if_flipped"] == 0
    for pick in five["picks"]:
        expected = pick["p_correct"] if pick["correct"] else 1 - pick["p_correct"]
        assert pick["p_actual"] == pytest.approx(expected)
    heroic = next(t for t in board(root)["teams"] if t["team"] == "HEROIC")
    assert heroic["actual_record"] == "3-0"
    assert heroic["p_actual_record"] == heroic["records"]["3-0"]

    pages = {"BLAST/Major/2025/Austin/Stage_1": WIKI}
    monkeypatch.setattr(
        "pickem.optimizer.leaderboard.fetch_wikitext",
        lambda page: pages[page].read_text(encoding="utf-8"),
    )
    code, _, err = run(
        capsys, "results", "--name", STAGE, "--root", copy, "--liquipedia",
        "BLAST/Major/2025/Austin/Stage_1", "--swing-sims", "300",
    )  # fmt: skip
    assert code == 0, err
    assert [c["post"] for c in board(copy)["cards"]] == [c["post"] for c in board(root)["cards"]]


def test_replay_without_reversal_reproduces_reality(root: Path, tmp_path: Path) -> None:
    stage = load_stage(Paths(root, STAGE).snapshot).stage
    by_seed = [stage.teams[int(i)] for i in stage.by_seed]
    replay = replay_from_results(by_seed, results_rows(tmp_path))
    outcomes, certain = conditional_outcomes(stage, replay.rounds, None, 50, 0)
    assert certain
    assert int(correct_counts(card_of(FIVE, stage.teams), outcomes)[0]) == 5
    assert int(correct_counts(card_of(PERFECT, stage.teams), outcomes)[0]) == 10


def test_render_and_readme(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert score(capsys, root)[0] == 0
    markdown = Paths(root, STAGE).markdown.read_text(encoding="utf-8")
    assert markdown.startswith(f"## Card leaderboard: {STAGE}")
    dash = chr(0x2013)
    assert "Results are not in yet." in markdown and f"| {dash} | {dash} |" in markdown
    assert "2013" not in markdown.replace("2013-", "")
    assert "### Teams: model vs actual" in markdown and "#### Perfect (friend)" in markdown
    rows = [line for line in markdown.splitlines() if line.startswith("| ") and "/10 |" in line]
    p5 = [float(row.split("|")[6].strip().rstrip("%")) for row in rows]
    assert p5 == sorted(p5, reverse=True)

    readme = root / "README.md"
    readme.write_text("# X\n\nnothing here\n", encoding="utf-8")
    code, _, err = run(capsys, "render", "--name", STAGE, "--root", root, "--readme", readme)
    assert code == 2 and "needs the markers" in err

    readme.write_text(
        f"# X\n\n<!-- leaderboard:{STAGE} -->\nold\n<!-- /leaderboard:{STAGE} -->\n\nend\n",
        encoding="utf-8",
    )
    run(
        capsys,
        "results",
        "--name",
        STAGE,
        "--root",
        root,
        "--results",
        results_rows(tmp_path),
        "--swing-sims",
        "200",
    )
    assert run(capsys, "render", "--name", STAGE, "--root", root, "--readme", readme)[0] == 0
    once = readme.read_text(encoding="utf-8")
    assert run(capsys, "render", "--name", STAGE, "--root", root, "--readme", readme)[0] == 0
    assert readme.read_text(encoding="utf-8") == once
    assert "old" not in once and once.endswith("\n\nend\n")
    assert "Cards that cleared:" in once and f"(leaderboards/{STAGE}.md)" in once
    full = Paths(root, STAGE).markdown.read_text(encoding="utf-8")
    assert "**Swing matches**" in full and "Round 5: NRG v Lynn Vision (Lynn Vision won)" in full
    assert "leaves the best card at that point" in full and "biggest gain first" not in full
