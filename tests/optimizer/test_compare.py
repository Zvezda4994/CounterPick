from pathlib import Path

import numpy as np
import pytest

from pickem.optimizer import Card, Outcomes, evaluate, optimize
from pickem.optimizer.card import OH_THREE, THREE_OH
from pickem.optimizer.cli import main
from pickem.optimizer.compare import compare_cards, parse_card, pick_costs, resolve_team
from pickem.swiss import StageInput, simulate_swiss

NAMES = (
    "Ninjas in Pyjamas", "PARIVISION", "HEROIC", "FaZe", "Liquid", "Eternal Fire", "100 Thieves",
    "9z", "M80", "DENDELE", "Imperial", "Wildcard", "GOSH", "FlyQuest", "Chinggis Warriors",
    "Lynn Vision",
)  # fmt: skip
EXAMPLE = Path(__file__).parents[2] / "examples" / "example_stage.json"


def test_resolve_team_names() -> None:
    assert resolve_team("liquid", NAMES) == 4
    assert resolve_team("NiP", NAMES) == 0  # initials
    assert resolve_team("lynn", NAMES) == 15  # unique prefix
    assert resolve_team(" 100 thieves ", NAMES) == 6
    assert resolve_team("100T", NAMES) == 6  # prefix ignoring spaces
    assert resolve_team("ChinggisW", NAMES) == 14
    with pytest.raises(ValueError, match="could be"):
        resolve_team("F", NAMES)  # FaZe or FlyQuest
    with pytest.raises(ValueError, match="unknown team 'Vitality'"):
        resolve_team("Vitality", NAMES)


def test_parse_card() -> None:
    card = parse_card(
        "advance: 100 Thieves, PARIVISION, FaZe, HEROIC, M80, liquid; "
        "3-0: 9z, NiP; 0-3: Lynn Vision, Chinggis Warriors",
        NAMES,
    )
    assert card == Card(three_oh=(7, 0), oh_three=(15, 14), advance=(6, 1, 3, 2, 8, 4))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("3-0: NiP, 9z; 0-3: Lynn Vision, Chinggis Warriors", "missing Advance"),
        ("3-0: NiP; 3-0: 9z", "given twice"),
        ("3-1: NiP", "unknown category"),
        ("NiP, 9z", "expected 'category: team, team'"),
        ("3-0: NiP; 0-3: Lynn Vision, Chinggis Warriors; advance: FaZe", "3-0 needs 2"),
        ("3-0: NiP, 9z; 0-3: NiP, GOSH; advance: FaZe, M80, Liquid, HEROIC, 100 Thieves, DENDELE",
         "only one category"),
    ],
)  # fmt: skip
def test_parse_card_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_card(text, NAMES)


@pytest.fixture(scope="module")
def stage_outcomes() -> tuple[Outcomes, Outcomes]:
    rng = np.random.default_rng(12)
    strength = np.linspace(0.8, -0.8, 16)
    gap = strength[:, None] - strength[None, :]
    p = 1.0 / (1.0 + np.exp(-gap))
    stage = StageInput(NAMES, tuple(range(1, 17)), p, p * p * (3 - 2 * p))
    search = Outcomes.from_result(simulate_swiss(stage, 30_000, rng))
    holdout = Outcomes.from_result(simulate_swiss(stage, 30_000, rng))
    return search, holdout


def test_comparison_is_paired(stage_outcomes: tuple[Outcomes, Outcomes]) -> None:
    _, holdout = stage_outcomes
    a = Card(three_oh=(0, 1), advance=(2, 3, 4, 5, 6, 7), oh_three=(14, 15))
    b = Card(three_oh=(0, 2), advance=(1, 3, 4, 5, 6, 7), oh_three=(14, 15))
    same = compare_cards(a, a, holdout)
    assert same.diff_p5 == 0 and same.diff_se == 0
    result = compare_cards(b, a, holdout)
    assert result.diff_p5 == pytest.approx(evaluate(b, holdout).p5 - evaluate(a, holdout).p5)
    unpaired = np.hypot(result.yours.se5, result.best.se5)  # SE if the cards had separate sims
    assert 0 < result.diff_se < 0.85 * unpaired  # scoring both on the same sims removes noise


def test_sabotaged_pick_is_the_costliest(stage_outcomes: tuple[Outcomes, Outcomes]) -> None:
    search, holdout = stage_outcomes
    best = optimize(search, restarts=5).best.card
    groups = [list(g) for g in best.groups]
    weakest = max(t for t in range(16) if t not in best.teams)  # bottom seed not on the card
    sabotaged_team = groups[THREE_OH][0]
    groups[THREE_OH][0] = weakest  # a bottom team to go 3-0
    card = Card.from_groups(groups)

    costs = pick_costs(card, search, holdout, NAMES)
    assert len(costs) == 10
    worst = costs[0]
    assert (worst.category, worst.team) == (THREE_OH, weakest)
    assert worst.gain > 0.02 and worst.fixed is not None
    assert worst.change is not None and NAMES[weakest] in worst.change
    assert evaluate(worst.fixed, holdout).p5 == pytest.approx(
        evaluate(card, holdout).p5 + worst.gain
    )
    assert [c.gain for c in costs] == sorted((c.gain for c in costs), reverse=True)
    assert sabotaged_team not in card.teams
    assert all(c.change is None for c in costs if c.gain == 0)
    assert {c.category for c in costs} == {0, 1, 2} and OH_THREE in {c.category for c in costs}


def test_cli_card_section(capsys: pytest.CaptureFixture[str]) -> None:
    card = (
        "3-0: Team 01, Team 03; 0-3: Team 15, Team 16; "
        "advance: Team 02, Team 04, Team 05, Team 06, Team 07, Team 08"
    )
    code = main(
        [str(EXAMPLE), "--sims", "3000", "--eval-sims", "2000", "--restarts", "2", "--card", card]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Your card vs the optimizer's best (holdout sims, same sims for both)" in out
    assert "your card" in out and "optimizer #1" in out and "paired +/-" in out
    assert "Your picks, costliest first" in out
    assert out.count("P(correct)") == 10


def test_cli_rejects_a_bad_card(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([str(EXAMPLE), "--sims", "1000", "--card", "3-0: Team 01, Vitality"])
    err = capsys.readouterr().err
    assert code == 2 and "unknown team 'Vitality'" in err
