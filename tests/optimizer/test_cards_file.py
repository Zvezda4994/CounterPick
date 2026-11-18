from pathlib import Path

import pytest

from pickem.optimizer.card import Card
from pickem.optimizer.cards_file import COLUMNS, CardEntry, append_card, load_cards

NAMES = (
    "Ninjas in Pyjamas", "PARIVISION", "HEROIC", "FaZe", "Liquid", "Eternal Fire", "100 Thieves",
    "9z", "M80", "DENDELE", "Imperial", "Wildcard", "GOSH", "FlyQuest", "Chinggis Warriors",
    "Lynn Vision",
)  # fmt: skip
HEADER = ",".join(COLUMNS)
ME = "Me,me,,9z,NiP,lynn vision,Chinggis Warriors,100 Thieves,PARIVISION,FaZe,HEROIC,M80,liquid"


def write(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "cards.csv"
    path.write_text("\n".join(["# my cards", HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_reads_cards_with_loose_names(tmp_path: Path) -> None:
    link = "https://example.com/post#picks"  # a '#' inside a cell is not a comment
    advance = ",".join([*NAMES[2:6], "M80", "DENDELE"])
    path = write(tmp_path, ME, f"Analyst,ANALYST,{link},PARIVISION,100t,GOSH,Wildcard,{advance}")
    me, analyst = load_cards(path, NAMES)
    assert me.name == "Me" and me.source == "me" and me.link == ""
    assert me.card == Card(three_oh=(7, 0), oh_three=(15, 14), advance=(6, 1, 3, 2, 8, 4))
    assert analyst.source == "analyst" and analyst.link == link
    assert analyst.card.three_oh == (1, 6) and analyst.card.oh_three == (11, 12)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((ME.replace(",me,", ",mom,"),), "card 1 \\(Me\\): source must be one of"),
        ((ME, ME), "card 2 \\(Me\\): another card already has this name"),
        ((ME.replace("9z", "Vitality"),), "card 1 \\(Me\\): unknown team 'Vitality'"),
        ((ME.replace("M80", "9z"),), "card 1 \\(Me\\): a team can be picked in only one"),
        ((ME.rsplit(",", 1)[0],), "card 1: expected 13 cells"),
        ((ME.replace("Me,", ",", 1),), "card 1: name is empty"),
    ],
)
def test_invalid_cards(tmp_path: Path, rows: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_cards(write(tmp_path, *rows), NAMES)


def test_wrong_header(tmp_path: Path) -> None:
    path = tmp_path / "cards.csv"
    path.write_text("name,source,picks\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns must be"):
        load_cards(path, NAMES)


def test_append_creates_and_refuses_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "cards" / "stage.csv"
    card = Card(three_oh=(0, 1), oh_three=(14, 15), advance=(2, 3, 4, 5, 6, 7))
    append_card(path, CardEntry("Seed chalk", "seed chalk", "", card), NAMES)
    append_card(path, CardEntry("Me", "me", "", card), NAMES)
    entries = load_cards(path, NAMES)
    assert [e.name for e in entries] == ["Seed chalk", "Me"]
    assert path.read_text(encoding="utf-8").splitlines()[0] == HEADER
    with pytest.raises(ValueError, match="already has a card named 'me'"):
        append_card(path, CardEntry("me", "me", "", card), NAMES)
