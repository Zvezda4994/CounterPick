import json
import re
from pathlib import Path
from typing import Any

import pytest

from pickem.data.liquipedia import parse_rounds, parse_seed_order, parse_stage

FIXTURES = Path(__file__).parents[1] / "fixtures"
STAGES: dict[int, dict[str, Any]] = {
    s["stage"]: s
    for s in json.loads((FIXTURES / "austin_2025.json").read_text(encoding="utf-8"))["stages"]
}


def wiki(stage: int) -> str:
    return (FIXTURES / "liquipedia" / f"austin_2025_stage_{stage}.wiki").read_text(encoding="utf-8")


@pytest.mark.parametrize("stage", [1, 2])
def test_austin_pages_match_the_fixture(stage: int) -> None:
    parsed = parse_stage(wiki(stage))
    assert list(parsed.seeds) == STAGES[stage]["seeds"]
    for got, want in zip(parsed.rounds, STAGES[stage]["rounds"], strict=True):
        assert [(m.team1, m.team2, m.winner, m.maps) for m in got] == [
            (*w["teams"], w["winner"], w["maps"]) for w in want
        ]


def standings(aliases: list[str]) -> str:
    rows = "\n".join(
        f"|{{{{TeamOpponent|{a}|tiebreaker={16 - i}}}}}" for i, a in enumerate(aliases)
    )
    return (
        f"{{{{SwissStandings|format=swiss3\n|round1={{{{Round|matchgroups=x}}}}\n\n{rows}\n}}}}\n"
    )


def match(a: str, b: str, extra: str = "", score: tuple[int, int] | None = None) -> str:
    s1 = f"|score={score[0]}" if score else ""
    s2 = f"|score={score[1]}" if score else ""
    return (
        "|{{Match\n"
        f"\t|opponent1={{{{TeamOpponent|{a}{s1}}}}}|opponent2={{{{TeamOpponent|{b}{s2}}}}}\n"
        f"{extra}\t}}}}\n"
    )


def test_singapore_layout_seed_order() -> None:
    aliases = [f"t{i:02d}" for i in range(1, 17)]
    shuffled = standings(aliases[::-1]).replace("tiebreaker=", "tb=")
    assert parse_seed_order(standings(aliases)) == tuple(aliases)
    with pytest.raises(ValueError, match="16 seeded teams"):
        parse_seed_order(shuffled)
    with pytest.raises(ValueError, match="not decided"):
        parse_seed_order(standings([*aliases[:15], ""]))


def test_every_way_a_winner_can_be_given() -> None:
    map_win_for_2 = "\t|map1={{Map|map=Nuke|finished=true\n\t\t|t1t=3|t1ct=4|t2t=8|t2ct=5}}\n"
    overtime_for_1 = (
        "\t|map1={{Map|map=Nuke|finished=true\n\t\t|t1t=6|t1ct=6|t2t=6|t2ct=6"
        "\n\t\t|o1t1t=3|o1t1ct=1|o1t2t=1|o1t2ct=1}}\n"
    )
    text = "matchsection=Round 1\n" + "".join(
        [
            match("a", "b", "\t|winner=2\n" + map_win_for_2.replace("t2t=8", "t2t=1")),
            match("c", "d", score=(2, 1)),
            match("e", "f", map_win_for_2),
            match("g", "h", overtime_for_1),
            match("i", "j"),  # not played yet
        ]
    )
    (round1, *_) = parse_rounds(text)
    assert [(a, b, side) for a, b, side, _ in round1] == [
        ("a", "b", 2),  # explicit winner beats the map score
        ("c", "d", 1),
        ("e", "f", 2),
        ("g", "h", 1),
        ("i", "j", None),
    ]


def test_unplayed_match_and_unknown_alias_fail_clearly() -> None:
    text = wiki(1)
    last = text.rfind("|{{Match")
    unplayed = text[:last] + re.sub(r"(o\d+)?t[12](t|ct)=\d+", "", text[last:])
    with pytest.raises(ValueError, match=r"round 5: .* has no result"):
        parse_stage(unplayed)

    renamed = text.replace("TeamOpponent|og}}", "TeamOpponent|og-cs}}")
    with pytest.raises(ValueError, match="--alias og-cs="):
        parse_stage(renamed)
    assert parse_stage(renamed, {"og-cs": "og"}).rounds == parse_stage(text).rounds


def test_incomplete_round_counts() -> None:
    text = wiki(1)
    last = text.rfind("|{{Match")
    with pytest.raises(ValueError, match="round 5 has 2 matches"):
        parse_stage(text[:last])
