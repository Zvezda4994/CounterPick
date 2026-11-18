import json
from pathlib import Path
from typing import Any

import pytest

from pickem.swiss.replay import MatchResult, replay_stage
from tests.swiss.test_austin_2025 import PUBLISHED

STAGES: dict[int, dict[str, Any]] = {
    s["stage"]: s
    for s in json.loads(
        (Path(__file__).parents[1] / "fixtures" / "austin_2025.json").read_text(encoding="utf-8")
    )["stages"]
}


def rounds(stage: int) -> list[list[MatchResult]]:
    return [
        [MatchResult(m["teams"][0], m["teams"][1], m["winner"], m["maps"]) for m in matches]
        for matches in STAGES[stage]["rounds"]
    ]


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_replay_matches_valves_published_results(stage: int) -> None:
    teams = STAGES[stage]["seeds"]
    replay = replay_stage(teams, range(1, 17), rounds(stage))
    for team, (record, difficulty) in PUBLISHED[stage].items():
        assert replay.record(team) == record
        assert replay.difficulty[teams.index(team)] == difficulty
    if stage < 3:
        assert list(replay.advancing) == STAGES[stage + 1]["seeds"][8:]
    assert sorted(replay.final_seed) == list(range(1, 17))


def test_first_wrong_round_is_named() -> None:
    teams = STAGES[1]["seeds"]
    played = rounds(1)
    first = played[0][0]
    played[0][0] = MatchResult(first.team1, first.team2, first.loser)
    with pytest.raises(ValueError, match="round 2 does not follow the Swiss rules"):
        replay_stage(teams, range(1, 17), played)


def test_conflicting_rematch_is_rejected() -> None:
    teams = [f"T{i:02d}" for i in range(16)]
    pairs = [(teams[i], teams[i + 8]) for i in range(8)]
    first = [MatchResult(a, b, a) for a, b in pairs]
    again = [MatchResult(a, b, b) for a, b in pairs]  # same pairs, opposite winners
    with pytest.raises(ValueError, match="met twice with different winners"):
        replay_stage(teams, range(1, 17), [first, again, first, first[:6], first[:3]])


def test_match_result_validation() -> None:
    with pytest.raises(ValueError, match="cannot play itself"):
        MatchResult("A", "A", "A")
    with pytest.raises(ValueError, match="did not play"):
        MatchResult("A", "B", "C")
    assert MatchResult("A", "B", "B").loser == "A"
