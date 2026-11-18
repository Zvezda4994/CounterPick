"""Replay the three Swiss stages of the BLAST.tv Austin Major 2025 against our rules engine.

Match data: tests/fixtures/austin_2025.json (Liquipedia, CC BY-SA 3.0). Seeds and published
Difficulty Scores come from Valve, so the pairing logic is checked against data that does not
depend on Liquipedia's own bracket.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pickem.swiss import StageInput, SwissResult, simulate_swiss

FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "austin_2025.json").read_text(encoding="utf-8")
)
STAGES: dict[int, dict[str, Any]] = {s["stage"]: s for s in FIXTURE["stages"]}

# Valve Regional Standings, global list of 2025-05-05: the invitation VRS used to seed Austin.
# https://github.com/ValveSoftware/counter-strike_regional_standings/blob/main/invitation/2025/standings_global_2025_05_05.md
VRS_GLOBAL_RANK = {
    "vit": 1, "mouz": 2, "falcons": 3, "spirit": 4, "g2": 5, "mongolz": 6, "aurora": 7,
    "faze": 8, "3dmax": 10, "navi": 11, "tl": 12, "vp": 13, "col": 14, "heroic": 15,
    "pain": 17, "b8": 18, "furia": 20, "mibr": 22, "betboom": 24, "tyloo": 25, "lvg": 26,
    "wildcard": 27, "fq": 29, "og": 30, "chinggis warriors": 31, "imp": 32, "m80": 34,
    "nemiga": 35, "nrg": 37, "legacy": 47, "metizport": 48, "fluxo": 61,
}  # fmt: skip

# Valve's rulebook, "Results: 2025 Austin" (L606-L643 at the pinned commit): final record and
# Difficulty for every team whose stage result is published.
PUBLISHED: dict[int, dict[str, tuple[str, int]]] = {
    1: {
        "fq": ("2-3", 5), "nrg": ("2-3", 5), "wildcard": ("2-3", 4), "imp": ("1-3", 3),
        "chinggis warriors": ("1-3", 2), "col": ("1-3", -1), "metizport": ("0-3", -2),
        "fluxo": ("0-3", -5),
    },
    2: {
        "b8": ("2-3", 3), "mibr": ("2-3", 0), "heroic": ("2-3", -4), "tyloo": ("1-3", -1),
        "m80": ("1-3", -2), "falcons": ("1-3", -4), "betboom": ("0-3", 0), "og": ("0-3", -1),
    },
    3: {
        "spirit": ("3-0", 1), "furia": ("3-0", -2), "faze": ("3-1", -1), "navi": ("3-1", -3),
        "vp": ("2-3", 6), "legacy": ("2-3", 2), "g2": ("2-3", 0), "aurora": ("1-3", 5),
        "3dmax": ("1-3", 4), "lvg": ("1-3", 0), "nemiga": ("0-3", 5), "tl": ("0-3", 0),
    },
}  # fmt: skip


def actual_pairs(stage: int, round_index: int) -> set[frozenset[str]]:
    return {frozenset(m["teams"]) for m in STAGES[stage]["rounds"][round_index]}


def simulated_pairs(res: SwissResult, sim: int, round_index: int) -> set[frozenset[str]]:
    log = res.matches
    assert log is not None
    m = int((log.best_of[round_index] > 0).sum())
    return {
        frozenset((res.teams[h], res.teams[lo]))
        for h, lo in zip(log.high[sim, round_index, :m], log.low[sim, round_index, :m], strict=True)
    }


def replay(stage: int, n_sims: int = 32) -> SwissResult:
    """Simulate with P(actual winner beats actual loser) = 1 for every match that was played
    and 0.5 for every pairing that never happened. If the engine pairs exactly as reality did,
    every sim reproduces the real stage; any wrong pairing shows up as a coin flip."""
    teams = STAGES[stage]["seeds"]
    index = {t: i for i, t in enumerate(teams)}
    p = np.full((16, 16), 0.5)
    for rnd in STAGES[stage]["rounds"]:
        for match in rnd:
            winner = match["winner"]
            (loser,) = set(match["teams"]) - {winner}
            p[index[winner], index[loser]] = 1.0
            p[index[loser], index[winner]] = 0.0
    stage_input = StageInput(tuple(teams), tuple(range(1, 17)), p, p)
    return simulate_swiss(stage_input, n_sims, np.random.default_rng(stage), record_matches=True)


def test_fixture_is_complete() -> None:
    for stage in STAGES.values():
        assert len(set(stage["seeds"])) == 16
        assert [len(r) for r in stage["rounds"]] == [8, 8, 8, 6, 3]


def test_stage1_seeds_are_the_vrs_order() -> None:
    teams = STAGES[1]["seeds"]
    assert sorted(teams, key=VRS_GLOBAL_RANK.__getitem__) == teams


@pytest.mark.parametrize("stage", [2, 3])
def test_invited_teams_are_seeded_1_to_8_by_vrs(stage: int) -> None:
    invited = STAGES[stage]["seeds"][:8]
    assert sorted(invited, key=VRS_GLOBAL_RANK.__getitem__) == invited


def test_round1_pairings_from_real_stage1_seeds() -> None:
    """Seeds from Valve's VRS alone reproduce the actual Austin Stage 1 opening matches."""
    teams = sorted(STAGES[1]["seeds"], key=VRS_GLOBAL_RANK.__getitem__)
    coin = np.full((16, 16), 0.5)
    res = simulate_swiss(
        StageInput(tuple(teams), tuple(range(1, 17)), coin, coin),
        10,
        np.random.default_rng(0),
        record_matches=True,
    )
    expected = {
        frozenset(pair)
        for pair in [
            ("col", "og"),
            ("heroic", "chinggis warriors"),
            ("b8", "imp"),
            ("betboom", "nemiga"),
            ("tyloo", "nrg"),
            ("lvg", "legacy"),
            ("wildcard", "metizport"),
            ("fq", "fluxo"),
        ]
    }
    assert actual_pairs(1, 0) == expected
    for sim in range(res.n_sims):
        assert simulated_pairs(res, sim, 0) == expected


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_every_round_reproduces_the_real_pairings(stage: int) -> None:
    res = replay(stage)
    for sim in range(res.n_sims):
        for r in range(5):
            assert simulated_pairs(res, sim, r) == actual_pairs(stage, r), (
                f"stage {stage} round {r + 1}"
            )
    assert (res.rematches == 0).all()


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_final_records_and_difficulty_match_valve(stage: int) -> None:
    res = replay(stage, n_sims=1)
    index = {t: i for i, t in enumerate(res.teams)}
    for team, (record, difficulty) in PUBLISHED[stage].items():
        i = index[team]
        assert f"{res.wins[0, i]}-{res.losses[0, i]}" == record, team
        assert res.difficulty[0, i] == difficulty, team


@pytest.mark.parametrize("stage", [1, 2])
def test_final_seed_gives_next_stage_seeds_9_to_16(stage: int) -> None:
    res = replay(stage, n_sims=1)
    advancing = sorted(
        (t for i, t in enumerate(res.teams) if res.final_seed[0, i] <= 8),
        key=lambda t: res.final_seed[0, res.teams.index(t)],
    )
    assert advancing == STAGES[stage + 1]["seeds"][8:]


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_bo3_exactly_where_real_matches_went_past_one_map(stage: int) -> None:
    res = replay(stage, n_sims=1)
    log = res.matches
    assert log is not None
    maps = {
        (r, frozenset(m["teams"])): m["maps"]
        for r, rnd in enumerate(STAGES[stage]["rounds"])
        for m in rnd
    }
    for r in range(5):
        m = int((log.best_of[r] > 0).sum())
        for j in range(m):
            pair = frozenset((res.teams[log.high[0, r, j]], res.teams[log.low[0, r, j]]))
            played = maps[(r, pair)]
            if log.best_of[r, j] == 1:
                assert played == 1, (r, pair)
            else:
                assert played in (2, 3), (r, pair)
