from pathlib import Path

import numpy as np
import pytest

from pickem.data.stage_file import StageFile
from pickem.fantasy.live import booster_indices, plan_round, played_matches
from pickem.fantasy.outcomes import simulate_map_outcomes
from pickem.fantasy.performance import simulate_players
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import BOOSTER_INDEX


def test_fixed_results_replay_reality(
    tmp_path: Path, stage_file: StageFile, pool: PlayerPool
) -> None:
    teams = stage_file.stage.teams
    # Round 1 is seed k v seed k+8; let every lower seed win.
    rows = ["round,team1,team2,winner"] + [
        f"1,{teams[k]},{teams[k + 8]},{teams[k + 8]}" for k in range(8)
    ]
    path = tmp_path / "r.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    done = played_matches(path, teams, 2)
    assert len(done) == 8
    fixed = [(teams.index(m.winner), teams.index(m.loser)) for m in done]
    out = simulate_map_outcomes(stage_file, 500, np.random.default_rng(2), fixed=fixed)
    assert (~out.won[:, :8, 0]).all() and out.won[:, 8:, 0].all()
    with pytest.raises(ValueError, match="round 1 needs 8"):
        path.write_text("\n".join(rows[:5]) + "\n", encoding="utf-8")
        played_matches(path, teams, 2)

    sims = simulate_players(out, pool, np.random.default_rng(3), keep_rounds=True)
    used = booster_indices("Hellcase,Top of scoreboard")
    plan = plan_round(sims, (4, 9, 14, 19, 24), 1, used)
    assert len(set(plan.roles)) == 5
    chosen = [b for b, _, _ in plan.boosters_now] + [b for b, _, _, _ in plan.boosters_later]
    assert len(chosen) == len(set(chosen)) == 16
    assert BOOSTER_INDEX["Hellcase"] not in chosen
    assert len({pos for _, pos, _ in plan.boosters_now}) == len(plan.boosters_now) <= 5
    with pytest.raises(ValueError, match="unknown booster"):
        booster_indices("Hellcase, Nope")
