import numpy as np

from pickem.data.stage_file import StageFile
from pickem.fantasy.outcomes import OT_BASE, MapOutcomes, simulate_map_outcomes
from pickem.swiss import StageInput


def test_matches_agree_with_records(outcomes: MapOutcomes) -> None:
    assert (outcomes.won.sum(axis=2) == outcomes.wins).all()
    assert ((outcomes.played & ~outcomes.won).sum(axis=2) == outcomes.losses).all()
    assert (outcomes.played[:, :, :3]).all()
    assert ((outcomes.wins == 3) | (outcomes.losses == 3)).all()


def test_maps_follow_the_match(outcomes: MapOutcomes) -> None:
    on = outcomes.map_won >= 0
    maps = on.sum(axis=3)
    won_maps = (outcomes.map_won == 1).sum(axis=3)
    lost_maps = (outcomes.map_won == 0).sum(axis=3)
    bo1 = outcomes.best_of == 1
    bo3 = outcomes.best_of == 3
    assert (maps[bo1] == 1).all()
    assert np.isin(maps[bo3], [2, 3]).all()
    assert (won_maps[outcomes.won & bo3] == 2).all()
    assert (lost_maps[~outcomes.won & bo3] == 2).all()
    assert (maps[~outcomes.played] == 0).all()
    winner_rounds = np.where(outcomes.map_won == 1, outcomes.own_rounds, outcomes.opp_rounds)
    loser_rounds = np.where(outcomes.map_won == 1, outcomes.opp_rounds, outcomes.own_rounds)
    assert np.isin(winner_rounds[on], [13, 16, 19]).all()
    assert (loser_rounds[on] < winner_rounds[on]).all()
    assert (loser_rounds[on & (winner_rounds == 13)] <= 11).all()


def test_even_maps_sweep_half_the_time(stage_file: StageFile) -> None:
    half = np.full((16, 16), 0.5)
    stage = StageInput(stage_file.stage.teams, stage_file.stage.seeds, half, half)
    out = simulate_map_outcomes(StageFile("even", stage), 4000, np.random.default_rng(3))
    bo3_wins = out.won & (out.best_of == 3)
    sweep = ((out.map_won == 0).sum(axis=3) == 0)[bo3_wins]
    assert abs(sweep.mean() - 0.5) < 0.02  # 1 / (3 - 2p) at p = 0.5
    on = out.map_won >= 0
    overtime = (out.own_rounds + out.opp_rounds)[on] > 24
    assert abs(overtime.mean() - OT_BASE) < 0.01
