from dataclasses import replace

import numpy as np

from pickem.fantasy.outcomes import MapOutcomes
from pickem.fantasy.performance import (
    R_DEV,
    R_INTERCEPT,
    R_TEAM,
    R_WON,
    PlayerSims,
    expected_card_level,
    team_average_card,
)
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import BOOSTER_INDEX, LEADER_INDEX, TEAM_LOSS, TEAM_WIN


def test_team_points_follow_the_record(
    outcomes: MapOutcomes, pool: PlayerPool, sims: PlayerSims
) -> None:
    wins = outcomes.wins[:, pool.team].astype(int)
    losses = outcomes.losses[:, pool.team].astype(int)
    out = ~outcomes.advanced[:, pool.team]
    assert (sims.team[out] == (TEAM_WIN * wins + TEAM_LOSS * losses)[out]).all()
    # Advanced: every padded round is worth the truncated mean of the played rounds.
    played = wins + losses
    per_round = np.trunc((TEAM_WIN * wins + TEAM_LOSS * losses) / played)
    expected = TEAM_WIN * wins + TEAM_LOSS * losses + per_round * (5 - played)
    assert (sims.team[~out] == expected[~out]).all()


def test_rating_points_track_the_card(pool: PlayerPool, sims: PlayerSims) -> None:
    # Teams that played all 5 rounds went 3-2 or 2-3, so their average match is worth about
    # 50 * (E[map rating at a 50% win rate] - 1); within a team, better cards score more.
    matches = sims.played.sum(axis=2)
    per_match = (sims.rating[matches == 5] / 5).mean()
    assert abs(per_match - 50 * (R_INTERCEPT + 0.5 * R_WON - 1)) < 1.5
    dev = pool.stats["rating"] - team_average_card(pool)
    rel_points = (
        sims.rating.mean(axis=0)
        - np.bincount(pool.team, weights=sims.rating.mean(axis=0))[pool.team] / 5
    )
    assert np.corrcoef(dev, rel_points)[0, 1] > 0.8


def test_team_card_average_counts_only_with_weight(pool: PlayerPool) -> None:
    shifted = pool.stats["rating"] + np.where(pool.team == 3, 0.2, 0.0)
    lifted = replace(pool, stats={**pool.stats, "rating": shifted})
    for weight, rise in ((0.0, 0.0), (1.0, R_TEAM * 0.2)):
        before = expected_card_level(pool, weight)
        after = expected_card_level(lifted, weight)
        assert np.allclose((after - before)[pool.team == 3], rise)
        assert np.allclose((after - before)[pool.team != 3], 0.0)
    team0 = pool.team == 0
    level = expected_card_level(pool)[team0]
    assert np.allclose(level - R_INTERCEPT, R_DEV * (pool.stats["rating"][team0] - 1.12))


def test_teammates_move_together(pool: PlayerPool, sims: PlayerSims) -> None:
    a, b = np.flatnonzero(pool.team == 0)[:2]
    # Shared map results give about 0.3 per map (win/loss gap 0.36, individual SD 0.31).
    assert np.corrcoef(sims.rating[:, a], sims.rating[:, b])[0, 1] > 0.2


def test_leader_and_boosters(outcomes: MapOutcomes, pool: PlayerPool, sims: PlayerSims) -> None:
    team0 = np.flatnonzero(pool.team == 0)
    leader = sims.roles[:, team0, LEADER_INDEX]
    assert (leader == leader[:, :1]).all()  # the same for every teammate
    probs = sims.booster_probabilities()
    assert probs.shape == (pool.size, 5, 18)
    assert (probs[:, :3, BOOSTER_INDEX["Hellcase"]] == 1).all()
    # Exactly the players of a team top its scoreboard; at least one of them every round.
    top = (sims.boosts[:, team0, 0] >> np.uint32(BOOSTER_INDEX["Top of scoreboard"])) & 1
    assert (top.sum(axis=1) >= 1).all()
    played_r5 = outcomes.played[:, 0, 4]
    hell = (sims.boosts[:, team0[0], 4] >> np.uint32(BOOSTER_INDEX["Hellcase"])) & 1
    assert (hell.astype(bool) == played_r5).all()
    assert ((probs >= 0) & (probs <= 1)).all()
