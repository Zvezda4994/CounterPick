import numpy as np
import pytest

from pickem.fantasy.rules import (
    LEADER_INDEX,
    ROLE_NAMES,
    SEASON_TIERS,
    STAT_ROLES,
    leader_points,
    padded,
    rating_points,
    season_points,
)


@pytest.mark.parametrize(
    ("rating", "points"),
    # Match ratings from HLTV match pages and the rating points HLTV fantasy gave for them
    # (IEM Beijing 2026, game 671).
    [(1.94, 47), (1.08, 4), (0.80, -10), (1.04, 2), (1.26, 13), (1.74, 37), (1.00, 0)],
)
def test_rating_points_match_hltv(rating: float, points: int) -> None:
    assert int(rating_points(rating)) == points


def test_padding_matches_hltv_rows() -> None:
    # Spinx (game 671): rounds of 47/6/5 and 10/6/2 (rating/team/role), then a padded round of
    # 28/6/3; Techno: 9/6/2 and 0/6/2, padded 4/6/2. Here the padded rounds are rounds 3-5.
    played = np.array([True, True, False, False, False])
    for pts, pad_value in (([47, 10], 28), ([6, 6], 6), ([5, 2], 3), ([9, 0], 4), ([2, 2], 2)):
        total = padded(np.array([*pts, 0, 0, 0]), played, np.array(True))
        assert int(total) == sum(pts) + 3 * pad_value


def test_padding_truncates_toward_zero_and_skips_eliminated() -> None:
    played = np.array([True, True, True, False, False])
    pts = np.array([-5, 0, 0, 0, 0])
    assert int(padded(pts, played, np.array(True))) == -5 + 2 * -1
    assert int(padded(pts, played, np.array(False))) == -5
    assert int(padded(np.array([3, 3, 3, 3, 3]), np.ones(5, bool), np.array(True))) == 15


def test_role_thresholds() -> None:
    roles = {r.name: r for r in STAT_ROLES}
    awp = roles["Main AWP"].points([0.36, 0.35, 0.20, 0.19])
    assert awp.tolist() == [5, 2, 2, -2]
    camper = roles["Camper"].points([0.54, 0.55, 0.65, 0.66])
    assert camper.tolist() == [5, 2, 2, -2]
    noob = roles["Noob"].points([0.84, 0.85, 1.12, 1.13])
    assert noob.tolist() == [5, 2, 2, -2]
    assert ROLE_NAMES[LEADER_INDEX] == "Leader" and len(ROLE_NAMES) == 12


def test_leader() -> None:
    won = np.array([True, True, True, True, False])
    lost_maps = np.array([0, 1, 0, 0, 2])
    ot = np.array([False, False, True, False, False])
    assert leader_points(won, lost_maps, ot).tolist() == [5, 2, 2, 5, -2]


def test_season_points() -> None:
    cutoffs = np.array([[100, 90, 80, 70, 60, 50, 40, 30, 20, 10]], dtype=float)
    scores = np.array([150, 100, 99, 75, 10, 9])
    got = season_points(scores, np.repeat(cutoffs, len(scores), axis=0))
    tiers = [pts for _, pts in SEASON_TIERS]
    assert got.tolist() == [tiers[0], tiers[0], tiers[1], tiers[3], tiers[9], 0]
