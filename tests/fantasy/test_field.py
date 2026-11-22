from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pickem.fantasy.field import ownership, sample_field
from pickem.fantasy.players import PlayerPool, load_players
from tests.fantasy.conftest import write_players


def test_ownership_keeps_known_rates_and_adds_to_five(pool: PlayerPool) -> None:
    picked = pool.picked.copy()
    picked[:3] = [0.40, 0.25, 0.10]
    shares = ownership(replace(pool, picked=picked))
    assert np.allclose(shares[:3], [0.40, 0.25, 0.10])
    assert np.isclose(shares.sum(), 5.0)
    with pytest.raises(ValueError, match="add up"):
        ownership(replace(pool, picked=np.full(pool.size, 0.07)))


def test_field_lineups_are_legal_and_match_the_rates(pool: PlayerPool) -> None:
    shares = ownership(pool)
    lineups = sample_field(pool, shares, 6000, 1_000_000, 2, np.random.default_rng(4))
    assert (np.sort(lineups, axis=1)[:, 1:] != np.sort(lineups, axis=1)[:, :-1]).all()
    assert (pool.price[lineups].sum(axis=1) <= 1_000_000).all()
    per_team = np.array([np.max(np.bincount(pool.team[row], minlength=16)) for row in lineups])
    assert (per_team <= 2).all()
    realized = np.bincount(lineups.ravel(), minlength=pool.size) / len(lineups)
    assert np.abs(realized - shares).max() < 0.03


def test_players_without_cards_are_estimated(tmp_path: Path, pool: PlayerPool) -> None:
    path = write_players(tmp_path / "p.csv", tuple(sorted(set(pool.team_names))), cards=False)
    bare = load_players(path, tuple(sorted(set(pool.team_names))))
    assert len(bare.estimated) == bare.size
    assert np.isclose(bare.stats["rating"][0], 1.0589 + 0.0387 * 3.0)
    with pytest.raises(ValueError, match="no player"):
        bare.index("nobody")
    assert bare.index("p000") == 0
