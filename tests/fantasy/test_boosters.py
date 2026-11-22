from pathlib import Path

import numpy as np
import pytest

from pickem.fantasy.boosters import calibrate_boosters, load_booster_rates, per_map_rate
from pickem.fantasy.performance import PlayerSims
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import BOOSTER_INDEX, BOOSTER_NAMES


def _rates_csv(path: Path, players: list[str], pct: dict[str, float]) -> Path:
    header = ["player", *BOOSTER_NAMES]
    rows = [",".join(header)]
    for p in players:
        rows.append(",".join([p, *(str(pct.get(b, 20.0)) for b in BOOSTER_NAMES)]))
    path.write_text("\n".join(rows).replace("Mister/Miss consistent", "Mister consistent") + "\n")
    return path


def test_calibrated_rates_hit_the_target(
    tmp_path: Path, pool: PlayerPool, sims: PlayerSims
) -> None:
    rates = load_booster_rates(
        _rates_csv(
            tmp_path / "r.csv", ["P000", "P001"], {"Kobe": 45, "Assist": 85, "Hellcase": 100}
        ),
        pool,
    )
    assert set(rates) == {pool.index("P000"), pool.index("P001")}
    out = calibrate_boosters(sims, rates, np.random.default_rng(9))
    assert sims.maps is not None
    q = per_map_rate(rates[pool.index("P000")], 2.47)
    p = pool.index("P000")
    for name in ("Kobe", "Assist", "Carry"):
        b = BOOSTER_INDEX[name]
        bit = (out.boosts[:, p, :] >> np.uint32(b)) & 1
        for k in (1, 3):
            sel = sims.maps[:, p, :] == k
            target = 1 - (1 - q[b]) ** k
            assert abs(bit[sel].mean() - target) < 0.03
        assert (bit[~sims.played[:, p, :]] == 0).all()
    hell = (out.boosts[:, p, :] >> np.uint32(BOOSTER_INDEX["Hellcase"])) & 1
    assert (hell.astype(bool) == sims.played[:, p, :]).all()
    other = pool.index("P010")
    assert (out.boosts[:, other] == sims.boosts[:, other]).all()


def test_bad_rate_files(tmp_path: Path, pool: PlayerPool) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("player,Kobe\nP000,40\n")
    with pytest.raises(ValueError, match="missing booster"):
        load_booster_rates(path, pool)
    with pytest.raises(ValueError, match="0-100"):
        load_booster_rates(_rates_csv(tmp_path / "x.csv", ["P000"], {"Kobe": 140}), pool)
