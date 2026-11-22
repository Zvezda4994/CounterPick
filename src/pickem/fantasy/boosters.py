"""HLTV's own booster trigger rates, folded into the simulated booster outcomes.

The booster picker on a team page shows, for each of the team's players and each booster, the
"average trigger rate (ATR)": the share of the player's past matches in which the booster would
have succeeded. ``calibrate_boosters`` makes the simulated rates match it while keeping whatever
link the model has between a booster and the player's performance (Carry still tends to land on
high-rating maps):

- ATR is per match. A booster succeeds if any map of the match qualifies, so a per-map rate q
  is backed out with 1 - (1 - q)^m = ATR, where m = HISTORICAL_MAPS_PER_MATCH (IEM Beijing
  2026: 74 maps in 30 matches, mostly BO3), and a round with k maps then targets 1 - (1 - q)^k.
- For each player, booster and map count, the simulated successes are thinned at random (if the
  model triggers too often) or failures are flipped at random (if too rarely) so that the share
  of rounds that succeed equals the target. Players without an ATR keep the model's rates.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pickem.fantasy.performance import PlayerSims
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import BOOSTER_NAMES, N_BOOSTERS

HISTORICAL_MAPS_PER_MATCH = 2.47
_ALIASES = {
    "mister consistent": "Mister/Miss consistent",
    "miss consistent": "Mister/Miss consistent",
}


def _booster_name(raw: str) -> str:
    folded = raw.strip().casefold()
    for name in BOOSTER_NAMES:
        if name.casefold() == folded:
            return name
    if folded in _ALIASES:
        return _ALIASES[folded]
    raise ValueError(f"unknown booster {raw!r}")


def load_booster_rates(path: str | Path, pool: PlayerPool) -> dict[int, NDArray[np.float64]]:
    """CSV with a ``player`` column and one column per booster (percent of matches)."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if not rows or "player" not in rows[0]:
        raise ValueError(f"{path}: needs a player column")
    columns = {c: _booster_name(c) for c in rows[0] if c != "player"}
    missing = set(BOOSTER_NAMES) - set(columns.values())
    if missing:
        raise ValueError(f"{path}: missing booster column(s) {', '.join(sorted(missing))}")
    rates: dict[int, NDArray[np.float64]] = {}
    for row in rows:
        values = np.zeros(N_BOOSTERS)
        for column, name in columns.items():
            pct = float(row[column])
            if not 0 <= pct <= 100:
                raise ValueError(f"{path}: {row['player']} {name} must be 0-100, got {pct}")
            values[BOOSTER_NAMES.index(name)] = pct / 100
        rates[pool.index(row["player"])] = values
    return rates


def per_map_rate(atr: NDArray[np.float64], maps_per_match: float) -> NDArray[np.float64]:
    """Per-map rate q with 1 - (1 - q)^maps_per_match = atr."""
    result: NDArray[np.float64] = 1.0 - (1.0 - np.clip(atr, 0.0, 1.0)) ** (1.0 / maps_per_match)
    return result


def calibrate_boosters(
    sims: PlayerSims,
    rates: Mapping[int, NDArray[np.float64]],
    rng: np.random.Generator,
    maps_per_match: float = HISTORICAL_MAPS_PER_MATCH,
) -> PlayerSims:
    if sims.maps is None:
        raise ValueError("these sims do not record maps per round")
    boosts = sims.boosts.copy()
    for player, atr in rates.items():
        q = per_map_rate(np.asarray(atr, dtype=np.float64), maps_per_match)
        maps = sims.maps[:, player, :]
        for b in range(N_BOOSTERS):
            bit = np.uint32(1) << np.uint32(b)
            hit = (boosts[:, player, :] & bit) != 0
            new = hit.copy()
            for k in np.unique(maps[maps > 0]):
                sel = maps == k
                target = 1.0 - (1.0 - q[b]) ** int(k)
                current = float(hit[sel].mean())
                u = rng.random(int(sel.sum()))
                if current >= target:
                    keep = u < (target / current if current > 0 else 0.0)
                    new[sel] = hit[sel] & keep
                else:
                    new[sel] = hit[sel] | (u < (target - current) / (1.0 - current))
            boosts[:, player, :] = (boosts[:, player, :] & ~bit) | (new.astype(np.uint32) << b)
    return replace(sims, boosts=boosts)
