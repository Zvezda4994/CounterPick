"""The player pool: one CSV row per draftable player.

Required columns: ``team``, ``player``, ``price``. Optional: ``hltv_rank`` (team world rank, used
by the field model) and the HLTV card stats shown on every player card in the draft (3-month
averages): ``rating``, ``ct_rating``, ``t_rating``, ``awp_kpr`` (AWP kills per round),
``hs_pct``, ``entry_pct``, ``clutch_pct``, ``support_pct``, ``multikill_pct`` (percent of
rounds), ``dpr``. ``picked_pct`` (share of HLTV-league teams that drafted the player) pins the
field model where known. Other columns are ignored.

A player without card stats gets price-based estimates fitted on IEM Beijing 2026 (game 671,
80 players): rating rises 0.039 per $10k (residual SD 0.075). Those players are listed in
``PlayerPool.estimated`` so the report can flag them; Main AWP cannot be judged without a card.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

CARD_STATS = (
    "rating",
    "ct_rating",
    "t_rating",
    "awp_kpr",
    "hs_pct",
    "entry_pct",
    "clutch_pct",
    "support_pct",
    "multikill_pct",
    "dpr",
)

# stat -> (value at $200k, change per $10k), least squares on game 671's 80 cards.
PRICE_FALLBACK: dict[str, tuple[float, float]] = {
    "rating": (1.0589, 0.0387),
    "ct_rating": (1.0806, 0.0383),
    "t_rating": (1.0362, 0.0390),
    "awp_kpr": (0.02, 0.0),
    "hs_pct": (51.68, -0.678),
    "entry_pct": (10.40, 0.595),
    "clutch_pct": (1.237, 0.0076),
    "support_pct": (21.33, -0.636),
    "multikill_pct": (16.13, 0.967),
    "dpr": (0.6435, -0.006),
}


@dataclass(frozen=True)
class PlayerPool:
    names: tuple[str, ...]
    team_names: tuple[str, ...]
    """Team of each player, as spelled in the stage file."""
    team: NDArray[np.intp]
    """Index of each player's team in the stage file's ``teams``."""
    price: NDArray[np.int64]
    stats: dict[str, NDArray[np.float64]]
    """Card stats per player (estimates where ``estimated``)."""
    hltv_rank: NDArray[np.float64]
    """Team world rank (NaN if not given)."""
    picked: NDArray[np.float64]
    """Share of league teams that drafted the player, 0-1 (NaN if unknown)."""
    estimated: tuple[str, ...] = field(default=())

    @property
    def size(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Player by exact name, then case-insensitive name, then unique case-insensitive
        prefix."""
        if name in self.names:
            return self.names.index(name)
        folded = name.casefold()
        exact = [i for i, n in enumerate(self.names) if n.casefold() == folded]
        if len(exact) == 1:
            return exact[0]
        prefix = [i for i, n in enumerate(self.names) if n.casefold().startswith(folded)]
        if len(prefix) == 1:
            return prefix[0]
        if prefix:
            options = ", ".join(self.names[i] for i in prefix)
            raise ValueError(f"player {name!r} is ambiguous: {options}")
        raise ValueError(f"no player named {name!r}")


def _number(text: str | None) -> float:
    if text is None:
        return math.nan
    cleaned = text.strip().replace("$", "").replace(",", "").rstrip("%")
    return float(cleaned) if cleaned else math.nan


def load_players(path: str | Path, stage_teams: tuple[str, ...]) -> PlayerPool:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if not rows:
        raise ValueError(f"{path}: no players")
    header = set(rows[0])
    missing = {"team", "player", "price"} - header
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}")

    names, team_names, team_idx, prices = [], [], [], []
    stats: dict[str, list[float]] = {s: [] for s in CARD_STATS}
    ranks, picked, estimated = [], [], []
    for line, row in enumerate(rows, start=2):
        name, team = row["player"].strip(), row["team"].strip()
        if team not in stage_teams:
            raise ValueError(f"{path}:{line}: team {team!r} is not in the stage file")
        price = _number(row["price"])
        if not price > 0:
            raise ValueError(f"{path}:{line}: {name} has no valid price")
        names.append(name)
        team_names.append(team)
        team_idx.append(stage_teams.index(team))
        prices.append(int(price))
        x = (price - 200_000) / 10_000
        card = {s: _number(row.get(s)) for s in CARD_STATS}
        if any(math.isnan(v) for v in card.values()):
            estimated.append(name)
        for s in CARD_STATS:
            base, slope = PRICE_FALLBACK[s]
            stats[s].append(card[s] if not math.isnan(card[s]) else base + slope * x)
        ranks.append(_number(row.get("hltv_rank")))
        share = _number(row.get("picked_pct"))
        picked.append(share / 100 if not math.isnan(share) else math.nan)

    if len(set(names)) != len(names):
        raise ValueError(f"{path}: a player is listed twice")
    counts = np.bincount(np.asarray(team_idx), minlength=len(stage_teams))
    short = [stage_teams[t] for t in range(len(stage_teams)) if counts[t] != 5]
    if short:
        raise ValueError(f"{path}: every team needs exactly 5 players ({', '.join(short)})")
    return PlayerPool(
        names=tuple(names),
        team_names=tuple(team_names),
        team=np.asarray(team_idx, dtype=np.intp),
        price=np.asarray(prices, dtype=np.int64),
        stats={s: np.asarray(v, dtype=np.float64) for s, v in stats.items()},
        hltv_rank=np.asarray(ranks, dtype=np.float64),
        picked=np.asarray(picked, dtype=np.float64),
        estimated=tuple(estimated),
    )
