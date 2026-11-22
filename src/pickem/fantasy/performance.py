"""Player performances, simulated map by map on top of the team outcomes.

Fitted on IEM Beijing 2026 (HLTV match pages, 30 matches, 740 player-maps; card = the player's
3-month HLTV card rating shown in fantasy game 671), by least squares, with the card split into
the team's average card and the player's deviation from it:
- map rating = 1.125 + 0.811 (card - team average card) + 0.293 (team average card - 1.051)
  - 0.098 won + 0.0439 (own - opponent rounds) + N(0, 0.313).
  The deviation coefficient is sharp (SE 0.13); the team-average one is not (SE 0.37) once the
  map result is known. Card averages also compare badly across tiers: a 3-month card earned
  against tier-2 opponents is inflated. So by default (``team_card_weight=0``) team quality
  enters only through the simulated results, which come from the odds, and the card says how a
  player ranks within his team. ``team_card_weight=1`` adds the IEM-fitted team term back.
- The teammates' leftover deviations share almost nothing beyond the map result (SD of a
  team-map's mean deviation 0.13 vs 0.137 expected from independence), so teammates are
  correlated through wins, losses and margins, and opponents mirror them.
- kills per round = 0.104 + 0.504 map rating + N(0, 0.095).
- deaths per round = 0.861 - 0.148 map rating - 0.103 won + N(0, 0.101).
The match rating is the round-weighted mean of the map ratings; rating points depend on it alone.

The other statistics are drawn around the player's card, scaled by the map's kills per round
relative to what the raw card rating implies ("form"), so card rates earned against weak
opposition shrink along with the rating. Rates marked POPULATION below are rough defaults,
not fitted (HLTV match pages do not show them); they shape boosters and Eco Friendly only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pickem.fantasy.outcomes import MapOutcomes
from pickem.fantasy.players import PlayerPool
from pickem.fantasy.rules import (
    BOOSTER_INDEX,
    LEADER_INDEX,
    N_BOOSTERS,
    N_ROLES,
    N_ROUNDS,
    STAT_ROLES,
    TEAM_LOSS,
    TEAM_WIN,
    leader_points,
    padded,
    rating_points,
)

R_INTERCEPT, R_WON, R_MARGIN, R_SD = 1.1246, -0.0981, 0.0439, 0.3133
R_DEV = 0.8107  # card minus the team's average card
R_TEAM, R_TEAM_MEAN = 0.2934, 1.051  # team average card, centred on IEM Beijing's mean
K_INTERCEPT, K_RATING, K_SD = 0.1044, 0.5044, 0.0954
D_INTERCEPT, D_RATING, D_WON, D_SD = 0.8611, -0.1482, -0.1025, 0.1009
SIDE_SD = 0.20  # side rating around the match rating; not fitted
HS_KAPPA = 40.0  # beta concentration of a map's headshot share; not fitted
SUPPORT_KAPPA = 40.0
# POPULATION defaults (per round unless noted):
ECO_RATE = 0.03  # SMG + shotgun kills
KAST_BASE, KAST_SLOPE, KAST_SD = 0.715, 0.17, 0.07
QUAD_SHARE = 0.04  # share of multi-kill rounds that are 4k+
CLUTCH_1VX_SHARE = 0.4  # share of clutches that are 1v2 or harder
PISTOL_BASE = 0.2  # chance to top one pistol round for an average teammate
ASSIST_RATE, FLASH_RATE = 0.15, 0.08  # damage and flash assists, scaled by support share
TRADE_SHARE = 0.2  # share of kills that are trades (Avenger) and of deaths traded (Bait)
KOBE_RATE, RAMBO_RATE = 0.004, 0.003
SAVE_P = 0.15  # survives a lost round
SUPPORT_MEAN = 21.4


@dataclass(frozen=True)
class PlayerSims:
    """Totals per sim and player, padding included (see ``pickem.fantasy.rules``)."""

    names: tuple[str, ...]
    rating: NDArray[np.int32]
    """(n, P) rating points."""
    team: NDArray[np.int32]
    """(n, P) team points."""
    roles: NDArray[np.int16]
    """(n, P, 12) role points if that role is held all stage (order: ``ROLE_NAMES``)."""
    boosts: NDArray[np.uint32]
    """(n, P, 5) bit b set if booster b would have succeeded that round."""
    played: NDArray[np.bool_]
    """(n, P, 5)."""
    role_rounds: NDArray[np.int8] | None = None
    """(n, P, 5, 12) role points per round (0 if not played); kept only on request."""
    maps: NDArray[np.int8] | None = None
    """(n, P, 5) maps the player's team played that round (0 if it did not play)."""

    @property
    def n_sims(self) -> int:
        return int(self.rating.shape[0])

    @property
    def base(self) -> NDArray[np.int32]:
        """(n, P) rating + team points."""
        result: NDArray[np.int32] = self.rating + self.team
        return result

    def booster_probabilities(self) -> NDArray[np.float64]:
        """(P, 5, 18) chance each booster succeeds for each player and round."""
        bits = (self.boosts[..., None] >> np.arange(N_BOOSTERS, dtype=np.uint32)) & 1
        result: NDArray[np.float64] = bits.mean(axis=0)
        return result


def _team_members(pool: PlayerPool) -> NDArray[np.intp]:
    """(16, 5) player indices per team."""
    order = np.argsort(pool.team, kind="stable")
    return order.reshape(-1, 5)


def _by_team(values: NDArray[np.float64], members: NDArray[np.intp]) -> NDArray[np.float64]:
    """(c, P, ...) -> (c, 16, 5, ...)."""
    return values[:, members]


def _from_team(
    values: NDArray[np.bool_], members: NDArray[np.intp], n_players: int
) -> NDArray[np.bool_]:
    out = np.empty((values.shape[0], n_players, *values.shape[3:]), dtype=np.bool_)
    out[:, members.ravel()] = values.reshape(values.shape[0], -1, *values.shape[3:])
    return out


def team_average_card(pool: PlayerPool) -> NDArray[np.float64]:
    """Each player's team's average card rating."""
    sums = np.bincount(pool.team, weights=pool.stats["rating"])
    counts = np.bincount(pool.team)
    result: NDArray[np.float64] = (sums / np.maximum(counts, 1))[pool.team]
    return result


def expected_card_level(pool: PlayerPool, team_card_weight: float = 0.0) -> NDArray[np.float64]:
    """Each player's map-rating baseline before the map result: R_INTERCEPT + R_DEV x (card -
    team average) + weight x R_TEAM x (team average - R_TEAM_MEAN)."""
    team_avg = team_average_card(pool)
    result: NDArray[np.float64] = (
        R_INTERCEPT
        + R_DEV * (pool.stats["rating"] - team_avg)
        + team_card_weight * R_TEAM * (team_avg - R_TEAM_MEAN)
    )
    return result


def _chunk(
    outcomes: MapOutcomes, pool: PlayerPool, rng: np.random.Generator, team_card_weight: float
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int16],
    NDArray[np.uint32],
    NDArray[np.bool_],
    NDArray[np.int8],
]:
    tp = pool.team
    n_players = pool.size
    stats = pool.stats
    played = outcomes.played[:, tp]  # (c, P, 5)
    won = outcomes.won[:, tp]
    best_of = outcomes.best_of[:, tp]
    map_won = outcomes.map_won[:, tp]  # (c, P, 5, 3)
    own = outcomes.own_rounds[:, tp].astype(np.float64)
    opp = outcomes.opp_rounds[:, tp].astype(np.float64)
    on = map_won >= 0
    rounds = own + opp
    shape = map_won.shape
    card = stats["rating"][None, :, None, None]
    won_map = (map_won == 1).astype(np.float64)
    level = expected_card_level(pool, team_card_weight)[None, :, None, None]

    rating = level + R_WON * won_map + R_MARGIN * (own - opp) + rng.normal(0.0, R_SD, shape)
    kpr = K_INTERCEPT + K_RATING * rating + rng.normal(0.0, K_SD, shape)
    kills = np.where(on, np.clip(np.rint(kpr * rounds), 0, None), 0).astype(np.int64)
    dpr = D_INTERCEPT + D_RATING * rating + D_WON * won_map + rng.normal(0.0, D_SD, shape)
    deaths = np.where(on, np.clip(np.rint(dpr * rounds), 0, rounds), 0).astype(np.int64)
    card_kpr = K_INTERCEPT + K_RATING * card
    form = np.clip(kpr / card_kpr, 0.2, 3.0)
    rounds_i = rounds.astype(np.int64)

    def per_player(stat: str, scale: float = 1.0) -> NDArray[np.float64]:
        return stats[stat][None, :, None, None] * scale

    awp_share = np.clip(per_player("awp_kpr") / card_kpr, 0.0, 1.0)
    awp = rng.binomial(kills, np.broadcast_to(awp_share, shape))
    hs_card = np.clip(per_player("hs_pct", 0.01), 0.01, 0.99)
    hs_p = rng.beta(HS_KAPPA * hs_card, HS_KAPPA * (1 - hs_card), shape)
    hs = rng.binomial(kills, hs_p)
    entries = rng.binomial(rounds_i, np.clip(per_player("entry_pct", 0.01) * form, 0.0, 1.0))
    multi = rng.binomial(rounds_i, np.clip(per_player("multikill_pct", 0.01) * form, 0.0, 1.0))
    sup_card = np.clip(per_player("support_pct", 0.01), 0.01, 0.99)
    support = rng.binomial(
        rounds_i, rng.beta(SUPPORT_KAPPA * sup_card, SUPPORT_KAPPA * (1 - sup_card), shape)
    )
    eco = rng.poisson(ECO_RATE * rounds * form)

    # Match aggregates.
    total_rounds = np.maximum(rounds.sum(axis=-1), 1.0)
    match_rating = (rating * rounds).sum(axis=-1) / total_rounds
    total_kills = kills.sum(axis=-1)
    match: dict[str, NDArray[np.float64]] = {
        "rating": match_rating,
        "awp_kpr": awp.sum(axis=-1) / total_rounds,
        "hs_pct": 100.0 * hs.sum(axis=-1) / np.maximum(total_kills, 1),
        "support_pct": 100.0 * support.sum(axis=-1) / total_rounds,
        "multikill_kpr": multi.sum(axis=-1) / total_rounds,
        "entry_kpr": entries.sum(axis=-1) / total_rounds,
        "eco_kpr": eco.sum(axis=-1) / total_rounds,
        "dpr": deaths.sum(axis=-1) / total_rounds,
    }
    match_shape = match_rating.shape
    for side, stat in (("t_rating", "t_rating"), ("ct_rating", "ct_rating")):
        offset = (stats[stat] - stats["rating"])[None, :, None]
        match[side] = match_rating + offset + rng.normal(0.0, SIDE_SD, match_shape)

    role_round = np.zeros((*match_shape, N_ROLES), dtype=np.int8)
    for k, role in enumerate(STAT_ROLES):
        role_round[..., k] = role.points(match[role.stat])
    maps_lost = (map_won == 0).sum(axis=-1)
    ot_bo1 = (best_of == 1) & (rounds[..., 0] > 24)
    role_round[..., LEADER_INDEX] = leader_points(won, maps_lost, ot_bo1)

    rating_round = rating_points(match_rating)
    team_round = np.where(won, TEAM_WIN, TEAM_LOSS).astype(np.int8)
    advanced = outcomes.advanced[:, tp]
    rating_total = padded(rating_round, played, advanced)
    team_total = padded(team_round, played, advanced)
    roles_total = np.stack(
        [padded(role_round[..., k], played, advanced) for k in range(N_ROLES)], axis=-1
    ).astype(np.int16)

    # Boosters, judged per map; a round succeeds if any of its maps does.
    members = _team_members(pool)
    noise = rng.random(shape) * 1e-3
    board = _by_team(kills - 0.01 * deaths + noise, members)
    hs_share = _by_team(np.where(kills > 0, hs / np.maximum(kills, 1), 0.0) + noise, members)
    team_on = _by_team(on.astype(np.float64), members) > 0
    top = _from_team((board == board.max(axis=2, keepdims=True)) & team_on, members, n_players)
    bottom = _from_team(
        (board == np.where(team_on, board, np.inf).min(axis=2, keepdims=True)) & team_on,
        members,
        n_players,
    )
    aim = _from_team(
        (hs_share == hs_share.max(axis=2, keepdims=True)) & team_on, members, n_players
    )
    rel = (stats["rating"] / team_average_card(pool))[None, :, None, None]
    kast = KAST_BASE + KAST_SLOPE * (rating - 1.0) + rng.normal(0.0, KAST_SD, shape)
    support_factor = per_player("support_pct") / SUPPORT_MEAN
    u = rng.random((N_BOOSTERS, *shape))
    success = {
        "Hellcase": on,
        "Top of scoreboard": top,
        "Bottom of scoreboard": bottom,
        "Carry": kills >= 20,
        "Aim bot": aim,
        "Mister/Miss consistent": kast > 0.85,
        "Cannon fodder": rounds - deaths <= 5,
        "Best pistol round": u[0] < 1.0 - (1.0 - PISTOL_BASE * rel) ** 2,
        "Quad": u[1]
        < 1.0 - (1.0 - QUAD_SHARE * per_player("multikill_pct", 0.01) * form) ** rounds,
        "Clutch": u[2] < 1.0 - (1.0 - CLUTCH_1VX_SHARE * per_player("clutch_pct", 0.01)) ** rounds,
        "Assist": rng.poisson(ASSIST_RATE * rounds * support_factor) >= 4,
        "Avenger": rng.poisson(TRADE_SHARE * kills) >= 4,
        "Bait": rng.poisson(TRADE_SHARE * deaths) >= 4,
        "Flash": rng.poisson(FLASH_RATE * rounds * support_factor) >= 4,
        "Kobe": u[3] < 1.0 - np.exp(-KOBE_RATE * rounds),
        "Farmer": eco >= 3,
        "Saver": rng.binomial(opp.astype(np.int64), SAVE_P) >= 3,
        "Rambo": u[4] < 1.0 - np.exp(-RAMBO_RATE * rounds),
    }
    bits = np.zeros(match_shape, dtype=np.uint32)
    for name, ok in success.items():
        hit = (np.asarray(ok) & on).any(axis=-1)
        bits |= hit.astype(np.uint32) << np.uint32(BOOSTER_INDEX[name])
    role_round = np.where(played[..., None], role_round, 0).astype(np.int8)
    return rating_total, team_total, roles_total, bits, played, role_round


def simulate_players(
    outcomes: MapOutcomes,
    pool: PlayerPool,
    rng: np.random.Generator,
    chunk: int = 1000,
    *,
    keep_rounds: bool = False,
    team_card_weight: float = 0.0,
) -> PlayerSims:
    """``keep_rounds`` also keeps role points per round (needed to plan a single round).
    ``team_card_weight`` scales how much a team's average card counts beyond the simulated
    results (0: not at all, the default; 1: as fitted on IEM Beijing)."""
    n = outcomes.n_sims
    if np.max(pool.team) >= len(outcomes.teams):
        raise ValueError("player pool refers to a team the stage does not have")
    parts = [
        _chunk(outcomes.take(slice(s, min(s + chunk, n))), pool, rng, team_card_weight)
        for s in range(0, n, chunk)
    ]
    rating, team, roles, boosts, played, role_rounds = (
        np.concatenate(x) for x in zip(*parts, strict=True)
    )
    assert played.shape[-1] == N_ROUNDS
    maps = (outcomes.map_won[:, pool.team] >= 0).sum(axis=-1).astype(np.int8)
    return PlayerSims(
        pool.names,
        rating,
        team,
        roles,
        boosts,
        played,
        role_rounds if keep_rounds else None,
        maps,
    )
