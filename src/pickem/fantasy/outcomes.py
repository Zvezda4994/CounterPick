"""Team outcomes for fantasy scoring: every simulated Swiss match, expanded to maps and rounds.

The Swiss simulator decides each match (``record_matches=True``). Maps are then drawn
conditionally on the match winner, which leaves the match-level distribution exactly as
simulated: with map probability p for the winner, a BO3 ends 2-0 with probability
p^2 / (p^2 (3 - 2p)) = 1 / (3 - 2p), and a 2-1 win drops map 1 or map 2 equally often.

Round scores (MR12, 13 to win; overtime is first to 4 of 6, so 16-12..14 or 19-15..17) use
IEM Beijing 2026 (74 maps): 12% of maps went to overtime, and a regulation loser took 7.4 rounds
on average (SD 2.4). Overtime is made likelier for even maps: P = OT_BASE * 4p(1 - p).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pickem.data.stage_file import StageFile
from pickem.fantasy.rules import N_ROUNDS
from pickem.probs.strength import stage_from_strengths
from pickem.swiss import StageInput, simulate_swiss

MAX_MAPS = 3
OT_BASE = 0.125
SECOND_OT = 0.15
LOSER_MEAN, LOSER_SD = 7.4, 2.4


@dataclass(frozen=True)
class MapOutcomes:
    """Axes: (sim, team, round[, map]); team order follows the stage file."""

    teams: tuple[str, ...]
    played: NDArray[np.bool_]
    """(n, 16, 5) the team played a match that round."""
    won: NDArray[np.bool_]
    """(n, 16, 5) and won it."""
    best_of: NDArray[np.int8]
    """(n, 16, 5) 1 or 3 (0 if not played)."""
    map_won: NDArray[np.int8]
    """(n, 16, 5, 3) 1 won, 0 lost, -1 not played."""
    own_rounds: NDArray[np.int16]
    """(n, 16, 5, 3) rounds the team won on the map (0 if not played)."""
    opp_rounds: NDArray[np.int16]
    """(n, 16, 5, 3)."""
    wins: NDArray[np.int8]
    """(n, 16) final wins."""
    losses: NDArray[np.int8]

    @property
    def n_sims(self) -> int:
        return int(self.played.shape[0])

    @property
    def advanced(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self.wins == 3
        return result

    @property
    def map_played(self) -> NDArray[np.bool_]:
        result: NDArray[np.bool_] = self.map_won >= 0
        return result

    def take(self, sims: slice) -> MapOutcomes:
        return MapOutcomes(
            self.teams,
            self.played[sims],
            self.won[sims],
            self.best_of[sims],
            self.map_won[sims],
            self.own_rounds[sims],
            self.opp_rounds[sims],
            self.wins[sims],
            self.losses[sims],
        )


def _draws(
    stage_file: StageFile, n: int, rng: np.random.Generator, spread: float, per_draw: int
) -> Iterator[tuple[StageInput, int]]:
    model = stage_file.strength_model
    if model is None or spread == 0:
        yield stage_file.stage, n
        return
    factor = model.factor()
    for start in range(0, n, per_draw):
        strengths = model.sample(rng, spread, factor)
        stage = stage_file.stage
        yield stage_from_strengths(stage.teams, stage.seeds, strengths), min(per_draw, n - start)


def _fix(stage: StageInput, fixed: Sequence[tuple[int, int]]) -> StageInput:
    if not fixed:
        return stage
    p1, p3 = stage.p_bo1.copy(), stage.p_bo3.copy()
    for winner, loser in fixed:
        for p in (p1, p3):
            p[winner, loser], p[loser, winner] = 1.0, 0.0
    return StageInput(stage.teams, stage.seeds, p1, p3)


def _round_scores(
    rng: np.random.Generator, p_map: NDArray[np.float64], shape: tuple[int, ...]
) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    """(map winner's rounds, map loser's rounds) for maps whose winner had map probability
    ``p_map`` (broadcast to ``shape``)."""
    closeness = 4.0 * p_map * (1.0 - p_map)
    overtime = rng.random(shape) < OT_BASE * closeness
    loser = np.clip(np.rint(rng.normal(LOSER_MEAN, LOSER_SD, shape)), 0, 11)
    extra = (rng.random(shape) < SECOND_OT).astype(np.int16)  # 0: one overtime, 1: two
    ot_winner = 16 + 3 * extra
    ot_loser = 12 + 3 * extra + rng.integers(0, 3, shape)
    win = np.where(overtime, ot_winner, 13).astype(np.int16)
    lose = np.where(overtime, ot_loser, loser).astype(np.int16)
    return win, lose


def _expand(stage: StageInput, n: int, rng: np.random.Generator) -> MapOutcomes:
    result = simulate_swiss(stage, n, rng, record_matches=True)
    log = result.matches
    assert log is not None
    t = len(stage.teams)
    played = np.zeros((n, t, N_ROUNDS), dtype=np.bool_)
    won = np.zeros((n, t, N_ROUNDS), dtype=np.bool_)
    best_of = np.zeros((n, t, N_ROUNDS), dtype=np.int8)
    map_won = np.full((n, t, N_ROUNDS, MAX_MAPS), -1, dtype=np.int8)
    own = np.zeros((n, t, N_ROUNDS, MAX_MAPS), dtype=np.int16)
    opp = np.zeros((n, t, N_ROUNDS, MAX_MAPS), dtype=np.int16)
    sims = np.arange(n)
    for r in range(N_ROUNDS):
        for j in range(log.best_of.shape[1]):
            bo = int(log.best_of[r, j])
            if bo == 0:
                continue
            high = log.high[:, r, j].astype(np.intp)
            low = log.low[:, r, j].astype(np.intp)
            high_won = log.high_won[:, r, j]
            winner = np.where(high_won, high, low)
            loser = np.where(high_won, low, high)
            p = stage.p_bo1[winner, loser]  # winner's map probability

            seq = np.full((n, MAX_MAPS), -1, dtype=np.int8)  # from the match winner's side
            seq[:, 0] = 1
            if bo == 3:
                sweep = rng.random(n) < 1.0 / (3.0 - 2.0 * p)
                drop_first = rng.random(n) < 0.5
                seq[:, 0] = np.where(sweep | ~drop_first, 1, 0)
                seq[:, 1] = np.where(sweep | drop_first, 1, 0)
                seq[:, 2] = np.where(sweep, -1, 1)
            elif bo != 1:
                raise ValueError(f"unexpected best-of-{bo} in the Swiss stage")
            # Map winner's probability: p on maps the match winner took, 1 - p on the others.
            p_map = np.where(seq == 1, p[:, None], 1.0 - p[:, None])
            win_r, lose_r = _round_scores(rng, p_map, (n, MAX_MAPS))
            on = seq >= 0
            w_own = np.where(on, np.where(seq == 1, win_r, lose_r), 0).astype(np.int16)
            w_opp = np.where(on, np.where(seq == 1, lose_r, win_r), 0).astype(np.int16)

            for team, is_winner in ((winner, True), (loser, False)):
                played[sims, team, r] = True
                won[sims, team, r] = is_winner
                best_of[sims, team, r] = bo
                if is_winner:
                    map_won[sims, team, r] = seq
                    own[sims, team, r] = w_own
                    opp[sims, team, r] = w_opp
                else:
                    map_won[sims, team, r] = np.where(on, 1 - seq, -1)
                    own[sims, team, r] = w_opp
                    opp[sims, team, r] = w_own
    return MapOutcomes(
        stage.teams, played, won, best_of, map_won, own, opp, result.wins, result.losses
    )


def simulate_map_outcomes(
    stage_file: StageFile,
    n_sims: int,
    rng: np.random.Generator,
    *,
    spread: float | None = None,
    sims_per_draw: int | None = None,
    fixed: Sequence[tuple[int, int]] = (),
) -> MapOutcomes:
    """``n_sims`` stages; with a strength model and spread > 0, strengths are redrawn every
    ``sims_per_draw`` sims exactly as in ``simulate_with_uncertainty``.

    ``fixed`` holds (winner, loser) team indices of matches already played: those pairs always
    end that way, so the Swiss pairings replay reality up to the current round (the same device
    as ``pickem.optimizer.analysis.conditional_outcomes``)."""
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    model = stage_file.strength_model
    spread = (model.spread if model else 0.0) if spread is None else spread
    per = (model.sims_per_draw if model else 1000) if sims_per_draw is None else sims_per_draw
    if spread < 0 or per < 1:
        raise ValueError("spread must be 0 or more and sims_per_draw at least 1")
    parts = [
        _expand(_fix(stage, fixed), k, rng)
        for stage, k in _draws(stage_file, n_sims, rng, spread, per)
    ]
    if len(parts) == 1:
        return parts[0]
    return MapOutcomes(
        parts[0].teams,
        *(
            np.concatenate([getattr(p, name) for p in parts])
            for name in (
                "played",
                "won",
                "best_of",
                "map_won",
                "own_rounds",
                "opp_rounds",
                "wins",
                "losses",
            )
        ),
    )
