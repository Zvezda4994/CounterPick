"""Vectorized Monte Carlo simulation of one 16-team Major Swiss stage.

All sims advance in lockstep: Python loops run only over rounds, record groups and candidate
pairings, never over simulations. Internally teams are indexed by ``seed - 1``; results are
mapped back to the caller's team order at the end.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pickem.swiss.rules import (
    LOSSES_TO_ELIMINATE,
    MAX_MATCHES_PER_ROUND,
    N_ROUNDS,
    N_TEAMS,
    WINS_TO_ADVANCE,
    Pairing,
    round_layouts,
)

IntArray = NDArray[np.intp]
Int8Array = NDArray[np.int8]

_PROB_ATOL = 1e-9
_SEED_INDEX = np.arange(N_TEAMS, dtype=np.int32)


def _as_prob_matrix(name: str, value: ArrayLike) -> NDArray[np.float64]:
    p = np.array(value, dtype=np.float64)
    if p.shape != (N_TEAMS, N_TEAMS):
        raise ValueError(f"{name} must have shape ({N_TEAMS}, {N_TEAMS}), got {p.shape}")
    off_diagonal = ~np.eye(N_TEAMS, dtype=bool)
    values = p[off_diagonal]
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{name} off-diagonal entries must be probabilities in [0, 1]")
    if not np.allclose((p + p.T)[off_diagonal], 1.0, rtol=0.0, atol=_PROB_ATOL):
        raise ValueError(f"{name}[i, j] + {name}[j, i] must equal 1 for every i != j")
    p.setflags(write=False)
    return p


@dataclass(frozen=True)
class StageInput:
    """One Swiss stage: 16 teams, their stage seeds and head-to-head win probabilities.

    ``p_bo1[i, j]`` is the probability that ``teams[i]`` beats ``teams[j]`` in a best-of-one,
    ``p_bo3[i, j]`` the same for a best-of-three. Diagonals are ignored.
    """

    teams: tuple[str, ...]
    seeds: tuple[int, ...]
    """Stage seed of each team (a permutation of 1..16); 1 is the highest seed."""
    p_bo1: NDArray[np.float64]
    p_bo3: NDArray[np.float64]

    def __post_init__(self) -> None:
        teams = tuple(self.teams)
        seeds = tuple(int(s) for s in self.seeds)
        if len(teams) != N_TEAMS or len(set(teams)) != N_TEAMS:
            raise ValueError(f"need {N_TEAMS} distinct team names, got {len(set(teams))}")
        if sorted(seeds) != list(range(1, N_TEAMS + 1)):
            raise ValueError(f"seeds must be a permutation of 1..{N_TEAMS}")
        object.__setattr__(self, "teams", teams)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "p_bo1", _as_prob_matrix("p_bo1", self.p_bo1))
        object.__setattr__(self, "p_bo3", _as_prob_matrix("p_bo3", self.p_bo3))

    @property
    def by_seed(self) -> IntArray:
        """Team index of seed 1, seed 2, ..., seed 16."""
        return np.argsort(np.asarray(self.seeds))


@dataclass(frozen=True)
class MatchLog:
    """Every match of every simulated stage.

    Slot ``j`` of round ``r`` is that round's ``j``-th match ordered by the higher seed's
    standing going into the round. Team indices refer to ``StageInput.teams``; slots unused in
    rounds 4-5 hold -1.
    """

    high: Int8Array
    """(n, 5, 8) higher seed going into the match."""
    low: Int8Array
    """(n, 5, 8) lower seed going into the match."""
    high_won: NDArray[np.bool_]
    """(n, 5, 8)."""
    best_of: Int8Array
    """(5, 8) 1 or 3 per slot, 0 for unused slots. The same in every sim."""


@dataclass(frozen=True)
class SwissResult:
    """Final state of ``n`` simulated stages. Team axes follow ``teams`` (the input order)."""

    teams: tuple[str, ...]
    wins: Int8Array
    """(n, 16) in 0..3."""
    losses: Int8Array
    """(n, 16) in 0..3."""
    difficulty: Int8Array
    """(n, 16) final Difficulty Score (Buchholz)."""
    final_seed: Int8Array
    """(n, 16) 1..16 by record, then difficulty, then initial seed; seeds 1-8 advanced."""
    rematches: Int8Array
    """(n,) rematches forced because no rematch-free pairing existed."""
    matches: MatchLog | None = None

    @property
    def n_sims(self) -> int:
        return int(self.wins.shape[0])


def simulate_swiss(
    stage: StageInput,
    n_sims: int,
    rng: np.random.Generator,
    *,
    record_matches: bool = False,
    batch_size: int = 100_000,
) -> SwissResult:
    """Simulate ``n_sims`` independent runs of the stage, in batches of ``batch_size``."""
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    chunks = []
    for start in range(0, n_sims, batch_size):
        n = min(batch_size, n_sims - start)
        u = rng.random((n, N_ROUNDS, MAX_MATCHES_PER_ROUND))
        chunks.append(simulate_from_uniforms(stage, u, record_matches=record_matches))
    return concat_results(chunks)


def simulate_from_uniforms(
    stage: StageInput, u: ArrayLike, *, record_matches: bool = False
) -> SwissResult:
    """Run the stage on caller-supplied uniforms ``u`` of shape (n, 5, 8).

    In sim ``i``, match slot ``j`` of round ``r`` goes to the higher seed iff
    ``u[i, r, j] < p`` (``p`` from ``p_bo1`` or ``p_bo3`` according to the stakes).
    """
    uniforms = np.asarray(u, dtype=np.float64)
    if uniforms.ndim != 3 or uniforms.shape[1:] != (N_ROUNDS, MAX_MATCHES_PER_ROUND):
        raise ValueError(f"u must have shape (n, {N_ROUNDS}, {MAX_MATCHES_PER_ROUND})")
    by_seed = stage.by_seed
    seed_grid = np.ix_(by_seed, by_seed)
    raw = _simulate_seed_space(
        uniforms, stage.p_bo1[seed_grid], stage.p_bo3[seed_grid], record_matches
    )

    column = np.asarray(stage.seeds, dtype=np.intp) - 1  # internal column of each input team
    matches = None
    if raw.high is not None and raw.low is not None and raw.high_won is not None:
        to_input = np.append(by_seed, -1).astype(np.int8)  # internal id -> input id; -1 -> -1
        matches = MatchLog(
            high=to_input[raw.high],
            low=to_input[raw.low],
            high_won=raw.high_won,
            best_of=_best_of_table(),
        )
    return SwissResult(
        teams=stage.teams,
        wins=raw.wins[:, column],
        losses=raw.losses[:, column],
        difficulty=raw.difficulty[:, column],
        final_seed=raw.final_seed[:, column],
        rematches=raw.rematches,
        matches=matches,
    )


@dataclass(frozen=True)
class _Raw:
    wins: Int8Array
    losses: Int8Array
    difficulty: Int8Array
    final_seed: Int8Array
    rematches: Int8Array
    high: Int8Array | None
    low: Int8Array | None
    high_won: NDArray[np.bool_] | None


def _simulate_seed_space(
    u: NDArray[np.float64],
    p_bo1: NDArray[np.float64],
    p_bo3: NDArray[np.float64],
    record_matches: bool,
) -> _Raw:
    n = u.shape[0]
    rows = np.arange(n)[:, None]
    wins = np.zeros((n, N_TEAMS), dtype=np.int8)
    losses = np.zeros((n, N_TEAMS), dtype=np.int8)
    opponents = np.full((n, N_TEAMS, N_ROUNDS), -1, dtype=np.int8)
    played = np.zeros((n, N_TEAMS), dtype=np.int32)  # bit j set: already played seed j + 1
    rematches = np.zeros(n, dtype=np.int8)
    shape = (n, N_ROUNDS, MAX_MATCHES_PER_ROUND)
    high_log = np.full(shape, -1, dtype=np.int8) if record_matches else None
    low_log = np.full(shape, -1, dtype=np.int8) if record_matches else None
    won_log = np.zeros(shape, dtype=np.bool_) if record_matches else None

    for layout in round_layouts():
        r = layout.round_index
        standings = _standings(wins, losses, opponents)
        highs, lows = [], []
        for group in layout.groups:
            members = standings[:, group.start : group.start + group.size]
            pairing, forced = _pick_pairing(members, group.candidates, played)
            highs.append(np.take_along_axis(members, pairing[:, :, 0], axis=1))
            lows.append(np.take_along_axis(members, pairing[:, :, 1], axis=1))
            rematches += forced
        high = np.concatenate(highs, axis=1)
        low = np.concatenate(lows, axis=1)
        m = layout.n_matches

        bo3 = np.asarray(layout.best_of_by_slot) == 3
        p_high = np.where(bo3, p_bo3[high, low], p_bo1[high, low])
        high_won = u[:, r, :m] < p_high
        winner = np.where(high_won, high, low)
        loser = np.where(high_won, low, high)

        # Each team plays at most once per round, so fancy-indexed updates never collide.
        wins[rows, winner] += 1
        losses[rows, loser] += 1
        opponents[rows, high, r] = low
        opponents[rows, low, r] = high
        played[rows, high] |= np.left_shift(np.int32(1), low.astype(np.int32))
        played[rows, low] |= np.left_shift(np.int32(1), high.astype(np.int32))

        if high_log is not None and low_log is not None and won_log is not None:
            high_log[:, r, :m] = high
            low_log[:, r, :m] = low
            won_log[:, r, :m] = high_won

    difficulty = _difficulty(wins, losses, opponents)
    standings = _standings(wins, losses, opponents)
    final_seed = np.empty((n, N_TEAMS), dtype=np.int8)
    final_seed[rows, standings] = np.arange(1, N_TEAMS + 1, dtype=np.int8)
    return _Raw(
        wins=wins,
        losses=losses,
        difficulty=difficulty,
        final_seed=final_seed,
        rematches=rematches,
        high=high_log,
        low=low_log,
        high_won=won_log,
    )


def _difficulty(wins: Int8Array, losses: Int8Array, opponents: Int8Array) -> Int8Array:
    """L262-L272: sum of current (wins - losses) over every opponent faced so far."""
    n = wins.shape[0]
    net = wins - losses
    faced = opponents >= 0
    index = np.where(faced, opponents, 0).reshape(n, -1).astype(np.intp)
    values = np.take_along_axis(net, index, axis=1).reshape(opponents.shape)
    return np.where(faced, values, 0).sum(axis=2).astype(np.int8)


def _standings(wins: Int8Array, losses: Int8Array, opponents: Int8Array) -> IntArray:
    """L256-L260: teams ordered by record, then difficulty (desc), then initial seed (asc).

    One integer key per team: record rank (wins desc, losses asc) in the high bits, then
    15 - difficulty (difficulty is within [-15, 15]), then the seed index.
    """
    difficulty = _difficulty(wins, losses, opponents).astype(np.int32)
    record_rank = (WINS_TO_ADVANCE - wins.astype(np.int32)) * (LOSSES_TO_ELIMINATE + 1) + losses
    key = (record_rank * 32 + (15 - difficulty)) * N_TEAMS + _SEED_INDEX
    return np.argsort(key, axis=1)


@cache
def _candidate_array(candidates: tuple[Pairing, ...]) -> IntArray:
    return np.asarray(candidates, dtype=np.intp)  # (n_candidates, pairs, 2)


def _rematch_count(
    masks: NDArray[np.int32], members: IntArray, pairing: IntArray
) -> NDArray[np.int64]:
    """Rematches in ``pairing`` (local ranks) for each row of ``members``."""
    high_masks = masks[:, pairing[:, 0]]
    low_ids = members[:, pairing[:, 1]]
    counts: NDArray[np.int64] = ((high_masks >> low_ids) & 1).sum(axis=1)
    return counts


def _pick_pairing(
    members: IntArray, candidates: tuple[Pairing, ...], played: NDArray[np.int32]
) -> tuple[IntArray, Int8Array]:
    """Top-priority candidate without a rematch in each sim (L169-L170).

    If every candidate has a rematch (the rulebook only forbids them "if possible"), take the
    one with the fewest rematches, top-most on ties, and report how many were forced.
    """
    cands = _candidate_array(candidates)
    n = members.shape[0]
    masks = np.take_along_axis(played, members, axis=1)
    choice = np.full(n, -1, dtype=np.intp)
    todo = np.arange(n)
    for c in range(cands.shape[0]):
        clash = _rematch_count(masks[todo], members[todo], cands[c]) > 0
        choice[todo[~clash]] = c
        todo = todo[clash]
        if todo.size == 0:
            break

    forced = np.zeros(n, dtype=np.int8)
    if todo.size:
        counts = np.stack(
            [_rematch_count(masks[todo], members[todo], cands[c]) for c in range(len(cands))],
            axis=1,
        )
        best = counts.argmin(axis=1)
        choice[todo] = best
        forced[todo] = counts[np.arange(todo.size), best]
    return cands[choice], forced


@cache
def _best_of_table() -> Int8Array:
    table = np.zeros((N_ROUNDS, MAX_MATCHES_PER_ROUND), dtype=np.int8)
    for layout in round_layouts():
        slots = layout.best_of_by_slot
        table[layout.round_index, : len(slots)] = slots
    table.setflags(write=False)
    return table


def concat_results(chunks: list[SwissResult]) -> SwissResult:
    """Stack independent runs of the same 16 teams into one result."""
    if not chunks:
        raise ValueError("nothing to concatenate")
    if any(c.teams != chunks[0].teams for c in chunks):
        raise ValueError("results must cover the same teams in the same order")
    if len(chunks) == 1:
        return chunks[0]

    def cat(name: str) -> Int8Array:
        stacked: Int8Array = np.concatenate([getattr(c, name) for c in chunks], axis=0)
        return stacked

    matches = None
    logs = [c.matches for c in chunks]
    if all(log is not None for log in logs):
        present = [log for log in logs if log is not None]
        matches = MatchLog(
            high=np.concatenate([log.high for log in present], axis=0),
            low=np.concatenate([log.low for log in present], axis=0),
            high_won=np.concatenate([log.high_won for log in present], axis=0),
            best_of=present[0].best_of,
        )
    return SwissResult(
        teams=chunks[0].teams,
        wins=cat("wins"),
        losses=cat("losses"),
        difficulty=cat("difficulty"),
        final_seed=cat("final_seed"),
        rematches=cat("rematches"),
        matches=matches,
    )
