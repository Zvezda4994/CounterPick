"""Replay a finished Swiss stage through the rules engine.

The results are fed to ``simulate_from_uniforms`` as certainties: P(winner beats loser) = 1 for
every match that was played. If the engine pairs every round exactly as the results say, the
replay reproduces the stage and yields records, Difficulty and final mid-stage seeds from the same
code that runs the simulations. The first round that the engine would pair differently is
reported, which catches a mistyped winner or pairing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pickem.swiss.rules import MAX_MATCHES_PER_ROUND, N_ROUNDS, N_TEAMS, round_layouts
from pickem.swiss.simulate import StageInput, simulate_from_uniforms

MATCHES_PER_ROUND = tuple(layout.n_matches for layout in round_layouts())


@dataclass(frozen=True)
class MatchResult:
    team1: str
    team2: str
    winner: str
    maps: int | None = None

    def __post_init__(self) -> None:
        if self.team1 == self.team2:
            raise ValueError(f"{self.team1} cannot play itself")
        if self.winner not in (self.team1, self.team2):
            raise ValueError(f"winner {self.winner!r} did not play in {self.team1} v {self.team2}")

    @property
    def loser(self) -> str:
        return self.team2 if self.winner == self.team1 else self.team1

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.team1, self.team2))


@dataclass(frozen=True)
class StageReplay:
    """A finished stage, per team in the order given to ``replay_stage``."""

    teams: tuple[str, ...]
    seeds: tuple[int, ...]
    wins: tuple[int, ...]
    losses: tuple[int, ...]
    difficulty: tuple[int, ...]
    final_seed: tuple[int, ...]
    rounds: tuple[tuple[MatchResult, ...], ...]

    @property
    def standings(self) -> tuple[str, ...]:
        """Teams in final mid-stage seed order (record, Difficulty, initial seed)."""
        return tuple(sorted(self.teams, key=lambda t: self.final_seed[self.teams.index(t)]))

    @property
    def advancing(self) -> tuple[str, ...]:
        return self.standings[: N_TEAMS // 2]

    def record(self, team: str) -> str:
        i = self.teams.index(team)
        return f"{self.wins[i]}-{self.losses[i]}"


def check_rounds(rounds: Sequence[Sequence[MatchResult]], teams: Sequence[str]) -> None:
    """Complete stage, known teams, nobody twice in a round."""
    if len(rounds) != N_ROUNDS:
        raise ValueError(f"expected {N_ROUNDS} rounds of results, got {len(rounds)}")
    known = set(teams)
    # Name problems first: they explain a wrong count better than the count does.
    for r, matches in enumerate(rounds, start=1):
        seen: set[str] = set()
        for m in matches:
            for team in (m.team1, m.team2):
                if team not in known:
                    raise ValueError(f"round {r}: unknown team {team!r}")
                if team in seen:
                    raise ValueError(f"round {r}: {team} plays twice")
                seen.add(team)
    for r, (matches, expected) in enumerate(zip(rounds, MATCHES_PER_ROUND, strict=True), start=1):
        if len(matches) != expected:
            raise ValueError(f"round {r} needs {expected} results, got {len(matches)}")


def replay_stage(
    teams: Sequence[str], seeds: Sequence[int], rounds: Sequence[Sequence[MatchResult]]
) -> StageReplay:
    teams = tuple(teams)
    check_rounds(rounds, teams)
    index = {t: i for i, t in enumerate(teams)}

    p = np.full((N_TEAMS, N_TEAMS), 0.5)
    for matches in rounds:
        for m in matches:
            w, lo = index[m.winner], index[m.loser]
            if p[w, lo] == 0.0:
                raise ValueError(f"{m.winner} and {m.loser} met twice with different winners")
            p[w, lo], p[lo, w] = 1.0, 0.0
    stage = StageInput(teams=teams, seeds=tuple(seeds), p_bo1=p, p_bo3=p)
    uniforms = np.full((1, N_ROUNDS, MAX_MATCHES_PER_ROUND), 0.5)
    result = simulate_from_uniforms(stage, uniforms, record_matches=True)
    log = result.matches
    assert log is not None

    for r, matches in enumerate(rounds):
        slots = int((log.best_of[r] > 0).sum())
        engine = {
            frozenset((teams[h], teams[lo]))
            for h, lo in zip(log.high[0, r, :slots], log.low[0, r, :slots], strict=True)
        }
        given = {match.pair for match in matches}
        if engine != given:
            expected = "; ".join(" v ".join(sorted(pair)) for pair in engine - given)
            found = "; ".join(" v ".join(sorted(pair)) for pair in given - engine)
            raise ValueError(
                f"round {r + 1} does not follow the Swiss rules from the earlier results: the "
                f"rules pair {expected}, but the results have {found}. Check this round's "
                "pairings and the earlier winners."
            )

    return StageReplay(
        teams=teams,
        seeds=tuple(int(s) for s in seeds),
        wins=tuple(int(v) for v in result.wins[0]),
        losses=tuple(int(v) for v in result.losses[0]),
        difficulty=tuple(int(v) for v in result.difficulty[0]),
        final_seed=tuple(int(v) for v in result.final_seed[0]),
        rounds=tuple(tuple(r) for r in rounds),
    )
