"""Scalar reference implementation of the Swiss rules, written for clarity, not speed.

It shares no code with ``pickem.swiss``: standings come from Python tuple sorting, rounds 2-3
are paired by recursive search, and the round 4-5 table is parsed from its own copy written in
the rulebook's notation. Teams are seed indices 0..15 (0 = seed 1).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

TABLE_TEXT = """
1v6 2v5 3v4
1v6 2v4 3v5
1v5 2v6 3v4
1v5 2v4 3v6
1v4 2v6 3v5
1v4 2v5 3v6
1v6 2v3 4v5
1v5 2v3 4v6
1v3 2v6 4v5
1v3 2v5 4v6
1v4 2v3 5v6
1v3 2v4 5v6
1v2 3v6 4v5
1v2 3v5 4v6
1v2 3v4 5v6
"""
TABLE = [
    [tuple(int(x) - 1 for x in pair.split("v")) for pair in line.split()]
    for line in TABLE_TEXT.strip().splitlines()
]

Pair = tuple[int, int]


@dataclass(frozen=True)
class Match:
    round: int
    high: int
    low: int
    high_won: bool
    best_of: int


@dataclass
class Stage:
    wins: list[int] = field(default_factory=lambda: [0] * 16)
    losses: list[int] = field(default_factory=lambda: [0] * 16)
    opponents: list[list[int]] = field(default_factory=lambda: [[] for _ in range(16)])
    matches: list[Match] = field(default_factory=list)
    rematches: int = 0

    def difficulty(self, team: int) -> int:
        return sum(self.wins[o] - self.losses[o] for o in self.opponents[team])

    def standings(self) -> list[int]:
        return sorted(
            range(16),
            key=lambda t: (-self.wins[t], self.losses[t], -self.difficulty(t), t),
        )

    def final_seed(self) -> list[int]:
        seed = [0] * 16
        for position, team in enumerate(self.standings()):
            seed[team] = position + 1
        return seed


def _rematches(pairs: Sequence[Pair], stage: Stage) -> int:
    return sum(b in stage.opponents[a] for a, b in pairs)


def _search(group: list[int], stage: Stage) -> list[Pair] | None:
    """Highest seed takes the lowest available non-rematch, with backtracking."""
    if not group:
        return []
    top, rest = group[0], group[1:]
    for opponent in reversed(rest):
        if opponent in stage.opponents[top]:
            continue
        tail = _search([t for t in rest if t != opponent], stage)
        if tail is not None:
            return [(top, opponent), *tail]
    return None


def _all_high_low(group: list[int]) -> Iterator[list[Pair]]:
    if not group:
        yield []
        return
    top, rest = group[0], group[1:]
    for opponent in reversed(rest):
        for tail in _all_high_low([t for t in rest if t != opponent]):
            yield [(top, opponent), *tail]


def _fewest_rematches(options: Sequence[list[Pair]], stage: Stage) -> list[Pair]:
    best = min(range(len(options)), key=lambda i: (_rematches(options[i], stage), i))
    stage.rematches += _rematches(options[best], stage)
    return options[best]


def _pair_group(round_index: int, group: list[int], stage: Stage) -> list[Pair]:
    if round_index in (1, 2):
        found = _search(group, stage)
        if found is not None:
            return found
        return _fewest_rematches(list(_all_high_low(group)), stage)
    rows = [[(group[a], group[b]) for a, b in row] for row in TABLE]
    for row in rows:
        if _rematches(row, stage) == 0:
            return row
    return _fewest_rematches(rows, stage)


def run_stage(
    u: Sequence[Sequence[float]],
    p_bo1: Sequence[Sequence[float]],
    p_bo3: Sequence[Sequence[float]],
) -> Stage:
    """Play one stage. Probabilities are indexed by seed index; ``u[r][j]`` decides the
    ``j``-th match of round ``r`` counted in order of the higher seed's standing."""
    stage = Stage()
    for r in range(5):
        order = stage.standings()
        position = {team: i for i, team in enumerate(order)}
        if r == 0:
            pairs = [(order[k], order[k + 8]) for k in range(8)]
        else:
            groups: dict[tuple[int, int], list[int]] = {}
            for team in order:
                if stage.wins[team] < 3 and stage.losses[team] < 3:
                    groups.setdefault((stage.wins[team], stage.losses[team]), []).append(team)
            pairs = [p for g in groups.values() for p in _pair_group(r, g, stage)]
        pairs.sort(key=lambda pair: position[pair[0]])

        for j, (high, low) in enumerate(pairs):
            assert (stage.wins[high], stage.losses[high]) == (stage.wins[low], stage.losses[low])
            best_of = 3 if stage.wins[high] == 2 or stage.losses[high] == 2 else 1
            p = (p_bo3 if best_of == 3 else p_bo1)[high][low]
            high_won = u[r][j] < p
            winner, loser = (high, low) if high_won else (low, high)
            stage.matches.append(Match(r, high, low, high_won, best_of))
            stage.opponents[high].append(low)
            stage.opponents[low].append(high)
            stage.wins[winner] += 1
            stage.losses[loser] += 1
    return stage
