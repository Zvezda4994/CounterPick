"""Swiss-stage rules from Valve's CS Major Supplemental Rulebook.

Every constant here cites the rulebook lines it implements, at a pinned commit, so a rule change
upstream shows up as a reviewable diff rather than a silent drift. See CLAUDE.md for the
interpretation of the ambiguous clauses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

RULEBOOK_COMMIT = "621b7d91b0a236044aed62a357fddc863dfaea49"
RULEBOOK_URL = (
    "https://github.com/ValveSoftware/counter-strike_rules_and_regs/blob/"
    f"{RULEBOOK_COMMIT}/major-supplemental-rulebook.md"
)

N_TEAMS = 16
N_ROUNDS = 5
MAX_MATCHES_PER_ROUND = N_TEAMS // 2
WINS_TO_ADVANCE = 3  # L166: three wins advance
LOSSES_TO_ELIMINATE = 3  # L166: three losses eliminate

Record = tuple[int, int]
"""(wins, losses)."""

Pairing = tuple[tuple[int, int], ...]
"""Pairs of 0-based ranks inside a group (0 = highest seed); each pair is (higher, lower),
and pairs are ordered by their higher seed."""

ROUND1_PAIRING: Pairing = tuple((k, k + 8) for k in range(8))
"""L190-L202 (Initial Swiss Match-ups): seed 1 v 9, 2 v 10, ..., 8 v 16 (0-based here)."""

ROUND45_PRIORITY_TABLE: tuple[Pairing, ...] = (
    ((1, 6), (2, 5), (3, 4)),
    ((1, 6), (2, 4), (3, 5)),
    ((1, 5), (2, 6), (3, 4)),
    ((1, 5), (2, 4), (3, 6)),
    ((1, 4), (2, 6), (3, 5)),
    ((1, 4), (2, 5), (3, 6)),
    ((1, 6), (2, 3), (4, 5)),
    ((1, 5), (2, 3), (4, 6)),
    ((1, 3), (2, 6), (4, 5)),
    ((1, 3), (2, 5), (4, 6)),
    ((1, 4), (2, 3), (5, 6)),
    ((1, 3), (2, 4), (5, 6)),
    ((1, 2), (3, 6), (4, 5)),
    ((1, 2), (3, 5), (4, 6)),
    ((1, 2), (3, 4), (5, 6)),
)
"""L170-L188: rounds 4 and 5 use the top-most row that does not result in a rematch.
1-based seeds within a 6-team group, exactly as printed in the rulebook."""


def is_bo3(record: Record) -> bool:
    """L167-L168: elimination and advancement matches are BO3, all other matches BO1.

    Both teams share the record, so the match decides advancement when they have one win
    left to get, and elimination when they have one loss left to give.
    """
    wins, losses = record
    return wins == WINS_TO_ADVANCE - 1 or losses == LOSSES_TO_ELIMINATE - 1


def high_low_pairings(size: int) -> tuple[Pairing, ...]:
    """Every perfect matching of a `size`-team group, in round 2-3 priority order.

    L170: "the highest seeded team faces the lowest seeded team available that does not result
    in a rematch". The highest unpaired seed tries opponents from the lowest seed upwards and
    the rest of the group is paired the same way, recursively. Taking the first matching in
    this order with no rematch is that greedy rule with lookahead: it only departs from plain
    greedy when plain greedy would strand the last teams in a forced rematch.
    """
    if size < 2 or size % 2:
        raise ValueError(f"group size must be a positive even number, got {size}")
    return tuple(tuple(m) for m in _high_low(tuple(range(size))))


def _high_low(ranks: tuple[int, ...]) -> Iterator[list[tuple[int, int]]]:
    if not ranks:
        yield []
        return
    top, rest = ranks[0], ranks[1:]
    for opponent in reversed(rest):
        remaining = tuple(r for r in rest if r != opponent)
        for tail in _high_low(remaining):
            yield [(top, opponent), *tail]


@dataclass(frozen=True)
class Group:
    """Teams sharing a record in one round, located by their slots in the standings order."""

    record: Record
    start: int
    size: int
    best_of: int
    candidates: tuple[Pairing, ...]
    """Pairings in priority order; the first without a rematch is played."""


@dataclass(frozen=True)
class RoundLayout:
    round_index: int
    """0-based."""
    groups: tuple[Group, ...]
    """Only groups that play this round, in standings order."""

    @property
    def n_matches(self) -> int:
        return sum(g.size for g in self.groups) // 2

    @property
    def best_of_by_slot(self) -> tuple[int, ...]:
        """Format of each match slot; slots are ordered by the higher seed's standing."""
        return tuple(g.best_of for g in self.groups for _ in range(g.size // 2))


def record_groups(rounds_played: int) -> tuple[tuple[Record, int, int], ...]:
    """(record, start slot, size) of every record after `rounds_played` rounds.

    Group sizes never depend on results in a 16-team Swiss (each match splits a group
    evenly), so in standings order (L256-L260: record first) every record group sits in a
    fixed slice. Records are ordered by wins descending, then losses ascending, which ranks
    3-0 > 3-1 > 3-2 > 2-3 > 1-3 > 0-3 at the end of the stage.
    """
    counts: dict[Record, int] = {(0, 0): N_TEAMS}
    for _ in range(rounds_played):
        nxt: dict[Record, int] = {}
        for (wins, losses), c in counts.items():
            if wins == WINS_TO_ADVANCE or losses == LOSSES_TO_ELIMINATE:
                nxt[(wins, losses)] = nxt.get((wins, losses), 0) + c
            else:
                nxt[(wins + 1, losses)] = nxt.get((wins + 1, losses), 0) + c // 2
                nxt[(wins, losses + 1)] = nxt.get((wins, losses + 1), 0) + c // 2
        counts = nxt
    out = []
    start = 0
    for record in sorted(counts, key=lambda r: (-r[0], r[1])):
        out.append((record, start, counts[record]))
        start += counts[record]
    return tuple(out)


def _candidates(round_index: int, size: int) -> tuple[Pairing, ...]:
    if round_index == 0:
        return (ROUND1_PAIRING,)
    if round_index in (1, 2):
        return high_low_pairings(size)
    if size != 6:
        raise AssertionError(f"round {round_index + 1} groups must have 6 teams, got {size}")
    return tuple(tuple((a - 1, b - 1) for a, b in row) for row in ROUND45_PRIORITY_TABLE)


@cache
def round_layouts() -> tuple[RoundLayout, ...]:
    """Static structure of all five rounds: who plays, in which slots, at which format."""
    layouts = []
    for r in range(N_ROUNDS):
        groups = tuple(
            Group(
                record=record,
                start=start,
                size=size,
                best_of=3 if is_bo3(record) else 1,
                candidates=_candidates(r, size),
            )
            for record, start, size in record_groups(r)
            if record[0] < WINS_TO_ADVANCE and record[1] < LOSSES_TO_ELIMINATE
        )
        layouts.append(RoundLayout(round_index=r, groups=groups))
    return tuple(layouts)
