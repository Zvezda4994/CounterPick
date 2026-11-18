"""Cards file: one Pick'Em card per row, for a stage's leaderboard.

Columns (lines starting with ``#`` are ignored)::

    name,source,link,three_oh_1,three_oh_2,oh_three_1,oh_three_2,advance_1,...,advance_6

One team per cell, so no CSV quoting is needed. Team names use the ``--card`` matching: exact,
initials ("NiP") or unique prefix, case ignored. ``source`` is one of ``SOURCES``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pickem.optimizer.card import ADVANCE, OH_THREE, THREE_OH, Card
from pickem.optimizer.compare import resolve_team

SOURCES = ("analyst", "friend", "me", "optimizer", "seed chalk", "common method")
PICK_COLUMNS = (
    "three_oh_1",
    "three_oh_2",
    "oh_three_1",
    "oh_three_2",
    *(f"advance_{i}" for i in range(1, 7)),
)
COLUMNS = ("name", "source", "link", *PICK_COLUMNS)


@dataclass(frozen=True)
class CardEntry:
    name: str
    source: str
    link: str
    card: Card


def _rows(path: Path) -> list[list[str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return [[cell.strip() for cell in row] for row in csv.reader(lines)]


def load_cards(path: str | Path, teams: Sequence[str]) -> list[CardEntry]:
    """Read and validate a cards file against the stage's team names."""
    path = Path(path)
    rows = _rows(path)
    if not rows:
        raise ValueError(f"{path}: no header row")
    header, body = rows[0], rows[1:]
    if tuple(header) != COLUMNS:
        missing = [c for c in COLUMNS if c not in header]
        extra = [c for c in header if c not in COLUMNS]
        raise ValueError(
            f"{path}: columns must be {','.join(COLUMNS)} (missing {missing}, unexpected {extra})"
        )
    entries: list[CardEntry] = []
    seen: set[str] = set()
    for number, row in enumerate(body, start=1):
        where = f"{path} card {number}"
        if len(row) != len(COLUMNS):
            raise ValueError(f"{where}: expected {len(COLUMNS)} cells, got {len(row)}")
        values = dict(zip(COLUMNS, row, strict=True))
        name = values["name"]
        if not name:
            raise ValueError(f"{where}: name is empty")
        where = f"{path} card {number} ({name})"
        if name.casefold() in seen:
            raise ValueError(f"{where}: another card already has this name")
        seen.add(name.casefold())
        source = values["source"].casefold()
        if source not in SOURCES:
            raise ValueError(f"{where}: source must be one of {', '.join(SOURCES)}")
        try:
            picks = [resolve_team(values[c], teams) for c in PICK_COLUMNS]
            card = Card(
                three_oh=tuple(picks[0:2]), oh_three=tuple(picks[2:4]), advance=tuple(picks[4:])
            )
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        entries.append(CardEntry(name=name, source=source, link=values["link"], card=card))
    return entries


def card_cells(card: Card, teams: Sequence[str]) -> list[str]:
    """Pick cells in column order, with canonical team names."""
    return [teams[t] for g in (card.three_oh, card.oh_three, card.advance) for t in g]


def append_card(path: str | Path, entry: CardEntry, teams: Sequence[str]) -> None:
    """Add a card, creating the file with its header if needed; names must stay unique."""
    path = Path(path)
    if path.exists() and any(
        e.name.casefold() == entry.name.casefold() for e in load_cards(path, teams)
    ):
        raise ValueError(f"{path} already has a card named {entry.name!r}")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if not path.exists():
        writer.writerow(COLUMNS)
    writer.writerow([entry.name, entry.source, entry.link, *card_cells(entry.card, teams)])
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + buffer.getvalue(), encoding="utf-8")


def picks_by_category(card: Card, teams: Sequence[str]) -> dict[str, list[str]]:
    return {
        "3-0": [teams[t] for t in card.groups[THREE_OH]],
        "0-3": [teams[t] for t in card.groups[OH_THREE]],
        "advance": [teams[t] for t in card.groups[ADVANCE]],
    }
