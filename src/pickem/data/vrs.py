"""Valve Regional Standings (VRS) tables from ValveSoftware/counter-strike_regional_standings.

Each file is markdown: a ``### Standings as of YYYY_MM_DD`` heading and a table
``| Standing | Points | Team Name | Roster | details |``. Seeding uses the Standing column, since
points can tie.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

USER_AGENT = "vrs-reader/0.1"


@dataclass(frozen=True)
class VrsEntry:
    standing: int
    points: int
    team: str
    roster: str = ""


@dataclass(frozen=True)
class VrsTable:
    entries: tuple[VrsEntry, ...]
    source: str
    as_of: str | None = None

    def matches(self, name: str) -> list[VrsEntry]:
        """Entries with exactly this team name (case ignored), best standing first."""
        key = name.strip().casefold()
        return sorted(
            (e for e in self.entries if e.team.casefold() == key), key=lambda e: e.standing
        )

    def lookup(self, name: str) -> VrsEntry:
        found = self.matches(name)
        if not found:
            raise ValueError(f"{name!r} is not in the VRS table {self.source}")
        return found[0]


def parse_vrs(text: str, source: str) -> VrsTable:
    entries = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        try:
            entries.append(
                VrsEntry(
                    standing=int(cells[0]),
                    points=int(cells[1]),
                    team=cells[2],
                    roster=cells[3] if len(cells) > 3 else "",
                )
            )
        except ValueError as exc:
            raise ValueError(f"{source}: cannot read VRS row {line!r}") from exc
    if not entries:
        raise ValueError(f"{source}: no VRS table rows found")
    heading = re.search(r"Standings as of (\d{4})_(\d{2})_(\d{2})", text)
    as_of = "-".join(heading.groups()) if heading else None
    return VrsTable(entries=tuple(entries), source=source, as_of=as_of)


def load_vrs(location: str | Path) -> VrsTable:
    """Read a VRS file from a local path or an http(s) URL."""
    location = str(location)
    if location.startswith(("http://", "https://")):
        request = urllib.request.Request(location, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(location).read_text(encoding="utf-8")
    return parse_vrs(text, location)
