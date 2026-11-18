"""Build a JSON fixture of a Major's Swiss stages from Liquipedia's raw wikitext.

Usage:
    uv run python scripts/liquipedia_swiss_fixture.py \
        --event "BLAST.tv Austin Major 2025" --out tests/fixtures/austin_2025.json \
        BLAST/Major/2025/Austin/Stage_1 BLAST/Major/2025/Austin/Stage_2 \
        BLAST/Major/2025/Austin/Stage_3

For each stage page the fixture records the seed order (from the ``SwissTableLeague`` block's
``temp_tie`` values) and every match per round with its winner and the number of maps played.
Liquipedia content is licensed CC BY-SA 3.0; the fixture keeps the source URLs for attribution.
Requests are spaced 2 s apart to respect Liquipedia's rate limits.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

BASE = "https://liquipedia.net/counterstrike/"
USER_AGENT = "swiss-fixture-builder/0.1"
MATCHES_PER_ROUND = (8, 8, 8, 6, 3)

# Liquipedia sometimes refers to one team by different template names on the same page.
DEFAULT_ALIASES = {"flyquest": "fq", "vitality": "vit", "liquid": "tl"}


def fetch_wikitext(page: str) -> str:
    request = urllib.request.Request(
        f"{BASE}{page}?action=raw",
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text: str = response.read().decode("utf-8")
    return text


def _norm(name: str) -> str:
    return name.strip().lower()


def parse_seeds(wikitext: str) -> list[str]:
    block = re.search(r"\{\{SwissTableLeague(.*?)\n\}\}", wikitext, re.S)
    if block is None:
        raise ValueError("no SwissTableLeague block")
    teams = {int(n): _norm(t) for n, t in re.findall(r"\|team(\d+)=([^|\n]+)", block.group(1))}
    ties = {int(n): int(v) for n, v in re.findall(r"\|temp_tie(\d+)=(\d+)", block.group(1))}
    if len(teams) != 16 or set(teams) != set(ties):
        raise ValueError(f"expected 16 teams with temp_tie values, got {len(teams)}")
    # temp_tie runs 16..1 from seed 1 to seed 16.
    return [teams[n] for n in sorted(teams, key=lambda n: -ties[n])]


def _rounds_won(map_text: str, team: int) -> int:
    pattern = rf"(?:^|\|)(?:o\d+)?t{team}(?:t|ct)=(\d+)"
    return sum(int(x) for x in re.findall(pattern, map_text))


def parse_matches(wikitext: str, aliases: dict[str, str]) -> list[list[dict[str, Any]]]:
    rounds: dict[int, list[dict[str, Any]]] = {}
    current: int | None = None
    for chunk in re.split(r"(?=matchsection=Round \d)|(?=\|\{\{Match\n)", wikitext):
        header = re.match(r"matchsection=Round (\d)", chunk)
        if header:
            current = int(header.group(1))
            continue
        if current is None or not chunk.startswith("|{{Match"):
            continue
        names = re.findall(r"opponent\d=\{\{TeamOpponent\|([^}|]+)", chunk)
        if len(names) != 2:
            raise ValueError(f"round {current}: cannot read opponents in {chunk[:80]!r}")
        a, b = (aliases.get(_norm(n), _norm(n)) for n in names)
        maps_a = maps_b = 0
        for map_text in re.findall(r"\|map\d=\{\{Map\|(.*?)\}\}", chunk, re.S):
            ra, rb = _rounds_won(map_text, 1), _rounds_won(map_text, 2)
            if ra == rb == 0:
                continue  # map not played
            if ra == rb:
                raise ValueError(f"round {current}: tied map in {a} vs {b}")
            maps_a += ra > rb
            maps_b += rb > ra
        if maps_a == maps_b:
            raise ValueError(f"round {current}: no winner in {a} vs {b}")
        rounds.setdefault(current, []).append(
            {"teams": [a, b], "winner": a if maps_a > maps_b else b, "maps": maps_a + maps_b}
        )
    counts = tuple(len(rounds.get(r, [])) for r in range(1, 6))
    if counts != MATCHES_PER_ROUND:
        raise ValueError(f"matches per round {counts}, expected {MATCHES_PER_ROUND}")
    return [rounds[r] for r in range(1, 6)]


def build_stage(page: str, aliases: dict[str, str]) -> dict[str, Any]:
    wikitext = fetch_wikitext(page)
    seeds = parse_seeds(wikitext)
    rounds = parse_matches(wikitext, aliases)
    unknown = {t for rnd in rounds for m in rnd for t in m["teams"]} - set(seeds)
    if unknown:
        raise ValueError(f"{page}: match teams not in the seed table: {sorted(unknown)}")
    return {"page": page, "url": f"{BASE}{page}", "seeds": seeds, "rounds": rounds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pages", nargs="+", help="Liquipedia page paths, one per stage")
    parser.add_argument("--event", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--alias", action="append", default=[], metavar="FROM=TO", help="extra team alias"
    )
    args = parser.parse_args()
    aliases = dict(DEFAULT_ALIASES)
    for item in args.alias:
        source, _, target = item.partition("=")
        aliases[_norm(source)] = _norm(target)

    stages = []
    for i, page in enumerate(args.pages):
        if i:
            time.sleep(2.0)
        stages.append({"stage": i + 1, **build_stage(page, aliases)})
    fixture = {
        "event": args.event,
        "source": "Liquipedia (https://liquipedia.net), CC BY-SA 3.0",
        "retrieved": date.today().isoformat(),
        "stages": stages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(stages)} stages)")


if __name__ == "__main__":
    main()
