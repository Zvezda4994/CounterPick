"""Read a Swiss stage's seeding and results from Liquipedia's raw wikitext.

Two page layouts are understood:
- ``{{SwissTableLeague ... |teamN=alias ... |temp_tieN=17-seed}}`` (e.g. Austin 2025)
- ``{{SwissStandings ... |{{TeamOpponent|alias|tiebreaker=17-seed}}}}`` (e.g. Singapore 2026)

Matches are ``{{Match|opponent1={{TeamOpponent|a}}|opponent2={{TeamOpponent|b}} ...}}`` under a
``matchsection=Round N`` header. A match's winner is its ``winner=`` parameter if present, else the
higher ``score=`` on the two TeamOpponents, else the side that won more maps (rounds summed over
halves and overtimes). Liquipedia content is licensed CC BY-SA 3.0.
"""

from __future__ import annotations

import re
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass

from pickem.swiss.replay import MATCHES_PER_ROUND, MatchResult

BASE = "https://liquipedia.net/counterstrike/"
USER_AGENT = "swiss-results-reader/0.1"
DEFAULT_ALIASES: Mapping[str, str] = {"flyquest": "fq", "vitality": "vit", "liquid": "tl"}
"""Second spellings seen on one page for the same team; used only when they are not seeds."""


@dataclass(frozen=True)
class LiquipediaStage:
    seeds: tuple[str, ...]
    """Team aliases in seed order (seed 1 first)."""
    rounds: tuple[tuple[MatchResult, ...], ...]
    """Results with the aliases used in ``seeds``."""


def page_url(page: str) -> str:
    return f"{BASE}{page}"


def fetch_wikitext(page: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        f"{page_url(page)}?action=raw",
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text: str = response.read().decode("utf-8")
    if not text.strip():
        raise ValueError(f"Liquipedia page {page!r} is empty or does not exist")
    return text


def _norm(alias: str) -> str:
    return alias.strip().lower()


def parse_seed_order(wikitext: str) -> tuple[str, ...]:
    """Team aliases in seed order, from either standings layout."""
    ranked: list[tuple[int, str]] = []
    table = re.search(r"\{\{SwissTableLeague(.*?)\n\}\}", wikitext, re.S)
    if table:
        teams = {int(n): t for n, t in re.findall(r"\|team(\d+)=([^|\n]*)", table.group(1))}
        ties = {int(n): int(v) for n, v in re.findall(r"\|temp_tie(\d+)=(\d+)", table.group(1))}
        ranked = [(ties[n], teams[n]) for n in teams if n in ties]
    else:
        found = re.findall(r"\{\{TeamOpponent\|([^|}]*)\|tiebreaker=(\d+)\}\}", wikitext)
        ranked = [(int(v), t) for t, v in found]
    if len(ranked) != 16:
        raise ValueError(f"expected 16 seeded teams in the standings, found {len(ranked)}")
    ranked.sort(key=lambda item: -item[0])  # 16 = seed 1
    seeds = tuple(_norm(alias) for _, alias in ranked)
    undecided = [i + 1 for i, alias in enumerate(seeds) if not alias]
    if undecided:
        raise ValueError(f"seeds {undecided} are not decided on the page yet")
    if len(set(seeds)) != 16:
        raise ValueError("the standings list a team twice")
    return seeds


def _rounds_won(map_text: str, side: int) -> int:
    return sum(int(x) for x in re.findall(rf"(?:^|\|)(?:o\d+)?t{side}(?:t|ct)=(\d+)", map_text))


def _winner_side(chunk: str) -> int | None:
    maps = re.findall(r"\|map\d+=\{\{Map\|(.*?)\}\}", chunk, re.S)
    outside_maps = re.sub(r"\|map\d+=\{\{Map\|.*?\}\}", "", chunk, flags=re.S)
    explicit = re.search(r"\|winner=([12])\b", outside_maps)
    if explicit:
        return int(explicit.group(1))
    scores = re.findall(r"opponent([12])=\{\{TeamOpponent\|[^}]*?\|score=(\d+)", chunk)
    if len(scores) == 2:
        by_side = {int(side): int(score) for side, score in scores}
        if by_side[1] != by_side[2]:
            return 1 if by_side[1] > by_side[2] else 2
    won = {1: 0, 2: 0}
    for map_text in maps:
        a, b = _rounds_won(map_text, 1), _rounds_won(map_text, 2)
        if a != b:
            won[1 if a > b else 2] += 1
    if won[1] == won[2]:
        return None
    return 1 if won[1] > won[2] else 2


def _maps_played(chunk: str) -> int:
    maps = re.findall(r"\|map\d+=\{\{Map\|(.*?)\}\}", chunk, re.S)
    return sum(1 for m in maps if _rounds_won(m, 1) + _rounds_won(m, 2) > 0)


def parse_rounds(wikitext: str) -> tuple[tuple[tuple[str, str, int | None, int], ...], ...]:
    """Per round: (alias 1, alias 2, winning side or None if unplayed, maps played)."""
    rounds: dict[int, list[tuple[str, str, int | None, int]]] = {}
    current: int | None = None
    for chunk in re.split(r"(?=matchsection=Round \d)|(?=\|\{\{Match\s*\n)", wikitext):
        header = re.match(r"matchsection=Round (\d)", chunk)
        if header:
            current = int(header.group(1))
            continue
        if current is None or not chunk.startswith("|{{Match"):
            continue
        names = re.findall(r"opponent[12]=\{\{TeamOpponent\|([^}|]*)", chunk)
        if len(names) != 2:
            raise ValueError(f"round {current}: cannot read the two teams of a match")
        rounds.setdefault(current, []).append(
            (_norm(names[0]), _norm(names[1]), _winner_side(chunk), _maps_played(chunk))
        )
    return tuple(tuple(rounds.get(r, [])) for r in range(1, len(MATCHES_PER_ROUND) + 1))


def parse_stage(wikitext: str, aliases: Mapping[str, str] | None = None) -> LiquipediaStage:
    """Seeds and complete results; raises ``ValueError`` if the stage is not finished."""
    seeds = parse_seed_order(wikitext)
    known = set(seeds)
    extra = {**DEFAULT_ALIASES, **{_norm(k): _norm(v) for k, v in (aliases or {}).items()}}

    def resolve(alias: str, r: int) -> str:
        if not alias:
            raise ValueError(f"round {r} still has undecided teams on the page")
        if alias in known:
            return alias
        if extra.get(alias) in known:
            return extra[alias]
        raise ValueError(
            f"round {r}: team {alias!r} is not in the standings ({', '.join(seeds)}); "
            f"map it with --alias {alias}=<seeded alias>"
        )

    rounds = []
    for r, matches in enumerate(parse_rounds(wikitext), start=1):
        expected = MATCHES_PER_ROUND[r - 1]
        if len(matches) != expected:
            raise ValueError(
                f"round {r} has {len(matches)} matches on the page, expected {expected}"
            )
        results = []
        for a, b, side, maps in matches:
            team1, team2 = resolve(a, r), resolve(b, r)
            if side is None:
                raise ValueError(f"round {r}: {team1} v {team2} has no result on the page yet")
            winner = team1 if side == 1 else team2
            results.append(MatchResult(team1, team2, winner, maps or None))
        rounds.append(tuple(results))
    return LiquipediaStage(seeds=seeds, rounds=tuple(rounds))
