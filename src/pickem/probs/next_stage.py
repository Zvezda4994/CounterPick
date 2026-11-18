"""pickem-next-stage: the next Swiss stage's odds sheet, built from the previous stage's results.

Rulebook (pinned commit, Stage Seeding L241-L260): teams invited directly to a stage are seeds
1-8 in pre-event order, which is their global VRS standing; the 8 teams advancing from the
previous stage are seeds 9-16 in their final mid-stage seed order (record, then Difficulty,
then initial seed). Round 1 is seed k v seed k+8 (L190-L202).

The previous stage is replayed through the Swiss rules engine (``pickem.swiss.replay``), so a
mistyped result shows up as a round that the rules would have paired differently.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pickem.data.liquipedia import fetch_wikitext, page_url, parse_stage
from pickem.data.results import load_results_csv
from pickem.data.vrs import VrsEntry, VrsTable, load_vrs
from pickem.probs.odds_csv import N_TEAMS, OddsSheet, load_odds_csv, r1_opponent, write_odds_sheet
from pickem.swiss.replay import MatchResult, StageReplay, replay_stage

N_INVITED = N_TEAMS // 2


@dataclass(frozen=True)
class Invite:
    name: str
    """Name used in the new sheet."""
    vrs_name: str
    entry: VrsEntry


@dataclass(frozen=True)
class NextStage:
    teams: tuple[str, ...]
    """Seed order: invited teams (seeds 1-8), then advancing teams (seeds 9-16)."""
    vrs_points: tuple[float, ...]
    invited: tuple[Invite, ...]
    replay: StageReplay
    warnings: tuple[str, ...] = ()


def parse_invited(items: Sequence[str]) -> list[tuple[str, str]]:
    """``Name`` or ``Name=VRS name`` -> (name for the sheet, name to look up in the VRS)."""
    out = []
    for item in items:
        name, sep, vrs_name = item.partition("=")
        name, vrs_name = name.strip(), vrs_name.strip() if sep else name.strip()
        if not name or not vrs_name:
            raise ValueError(f"invited team {item!r} is empty")
        out.append((name, vrs_name))
    return out


def read_invited_file(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def replay_from_results(teams_by_seed: Sequence[str], path: str | Path) -> StageReplay:
    """Replay a typed results CSV; ``teams_by_seed[i]`` is seed ``i + 1``."""
    rounds = load_results_csv(path, teams_by_seed)
    return replay_stage(teams_by_seed, range(1, len(teams_by_seed) + 1), rounds)


def replay_from_liquipedia(
    teams_by_seed: Sequence[str], wikitext: str, aliases: Mapping[str, str] | None = None
) -> StageReplay:
    """Liquipedia aliases are matched to ``teams_by_seed`` by seed; the replay then checks that
    round 1 really was seed k v k+8 under that mapping."""
    stage = parse_stage(wikitext, aliases)
    name = dict(zip(stage.seeds, teams_by_seed, strict=True))
    rounds = tuple(
        tuple(MatchResult(name[m.team1], name[m.team2], name[m.winner], m.maps) for m in matches)
        for matches in stage.rounds
    )
    return replay_stage(teams_by_seed, range(1, len(teams_by_seed) + 1), rounds)


def build_next_stage(
    previous: OddsSheet,
    replay: StageReplay,
    invited: Sequence[tuple[str, str]],
    vrs: VrsTable,
) -> NextStage:
    if len(invited) != N_INVITED:
        raise ValueError(f"need {N_INVITED} invited teams, got {len(invited)}")
    names = [name for name, _ in invited]
    if len(set(names)) != N_INVITED:
        raise ValueError("an invited team is listed twice")
    advancing = replay.advancing
    clash = sorted(set(names) & set(advancing))
    if clash:
        raise ValueError(
            f"{', '.join(clash)} advanced from the previous stage, so cannot be invited"
        )

    warnings = []
    invites = []
    for name, vrs_name in invited:
        found = vrs.matches(vrs_name)
        if not found:
            raise ValueError(
                f"invited team {vrs_name!r} is not in the VRS table {vrs.source}; spell it as the "
                "file does, or write Name=VRS name"
            )
        if len(found) > 1:
            warnings.append(
                f"{vrs_name} appears {len(found)} times in the VRS table; using standing "
                f"{found[0].standing} ({found[0].roster or 'no roster listed'})"
            )
        invites.append(Invite(name, vrs_name, found[0]))
    if len({i.entry.standing for i in invites}) != N_INVITED:
        raise ValueError("two invited names point to the same VRS entry")
    invites.sort(key=lambda i: i.entry.standing)

    previous_points = dict(zip(previous.teams, previous.vrs_points, strict=True))
    return NextStage(
        teams=(*(i.name for i in invites), *advancing),
        vrs_points=(
            *(float(i.entry.points) for i in invites),
            *(float(previous_points[t]) for t in advancing),
        ),
        invited=tuple(invites),
        replay=replay,
        warnings=tuple(warnings),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickem-next-stage",
        description="Build the next Swiss stage's odds sheet from the previous stage's results.",
    )
    parser.add_argument("--previous", required=True, help="previous stage's odds sheet (CSV)")
    results = parser.add_mutually_exclusive_group(required=True)
    results.add_argument(
        "--results", help="previous stage's results CSV (round,team1,team2,winner)"
    )
    results.add_argument(
        "--liquipedia",
        metavar="PAGE",
        help="read results from Liquipedia, e.g. PGL/2026/Singapore/Stage_1",
    )
    parser.add_argument("--vrs", required=True, help="Valve VRS standings file (path or URL)")
    invited = parser.add_mutually_exclusive_group(required=True)
    invited.add_argument(
        "--invited", help='comma-separated, e.g. "FUT,Nemiga,..." (Name=VRS name allowed)'
    )
    invited.add_argument("--invited-file", help="text file, one invited team per line")
    parser.add_argument("--out", required=True, help="odds sheet to write")
    parser.add_argument("--force", action="store_true", help="replace --out if it exists")
    parser.add_argument(
        "--alias", action="append", default=[], metavar="FROM=TO", help="Liquipedia team alias"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        previous = load_odds_csv(args.previous)
        if args.results:
            replay = replay_from_results(previous.teams, args.results)
            source = str(args.results)
        else:
            aliases = dict(item.partition("=")[::2] for item in args.alias)
            wikitext = fetch_wikitext(args.liquipedia)
            replay = replay_from_liquipedia(previous.teams, wikitext, aliases)
            source = page_url(args.liquipedia)
        vrs = load_vrs(args.vrs)
        items = args.invited.split(",") if args.invited else read_invited_file(args.invited_file)
        next_stage = build_next_stage(previous, replay, parse_invited(items), vrs)
        as_of = f", standings as of {vrs.as_of}" if vrs.as_of else ""
        comments = [
            f"Odds sheet built by pickem-next-stage on {date.today().isoformat()}.",
            f"Seeds 1-8: invited teams in VRS standing order ({vrs.source}{as_of}).",
            f"Seeds 9-16: teams advancing from {args.previous}, in final mid-stage seed order",
            f"(record, then Difficulty, then initial seed), replayed from {source}.",
            "Round 1 is seed k v seed k+8. Fill in DECIMAL odds as described in",
            "examples/singapore_2026_stage1_odds.csv, then run pickem-odds on this file.",
        ]
        write_odds_sheet(
            args.out, next_stage.teams, next_stage.vrs_points, comments, overwrite=args.force
        )
    except (OSError, ValueError) as exc:
        print(f"pickem-next-stage: error: {exc}", file=sys.stderr)
        return 2
    print(render(next_stage, previous=args.previous, source=source, out=args.out))
    return 0


def render(next_stage: NextStage, *, previous: str, source: str, out: str) -> str:
    replay = next_stage.replay
    width = max(len(t) for t in (*replay.teams, *next_stage.teams))
    lines = [f"Previous stage: {previous} (results from {source})", "Final standings:"]
    for team in replay.standings:
        i = replay.teams.index(team)
        status = "advances" if replay.final_seed[i] <= N_INVITED else "eliminated"
        lines.append(
            f"  {replay.final_seed[i]:>2}  {team:<{width}}  {replay.record(team)}"
            f"  Difficulty {replay.difficulty[i]:+d}  {status}"
        )
    invited = {i.name: i for i in next_stage.invited}
    lines += ["", f"Next stage seeding (written to {out}):"]
    for seed, (team, points) in enumerate(
        zip(next_stage.teams, next_stage.vrs_points, strict=True), 1
    ):
        origin = (
            f"invited, VRS standing {invited[team].entry.standing}"
            if team in invited
            else f"advanced {replay.record(team)}"
        )
        lines.append(f"  {seed:>2}  {team:<{width}}  VRS {points:>6.0f}  {origin}")
    pairs = [
        f"{next_stage.teams[i]} v {next_stage.teams[r1_opponent(i)]}" for i in range(N_INVITED)
    ]
    lines += ["", "Round 1: " + "; ".join(pairs)]
    if next_stage.warnings:
        lines += ["", "Warnings:"] + [f"  {w}" for w in next_stage.warnings]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
