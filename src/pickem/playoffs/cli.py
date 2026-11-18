"""pickem-playoffs: the best playoff Pick'Em picks for the 8-team bracket, by brute force.

Every legal set of picks is scored exactly against all 256 bracket outcomes, under both rule sets
(bracket-consistent and independent per round). With strength uncertainty, picks are chosen on one
set of strength draws and reported on another.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pickem.data.liquipedia import fetch_wikitext, page_url
from pickem.data.stage_file import load_stage
from pickem.optimizer.compare import resolve_team
from pickem.playoffs.bracket import (
    FORMATS,
    QF_SEEDS,
    PlayoffFormat,
    match_probabilities,
    outcome_probabilities,
    playoff_seeds,
    team_odds,
)
from pickem.playoffs.picks import (
    MODES,
    Mode,
    Objective,
    PickScore,
    PlayoffPick,
    best_picks,
    favourites_pick,
    parse_picks,
    picks_for,
    rank,
    score_picks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickem-playoffs",
        description="Brute-force the best playoff Pick'Em picks for the 8-team bracket.",
    )
    parser.add_argument("--stage", required=True, help="Stage 3 stage file (from pickem-odds)")
    seeding = parser.add_mutually_exclusive_group(required=True)
    seeding.add_argument("--results", help="Stage 3 results CSV (round,team1,team2,winner)")
    seeding.add_argument("--liquipedia", metavar="PAGE", help="e.g. PGL/2026/Singapore/Stage_3")
    seeding.add_argument("--bracket", help='8 teams in playoff seed order, "T1,...,T8" (what-if)')
    parser.add_argument("--alias", action="append", default=[], metavar="FROM=TO")
    parser.add_argument("--mode", choices=("both", *MODES), default="both")
    parser.add_argument("--format", choices=tuple(FORMATS), default="singapore")
    parser.add_argument(
        "--spread", type=float, help="strength-uncertainty multiplier (default: file's)"
    )
    parser.add_argument("--draws", type=int, default=20_000, help="strength draws per set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-qf", type=int, default=2)
    parser.add_argument("--min-sf", type=int, default=1)
    parser.add_argument("--min-gf", type=int, default=1)
    parser.add_argument("--picks", help='score your own picks: "qf: A, B, C, D; sf: A, C; gf: A"')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.draws < 2 or args.top < 1:
            raise ValueError("--draws must be at least 2 and --top at least 1")
        stage_file = load_stage(args.stage)
        teams = stage_file.stage.teams
        if args.bracket:
            bracket = tuple(teams[resolve_team(n, teams)] for n in args.bracket.split(","))
            source = "--bracket"
        elif args.results:
            bracket = playoff_seeds(stage_file, results_csv=args.results)
            source = f"Stage 3 results in {args.results}"
        else:
            aliases = dict(item.partition("=")[::2] for item in args.alias)
            bracket = playoff_seeds(
                stage_file, wikitext=fetch_wikitext(args.liquipedia), aliases=aliases
            )
            source = f"Stage 3 results at {page_url(args.liquipedia)}"
        fmt = FORMATS[args.format]
        objective = Objective(args.min_qf, args.min_sf, args.min_gf)
        rng = np.random.default_rng(args.seed)
        search = match_probabilities(
            stage_file, bracket, spread=args.spread, draws=args.draws, rng=rng
        )
        report = (
            match_probabilities(stage_file, bracket, spread=args.spread, draws=args.draws, rng=rng)
            if search.n_draws > 1
            else search
        )
        user = parse_picks(args.picks, bracket) if args.picks else None
    except (OSError, ValueError) as exc:
        print(f"pickem-playoffs: error: {exc}", file=sys.stderr)
        return 2

    search_draws = outcome_probabilities(search, fmt)
    report_draws = outcome_probabilities(report, fmt)
    modes: tuple[Mode, ...] = MODES if args.mode == "both" else (args.mode,)
    print(
        render(
            bracket=bracket,
            source=source,
            stage=args.stage,
            fmt=fmt,
            objective=objective,
            search_probs=search_draws.mean(axis=0),
            report_draws=report_draws,
            modes=modes,
            top=args.top,
            user=user,
        )
    )
    return 0


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def _pick_line(pick: PlayoffPick, teams: Sequence[str]) -> str:
    d = pick.describe(teams)
    return f"QF: {', '.join(d['QF'])} | SF: {', '.join(d['SF'])} | GF: {d['GF'][0]}"


def _score_lines(label: str, score: PickScore, teams: Sequence[str]) -> list[str]:
    error = f" +/- {100 * score.se:.2f}" if score.se else ""
    return [
        f"{label:<4}P(objective) {_pct(score.p_objective)}{error}   E[QF correct] "
        f"{score.expected_qf:.2f}   E[SF correct] {score.expected_sf:.2f}   "
        f"P(GF correct) {_pct(score.p_gf)}",
        f"     {_pick_line(score.pick, teams)}",
    ]


def render(
    *,
    bracket: Sequence[str],
    source: str,
    stage: str,
    fmt: PlayoffFormat,
    objective: Objective,
    search_probs: NDArray[np.float64],
    report_draws: NDArray[np.float64],
    modes: Sequence[Mode],
    top: int,
    user: PlayoffPick | None,
) -> str:
    draws = report_draws.shape[0]
    report_probs = report_draws.mean(axis=0)
    per_draw = report_draws if draws > 1 else None
    probabilities = (
        f"strengths from {stage} with uncertainty: best picks chosen on {draws:,} "
        f"strength draws, reported on {draws:,} independent ones"
        if draws > 1
        else f"fixed probabilities from {stage} (exact, no sampling)"
    )
    lines = [
        f"Playoffs: seeded from {source}",
        f"Format: {fmt.description}",
        f"Probabilities: {probabilities}",
        f"Objective: {objective.describe()}",
        "",
        "Bracket",
    ]
    for half, matches in (("A", (0, 1)), ("B", (2, 3))):
        pairs = "   |   ".join(
            f"({a}) {bracket[a - 1]} v ({b}) {bracket[b - 1]}"
            for a, b in (QF_SEEDS[k] for k in matches)
        )
        lines.append(f"  Half {half}: {pairs}")

    odds = team_odds(report_probs, fmt)
    width = max(len(t) for t in bracket)
    third = "P(3rd)" if fmt.third_place else "P(SF loss)"
    lines += [
        "",
        f"Seed  {'Team':<{width}}  P(win QF)  P(reach GF)  P(champion)  {third}",
    ]
    for i, team in enumerate(bracket):
        lines.append(
            f"{i + 1:>4}  {team:<{width}}  {_pct(odds.win_qf[i]):>9}  "
            f"{_pct(odds.reach_final[i]):>11}  {_pct(odds.champion[i]):>11}  "
            f"{_pct(odds.third[i]):>6}"
        )

    best: dict[Mode, PickScore] = {}
    for mode in modes:
        count = len(picks_for(mode))
        label = "bracket-consistent" if mode == "consistent" else "independent per-round"
        scores = best_picks(mode, search_probs, report_probs, objective, per_draw, top)
        best[mode] = scores[0]
        lines += ["", f"Best {label} picks (top {len(scores)} of {count:,})"]
        for i, score in enumerate(scores, start=1):
            lines += _score_lines(f"#{i}", score, bracket)
        favourite = favourites_pick(mode, search_probs)
        (fav_score,) = score_picks([favourite], report_probs, objective, per_draw)
        lines += [f"Favourites ({label}): each slot's most likely winner"]
        lines += _score_lines("", fav_score, bracket)

    if len(best) == 2:
        gap = best["independent"].p_objective - best["consistent"].p_objective
        lines += [
            "",
            f"Independent rules would add {100 * gap:+.1f} points of P(objective) to the best "
            "card.",
        ]

    if user is not None:
        valid = "valid under both rules" if user.consistent else "valid only with independent picks"
        (score,) = score_picks([user], report_probs, objective, per_draw)
        lines += ["", f"Your picks ({valid})"]
        lines += _score_lines("", score, bracket)
        for mode in modes:
            if mode == "consistent" and not user.consistent:
                continue
            ranked = rank(score_picks(picks_for(mode), report_probs, objective))
            position = next(i for i, s in enumerate(ranked, start=1) if s.pick == user)
            lines.append(f"     ranks #{position} of {len(ranked):,} {mode} picks")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
