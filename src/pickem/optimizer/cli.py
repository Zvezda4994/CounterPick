"""pickem-optimize: the Pick'Em card that maximizes P(>= 5 correct) for one Swiss stage.

Cards are searched on one batch of simulations and, by default, re-scored on a second,
independent batch: the best of thousands of cards looks better on the sims it was picked on
than it really is, so the holdout numbers are the honest ones.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import numpy as np

from pickem.data.stage_file import StageFile, load_stage
from pickem.optimizer.card import (
    ADVANCE,
    CATEGORY_NAMES,
    OH_THREE,
    THREE_OH,
    Card,
    CardStats,
    Outcomes,
    evaluate,
    marginals,
)
from pickem.optimizer.search import OptimizeResult, optimize
from pickem.swiss import simulate_swiss


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickem-optimize",
        description="Find the Pick'Em card that maximizes P(>=5 correct) for a Swiss stage.",
    )
    parser.add_argument("stage", help="stage input file (JSON, see pickem.data.stage_file)")
    parser.add_argument("--sims", type=_positive_int, default=100_000, help="search sims")
    parser.add_argument(
        "--eval-sims",
        type=_non_negative_int,
        default=None,
        help="independent sims to score the reported cards on (default: same as --sims; "
        "0 reports in-sample numbers)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--top", type=_positive_int, default=5, help="cards to report")
    parser.add_argument(
        "--prune", type=int, default=6, help="3-0 and 0-3 candidates kept per category"
    )
    parser.add_argument(
        "--restarts", type=_non_negative_int, default=20, help="random local-search restarts"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_sims = args.sims if args.eval_sims is None else args.eval_sims
    try:
        stage_file = load_stage(args.stage)
    except (OSError, ValueError) as exc:
        print(f"pickem-optimize: error: {exc}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    start = time.perf_counter()
    search = Outcomes.from_result(simulate_swiss(stage_file.stage, args.sims, rng))
    try:
        result = optimize(search, top=args.top, prune=args.prune, restarts=args.restarts, rng=rng)
    except ValueError as exc:
        print(f"pickem-optimize: error: {exc}", file=sys.stderr)
        return 2
    holdout = (
        Outcomes.from_result(simulate_swiss(stage_file.stage, eval_sims, rng))
        if eval_sims
        else None
    )
    elapsed = time.perf_counter() - start

    print(render_report(stage_file, search, holdout, result, seed=args.seed, elapsed=elapsed))
    return 0


def _pct(p: float) -> str:
    return f"{100 * p:5.1f}%"


def render_report(
    stage_file: StageFile,
    search: Outcomes,
    holdout: Outcomes | None,
    result: OptimizeResult,
    *,
    seed: int,
    elapsed: float,
) -> str:
    stage = stage_file.stage
    names = stage.teams
    by_seed = [int(t) for t in stage.by_seed]
    width = max(len(n) for n in names)
    lines = [f"Pick'Em optimizer: {stage_file.name}"]
    lines.append(
        f"Search: {search.n_sims:,} sims, {result.cards_evaluated:,} cards scored, "
        f"seed {seed}, {elapsed:.1f}s total"
    )
    lines.append(
        f"Holdout: {holdout.n_sims:,} independent sims"
        if holdout
        else "Holdout: none (all numbers are in-sample and optimistic)"
    )

    lines += ["", "Team marginals (search sims)"]
    lines.append(f"Seed  {'Team':<{width}}  P(3-0)  P(3-1/3-2)  P(0-3)")
    table = marginals(search)
    for t in by_seed:
        row = table.iloc[t]
        lines.append(
            f"{stage.seeds[t]:>4}  {names[t]:<{width}}  {_pct(row['P(3-0)'])}"
            f"      {_pct(row['P(3-1/3-2)'])}  {_pct(row['P(0-3)'])}"
        )

    def ordered(teams: Sequence[int]) -> str:
        return ", ".join(names[t] for t in sorted(teams, key=lambda t: stage.seeds[t]))

    lines.append(
        f"Candidates after pruning: 3-0 {{{ordered(result.candidates[THREE_OH])}}}; "
        f"0-3 {{{ordered(result.candidates[OH_THREE])}}}"
    )

    def block(label: str, card: Card, in_sample: CardStats) -> list[str]:
        stats = evaluate(card, holdout) if holdout else in_sample
        header = (
            f"{label}  P(>=5) {_pct(stats.p5)} +/- {100 * stats.se5:.2f}"
            f"   P(>=6) {_pct(stats.p6)}   E[correct] {stats.expected:.2f}"
        )
        if holdout:
            header += f"   [search P(>=5) {_pct(in_sample.p5).strip()}]"
        return [
            header,
            f"     {CATEGORY_NAMES[THREE_OH] + ':':<9}{ordered(card.three_oh)}",
            f"     {CATEGORY_NAMES[OH_THREE] + ':':<9}{ordered(card.oh_three)}",
            f"     {CATEGORY_NAMES[ADVANCE] + ':':<9}{ordered(card.advance)}",
        ]

    source = "holdout sims" if holdout else "search sims"
    lines += ["", f"Top {len(result.top)} cards (ranked on search sims; numbers from {source})"]
    for i, scored in enumerate(result.top, start=1):
        lines += block(f"#{i:<2}", scored.card, scored.stats)
    lines += ["", "Favorites baseline (top-2 P(3-0), top-2 P(0-3), top-6 P(3-1/3-2))"]
    lines += block("   ", result.favorites.card, result.favorites.stats)

    best = evaluate(result.best.card, holdout) if holdout else result.best.stats
    fav = evaluate(result.favorites.card, holdout) if holdout else result.favorites.stats
    lines += [
        "",
        f"Best card vs favorites: {100 * (best.p5 - fav.p5):+.1f} points of P(>=5), "
        f"{best.expected - fav.expected:+.2f} expected correct",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
