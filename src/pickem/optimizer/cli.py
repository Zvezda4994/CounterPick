"""pickem-optimize: the Pick'Em card that maximizes P(>= 5 correct) for one Swiss stage.

Cards are searched on one batch of simulations and, by default, re-scored on a second,
independent batch: the best of thousands of cards looks better on the sims it was picked on
than it really is, so the holdout numbers are the honest ones.

If the stage file carries a strength model (written by pickem-odds), the sims also sample team
strengths from the fit's uncertainty; ``--spread`` scales it and ``--compare-spread`` shows how
the best card changes against fixed probabilities (spread 0).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

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
from pickem.probs.uncertain import simulate_with_uncertainty
from pickem.swiss import SwissResult, simulate_swiss


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


def _non_negative_float(text: str) -> float:
    value = float(text)
    if not value >= 0:
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
    parser.add_argument(
        "--spread",
        type=_non_negative_float,
        default=None,
        help="strength-uncertainty multiplier (default: the stage file's; 0 = fixed probabilities)",
    )
    parser.add_argument(
        "--sims-per-draw",
        type=_positive_int,
        default=None,
        help="sims that share one draw of strengths (default: the stage file's)",
    )
    parser.add_argument(
        "--compare-spread",
        action="store_true",
        help="also optimize with fixed probabilities (spread 0) and compare the best cards",
    )
    return parser


@dataclass(frozen=True)
class Run:
    spread: float
    search: Outcomes
    result: OptimizeResult
    holdout: Outcomes | None


class _Error(Exception):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_sims = args.sims if args.eval_sims is None else args.eval_sims
    try:
        stage_file = load_stage(args.stage)
        model = stage_file.strength_model
        spread = args.spread if args.spread is not None else (model.spread if model else 0.0)
        per_draw = args.sims_per_draw or (model.sims_per_draw if model else 1000)
        if spread > 0 and model is None:
            raise _Error(
                f"{args.stage} has no strength_model (write it with pickem-odds), so only "
                "--spread 0 is possible"
            )
        if args.compare_spread and spread == 0:
            raise _Error("--compare-spread needs a stage file with a strength model and spread > 0")
        if args.compare_spread and eval_sims == 0:
            raise _Error("--compare-spread needs holdout sims (--eval-sims > 0)")
    except (OSError, ValueError, _Error) as exc:
        print(f"pickem-optimize: error: {exc}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)

    def simulate(n: int, at_spread: float) -> SwissResult:
        if at_spread == 0 or model is None:
            return simulate_swiss(stage_file.stage, n, rng)
        return simulate_with_uncertainty(
            stage_file.stage, model, n, rng, spread=at_spread, sims_per_draw=per_draw
        )

    def run(at_spread: float) -> Run:
        search = Outcomes.from_result(simulate(args.sims, at_spread))
        result = optimize(search, top=args.top, prune=args.prune, restarts=args.restarts, rng=rng)
        holdout = Outcomes.from_result(simulate(eval_sims, at_spread)) if eval_sims else None
        return Run(at_spread, search, result, holdout)

    start = time.perf_counter()
    try:
        main_run = run(spread)
        fixed_run = run(0.0) if args.compare_spread else None
    except ValueError as exc:
        print(f"pickem-optimize: error: {exc}", file=sys.stderr)
        return 2
    elapsed = time.perf_counter() - start

    uncertainty = (
        f"spread {spread:.2f}, new team strengths every {per_draw:,} sims"
        if spread > 0
        else "none (fixed probabilities)"
    )
    report = render_report(
        stage_file,
        main_run.search,
        main_run.holdout,
        main_run.result,
        seed=args.seed,
        elapsed=elapsed,
        uncertainty=uncertainty,
    )
    if fixed_run is not None:
        report += "\n" + render_spread_comparison(stage_file, fixed_run, main_run)
    print(report)
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
    uncertainty: str = "none (fixed probabilities)",
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
    lines.append(f"Strength uncertainty: {uncertainty}")

    lines += ["", "Team marginals (search sims)"]
    lines.append(f"Seed  {'Team':<{width}}  P(3-0)  P(3-1/3-2)  P(0-3)")
    table = marginals(search)
    for t in by_seed:
        row = table.iloc[t]
        lines.append(
            f"{stage.seeds[t]:>4}  {names[t]:<{width}}  {_pct(row['P(3-0)'])}"
            f"      {_pct(row['P(3-1/3-2)'])}  {_pct(row['P(0-3)'])}"
        )

    lines.append(
        f"Candidates after pruning: 3-0 {{{_ordered(stage_file, result.candidates[THREE_OH])}}}; "
        f"0-3 {{{_ordered(stage_file, result.candidates[OH_THREE])}}}"
    )

    def block(label: str, card: Card, in_sample: CardStats) -> list[str]:
        stats = evaluate(card, holdout) if holdout else in_sample
        header = (
            f"{label}  P(>=5) {_pct(stats.p5)} +/- {100 * stats.se5:.2f}"
            f"   P(>=6) {_pct(stats.p6)}   E[correct] {stats.expected:.2f}"
        )
        if holdout:
            header += f"   [search P(>=5) {_pct(in_sample.p5).strip()}]"
        return [header, *_card_lines(stage_file, card)]

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


def _ordered(stage_file: StageFile, teams: Sequence[int]) -> str:
    stage = stage_file.stage
    return ", ".join(stage.teams[t] for t in sorted(teams, key=lambda t: stage.seeds[t]))


def _card_lines(stage_file: StageFile, card: Card) -> list[str]:
    return [
        f"     {CATEGORY_NAMES[c] + ':':<9}{_ordered(stage_file, card.groups[c])}"
        for c in (THREE_OH, OH_THREE, ADVANCE)
    ]


def render_spread_comparison(stage_file: StageFile, fixed: Run, uncertain: Run) -> str:
    """Best card with fixed probabilities vs with strength uncertainty, each scored both ways."""
    assert fixed.holdout is not None and uncertain.holdout is not None
    label_fixed, label_unc = "spread 0", f"spread {uncertain.spread:.2f}"
    cards = {label_fixed: fixed.result.best.card, label_unc: uncertain.result.best.card}
    evals = {label_fixed: fixed.holdout, label_unc: uncertain.holdout}

    lines = [
        "",
        "Strength uncertainty: does it change the best card? (holdout P(>=5))",
        f"{'':<22}" + "".join(f"{'sims at ' + name:>23}" for name in evals),
    ]
    for card_label, card in cards.items():
        cells = ""
        for outcomes in evals.values():
            stats = evaluate(card, outcomes)
            text = f"{100 * stats.p5:.1f}% +/-{100 * stats.se5:.1f}"
            cells += f"{text:>23}"
        lines.append(f"{'best card @ ' + card_label:<22}{cells}")

    a, b = cards[label_fixed], cards[label_unc]
    if a == b:
        lines.append("Same best card with and without strength uncertainty.")
        return "\n".join(lines)
    lines.append("The best cards differ:")
    for c in (THREE_OH, OH_THREE, ADVANCE):
        only_fixed = sorted(set(a.groups[c]) - set(b.groups[c]))
        only_unc = sorted(set(b.groups[c]) - set(a.groups[c]))
        if only_fixed or only_unc:
            lines.append(
                f"  {CATEGORY_NAMES[c]}: {label_fixed} picks {{{_ordered(stage_file, only_fixed)}}}"
                f", {label_unc} picks {{{_ordered(stage_file, only_unc)}}}"
            )
    lines += [f"Best card @ {label_fixed}:", *_card_lines(stage_file, a)]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
