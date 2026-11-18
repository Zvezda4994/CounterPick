"""pickem-odds: odds sheet (CSV) -> fitted team strengths -> stage JSON for pickem-optimize."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pickem.data.stage_file import StageFile, save_stage
from pickem.probs.fit import FitConfig, StrengthFit, fit_strengths
from pickem.probs.margin import METHODS
from pickem.probs.odds_csv import N_TEAMS, load_odds_csv
from pickem.probs.strength import StrengthModel, stage_from_strengths


def build_parser() -> argparse.ArgumentParser:
    defaults = FitConfig()
    parser = argparse.ArgumentParser(
        prog="pickem-odds",
        description="Fit team strengths from an odds sheet and write a stage file for "
        "pickem-optimize.",
    )
    parser.add_argument("odds", help="odds sheet CSV (see examples/singapore_2026_stage1_odds.csv)")
    parser.add_argument("--out", required=True, help="stage JSON to write")
    parser.add_argument("--name", help="stage name (default: the CSV file name)")
    parser.add_argument("--margin", choices=METHODS, default=defaults.method)
    parser.add_argument("--sigma-match", type=float, default=defaults.sigma_match)
    parser.add_argument("--sigma-qualify", type=float, default=defaults.sigma_qualify)
    parser.add_argument("--sigma-major", type=float, default=defaults.sigma_major)
    parser.add_argument("--prior-sd", type=float, default=defaults.prior_sd)
    parser.add_argument(
        "--vrs-slope",
        type=float,
        default=defaults.vrs_slope_mean,
        help="prior mean for logit strength per 100 VRS points",
    )
    parser.add_argument("--vrs-slope-sd", type=float, default=defaults.vrs_slope_sd)
    parser.add_argument(
        "--spread", type=float, default=1.0, help="default strength-uncertainty multiplier"
    )
    parser.add_argument("--sims-per-draw", type=int, default=1000)
    parser.add_argument("--fit-sims", type=int, default=defaults.fit_sims)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        sheet = load_odds_csv(args.odds)
        config = FitConfig(
            method=args.margin,
            sigma_match=args.sigma_match,
            sigma_qualify=args.sigma_qualify,
            sigma_major=args.sigma_major,
            prior_sd=args.prior_sd,
            vrs_slope_mean=args.vrs_slope,
            vrs_slope_sd=args.vrs_slope_sd,
            fit_sims=args.fit_sims,
            seed=args.seed,
        )
        fit = fit_strengths(sheet, config)
        model = fit.strength_model(args.spread, args.sims_per_draw)
    except (OSError, ValueError) as exc:
        print(f"pickem-odds: error: {exc}", file=sys.stderr)
        return 2

    stage = stage_from_strengths(fit.teams, fit.seeds, fit.strengths)
    name = args.name or Path(args.odds).stem
    try:
        save_stage(args.out, StageFile(name, stage, model), extra={"fit": summary(fit, args)})
    except OSError as exc:
        print(f"pickem-odds: error: cannot write {args.out} ({exc})", file=sys.stderr)
        return 2
    print(render_fit_report(fit, model, source=args.odds, out=args.out, config=config))
    return 0


def summary(fit: StrengthFit, args: argparse.Namespace) -> dict[str, object]:
    return {
        "source": str(args.odds),
        "margin": args.margin,
        "vrs_slope_per_100": fit.vrs_slope,
        "win_major_temperature": fit.beta,
        "qualify_correction_rounds": fit.correction_rounds,
        "converged": fit.converged,
        "residual_rms": fit.residual_rms,
    }


def _pct(p: float) -> str:
    return "    -" if math.isnan(p) else f"{100 * p:4.1f}%"


def render_fit_report(
    fit: StrengthFit, model: StrengthModel, *, source: str, out: str, config: FitConfig
) -> str:
    market = fit.market
    lines = [f"Odds fit: {source} -> {out}"]
    lines.append(
        f"Markets used: round 1 {market.n_r1_pairs}/{N_TEAMS // 2} pairs, qualify "
        f"{market.n_qualify}/{N_TEAMS} teams, win Major {market.n_win}/{N_TEAMS} teams "
        f"(margin removal: {config.method})"
    )
    for label, key in (("round 1", "r1"), ("qualify", "qualify")):
        values = market.overrounds.get(key, [])
        if values:
            lines.append(f"Overround {label}: average {100 * float(np.mean(values)):.1f}%")
    if market.n_r1_pairs + market.n_qualify + market.n_win == 0:
        lines.append("No odds given: strengths come from the VRS prior alone.")
    temperature = (
        f"; win-Major temperature {fit.beta:.2f} +/- {fit.beta_sd:.2f}"
        if fit.beta is not None and fit.beta_sd is not None
        else ""
    )
    slope = f"{fit.vrs_slope:.2f} +/- {fit.vrs_slope_sd:.2f}"
    lines.append(f"VRS slope {slope} logit per 100 points{temperature}")
    status = "converged" if fit.converged else "NOT converged: check the qualify odds"
    lines.append(
        f"Qualify check: {fit.correction_rounds} fit round(s), {config.fit_sims:,} Swiss sims "
        f"each ({status})"
    )
    if fit.residual_rms:
        parts = ", ".join(f"{k} {v:.2f}" for k, v in fit.residual_rms.items())
        lines.append(f"Residual RMS in market SDs: {parts}")
    lines.append(
        f"Strength uncertainty for pickem-optimize: spread {model.spread:.2f}, new strengths "
        f"every {model.sims_per_draw:,} sims"
    )

    width = max(len(t) for t in fit.teams)
    lines += [
        "",
        f"Seed  {'Team':<{width}}   VRS  Strength       R1 mkt  model   Qual mkt    sim"
        "   Win mkt  model",
    ]
    for i, team in enumerate(fit.teams):
        lines.append(
            f"{i + 1:>4}  {team:<{width}}  {fit.vrs_points[i]:>4.0f}  "
            f"{fit.strengths[i]:+.2f} +/-{fit.sd[i]:.2f}   {_pct(market.r1[i])}  "
            f"{_pct(fit.model_r1[i])}     {_pct(market.qualify[i])}  {_pct(fit.model_qualify[i])}"
            f"     {_pct(market.win_share[i])}  {_pct(fit.model_win_share[i])}"
        )
    if market.warnings:
        lines += ["", "Warnings:"] + [f"  {w}" for w in market.warnings]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
