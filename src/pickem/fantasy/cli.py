"""pickem-fantasy: HLTV fantasy lineups for one Swiss stage.

Reports the lineup with the most expected points and a high-upside lineup that maximizes
expected HLTV season points (or P(top N%)) against a simulated field. Lineups are chosen on one
batch of sims and field and reported on an independent batch, like pickem-optimize.

    pickem-fantasy data/stage1.json --players data/stage1_fantasy.csv
    pickem-fantasy data/stage1.json --players ... --lineup "s1mple, Senzu, rain, device, NAF"
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pickem.data.stage_file import StageFile, load_stage
from pickem.fantasy.boosters import calibrate_boosters, load_booster_rates
from pickem.fantasy.field import field_scores, ownership, sample_field, tier_cutoffs
from pickem.fantasy.lineup import Evaluator, Plan, check_lineup
from pickem.fantasy.live import booster_indices, plan_round, played_matches, teams_alive
from pickem.fantasy.outcomes import simulate_map_outcomes
from pickem.fantasy.performance import PlayerSims, simulate_players
from pickem.fantasy.players import PlayerPool, load_players
from pickem.fantasy.rules import (
    BOOSTER_NAMES,
    DEFAULT_BUDGET,
    DEFAULT_MAX_PER_TEAM,
    N_ROUNDS,
    ROLE_NAMES,
    SEASON_TIERS,
)
from pickem.fantasy.search import UpsideStats, max_expected, max_upside, upside


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickem-fantasy",
        description="HLTV fantasy lineups for a Swiss stage: max expected points and high upside.",
    )
    parser.add_argument("stage", help="stage file (JSON, as for pickem-optimize)")
    parser.add_argument("--players", required=True, help="player pool CSV (team, player, price…)")
    parser.add_argument("--budget", type=_positive_int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--max-per-team",
        type=_positive_int,
        default=DEFAULT_MAX_PER_TEAM,
        help=f"players allowed from one team (default {DEFAULT_MAX_PER_TEAM})",
    )
    parser.add_argument("--sims", type=_positive_int, default=10_000, help="search sims")
    parser.add_argument(
        "--eval-sims", type=_positive_int, default=None, help="holdout sims (default: --sims)"
    )
    parser.add_argument("--field", type=_positive_int, default=4000, help="field lineups")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective",
        default="season",
        help="high-upside target: season (E[season points], default) or top1, top5, top10, ...",
    )
    parser.add_argument("--restarts", type=int, default=6, help="random local-search starts")
    parser.add_argument("--spread", type=float, default=None, help="strength-uncertainty scale")
    parser.add_argument("--sims-per-draw", type=_positive_int, default=None)
    parser.add_argument(
        "--team-card-weight",
        type=float,
        default=0.0,
        help="how much a team's average card rating counts beyond the odds (0 = not at all, "
        "default; 1 = as fitted on IEM Beijing, where all teams were tier 1)",
    )
    parser.add_argument(
        "--lineup", help='score your own lineup, e.g. "s1mple, Senzu, rain, device, NAF"'
    )
    parser.add_argument("--top-players", type=int, default=15, help="player table length")
    parser.add_argument(
        "--booster-rates",
        help="HLTV booster trigger rates (CSV: player + 18 booster columns, percent of matches); "
        "listed players use them instead of the model's defaults",
    )
    live = parser.add_argument_group("during the stage")
    live.add_argument(
        "--round",
        type=int,
        help="plan roles and boosters of --lineup for this round (1-5) instead of searching",
    )
    live.add_argument("--results", help="results so far (round,team1,team2,winner)")
    live.add_argument(
        "--used-boosters", default="", help='boosters already spent, e.g. "Hellcase, Carry"'
    )
    return parser


def _with_rates(
    args: argparse.Namespace, pool: PlayerPool, sims: PlayerSims, rng: np.random.Generator
) -> PlayerSims:
    if not args.booster_rates:
        return sims
    return calibrate_boosters(sims, load_booster_rates(args.booster_rates, pool), rng)


@dataclass(frozen=True)
class World:
    evaluator: Evaluator
    cutoffs: np.ndarray


def _world(args: argparse.Namespace, pool: PlayerPool, n: int, seed: int) -> World:
    stage_file = load_stage(args.stage)
    rng = np.random.default_rng(seed)
    outcomes = simulate_map_outcomes(
        stage_file, n, rng, spread=args.spread, sims_per_draw=args.sims_per_draw
    )
    evaluator = Evaluator(
        _with_rates(
            args,
            pool,
            simulate_players(outcomes, pool, rng, team_card_weight=args.team_card_weight),
            rng,
        )
    )
    lineups = sample_field(pool, ownership(pool), args.field, args.budget, args.max_per_team, rng)
    return World(evaluator, tier_cutoffs(field_scores(evaluator, lineups, rng)))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _stats_line(stats: UpsideStats) -> str:
    tiers = {round(100 * f): stats.tiers[j] for j, (f, _) in enumerate(SEASON_TIERS)}
    return (
        f"E[points] {stats.expected:.1f} (SD {stats.sd:.1f})   P(top 1%) {_pct(tiers[1])}   "
        f"P(top 5%) {_pct(tiers[5])}   P(top 10%) {_pct(tiers[10])}   "
        f"P(top 50%) {_pct(tiers[50])}   E[season points] {stats.season:.2f}"
    )


def _plan_lines(plan: Plan, pool: PlayerPool, world: World) -> list[str]:
    ev = world.evaluator
    width = max(len(pool.names[p]) for p in plan.players)
    team_w = max(len(pool.team_names[p]) for p in plan.players)
    lines = [
        f"  {'Player':<{width}}  {'Team':<{team_w}}  {'Price':>9}  {'Role':<14}  "
        f"{'Base':>6}  {'Role pts':>8}  {'Boosts':>6}"
    ]
    boost_by_pos = [0.0] * len(plan.players)
    for b, pos, rnd in plan.boosters:
        boost_by_pos[pos] += 5 * float(ev.booster_prob[plan.players[pos], rnd, b])
    for pos, (p, role) in enumerate(zip(plan.players, plan.roles, strict=True)):
        lines.append(
            f"  {pool.names[p]:<{width}}  {pool.team_names[p]:<{team_w}}  "
            f"${int(pool.price[p]):>8,}  {ROLE_NAMES[role]:<14}  {ev.base_mean[p]:>6.1f}  "
            f"{ev.role_mean[p, role]:>8.1f}  {boost_by_pos[pos]:>6.1f}"
        )
    cost = int(pool.price[list(plan.players)].sum())
    lines.append(f"  Cost ${cost:,}; expected {plan.expected:.1f} points (search sims)")
    for rnd in range(N_ROUNDS):
        used = [
            f"{BOOSTER_NAMES[b]} -> {pool.names[plan.players[pos]]}"
            for b, pos, r in plan.boosters
            if r == rnd
        ]
        lines.append(f"  Round {rnd + 1} boosters: " + ("; ".join(used) if used else "none"))
    return lines


def _player_table(pool: PlayerPool, world: World, top: int) -> list[str]:
    ev = world.evaluator
    order = np.argsort(-ev.player_value)[:top]
    width = max(len(pool.names[p]) for p in order)
    lines = [
        f"  {'Player':<{width}}  {'Team':<18}  {'Price':>9}  {'Value':>6}  {'Base':>6}  "
        f"{'Best role':<14}  {'Pts per $10k':>12}"
    ]
    for p in order:
        role = int(np.argmax(ev.role_mean[p]))
        lines.append(
            f"  {pool.names[p]:<{width}}  {pool.team_names[p]:<18}  ${int(pool.price[p]):>8,}  "
            f"{ev.player_value[p]:>6.1f}  {ev.base_mean[p]:>6.1f}  {ROLE_NAMES[role]:<14}  "
            f"{10_000 * ev.player_value[p] / pool.price[p]:>12.2f}"
        )
    return lines


class _Error(Exception):
    pass


def _round_report(
    args: argparse.Namespace, stage_file: StageFile, pool: PlayerPool, lineup: tuple[int, ...]
) -> str:
    k = args.round
    if not 1 <= k <= N_ROUNDS:
        raise _Error(f"--round must be 1..{N_ROUNDS}")
    teams = stage_file.stage.teams
    if k > 1 and not args.results:
        raise _Error("--round 2 and later need --results with the rounds played so far")
    done = played_matches(args.results, teams, k) if k > 1 else []
    fixed = [(teams.index(m.winner), teams.index(m.loser)) for m in done]
    rng = np.random.default_rng(args.seed)
    outcomes = simulate_map_outcomes(
        stage_file,
        args.sims,
        rng,
        spread=args.spread,
        sims_per_draw=args.sims_per_draw,
        fixed=fixed,
    )
    sims = simulate_players(
        outcomes, pool, rng, keep_rounds=True, team_card_weight=args.team_card_weight
    )
    sims = _with_rates(args, pool, sims, rng)
    used = booster_indices(args.used_boosters)
    plan = plan_round(sims, lineup, k - 1, used)
    alive = teams_alive(sims, lineup, k - 1)
    lines = [
        f"pickem-fantasy: round {k} plan, {args.sims:,} sims with {len(done)} results fixed; "
        f"{len(used)} booster(s) already used",
        f"  {'Player':<14}  {'Plays':>6}  {'Role':<14}  {'E[role]':>7}  Booster (P success)",
    ]
    now = {pos: (b, p) for b, pos, p in plan.boosters_now}
    for pos, player in enumerate(lineup):
        booster = f"{BOOSTER_NAMES[now[pos][0]]} ({_pct(now[pos][1])})" if pos in now else "-"
        lines.append(
            f"  {pool.names[player]:<14}  {_pct(float(alive[pos])):>6}  "
            f"{ROLE_NAMES[plan.roles[pos]]:<14}  {plan.role_expected[pos]:>7.2f}  {booster}"
        )
    if plan.boosters_later:
        lines.append("  Tentative later boosters (re-plan each round):")
        for b, pos, rnd, p in plan.boosters_later:
            lines.append(
                f"    Round {rnd + 1}: {BOOSTER_NAMES[b]} -> {pool.names[lineup[pos]]} ({_pct(p)})"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_sims = args.eval_sims or args.sims
    try:
        stage_file = load_stage(args.stage)
        pool = load_players(args.players, stage_file.stage.teams)
        own = None
        if args.lineup:
            own = tuple(pool.index(name.strip()) for name in args.lineup.split(","))
            problems = check_lineup(pool, own, args.budget, args.max_per_team)
            if problems:
                raise _Error("--lineup " + "; ".join(problems))
        if args.round is not None:
            if own is None:
                raise _Error("--round needs --lineup")
            print(_round_report(args, stage_file, pool, own))
            return 0
        search = _world(args, pool, args.sims, args.seed)
        holdout = _world(args, pool, eval_sims, args.seed + 1_000_003)
        best_e = max_expected(search.evaluator, pool, args.budget, args.max_per_team)[0]
        rng = np.random.default_rng(args.seed + 7)
        best_u = max_upside(
            search.evaluator,
            pool,
            search.cutoffs,
            args.budget,
            args.max_per_team,
            rng,
            objective=args.objective,
            starts=[best_e.players],
            restarts=args.restarts,
        )
    except (OSError, ValueError, _Error) as exc:
        print(f"pickem-fantasy: error: {exc}", file=sys.stderr)
        return 2

    out = [
        f"pickem-fantasy: {stage_file.name}: {args.sims:,} search sims, {eval_sims:,} holdout "
        f"sims, field of {args.field:,} lineups per batch",
        f"Budget ${args.budget:,}, at most {args.max_per_team} per team. Numbers below are on the "
        "holdout sims.",
    ]
    if pool.estimated:
        shown = ", ".join(pool.estimated[:8]) + (" ..." if len(pool.estimated) > 8 else "")
        out.append(
            f"Warning: {len(pool.estimated)} player(s) have no card stats, so their stats are "
            f"estimated from price ({shown})."
        )
    sections: list[tuple[str, Plan]] = [
        ("Max expected points", best_e),
        (
            f"High upside (objective: {args.objective}; {best_u.evaluated:,} lineups tried)",
            best_u.plan,
        ),
    ]
    if own is not None:
        sections.append(("Your lineup", search.evaluator.plan(own)))
    for title, plan in sections:
        holdout_plan = holdout.evaluator.plan(plan.players)
        stats = upside(holdout.evaluator.scores(holdout_plan), holdout.cutoffs)
        out += ["", title, *_plan_lines(plan, pool, search), "  " + _stats_line(stats)]
    out += [
        "",
        f"Players by value (base + best role + booster share), top {args.top_players}:",
        *_player_table(pool, search, args.top_players),
        "",
        "Roles and boosters are pre-lock plans; both can be changed before each round.",
    ]
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
