"""pickem-cards: a per-stage leaderboard of Pick'Em cards, scored before lock, judged after.

    pickem-cards freeze  --name NAME --stage data/stage1.json [--odds ODDS.csv]
                         [--add-optimizer] [--add-chalk]
    pickem-cards score   --name NAME [--rescore CARD ...]
    pickem-cards results --name NAME (--results CSV | --liquipedia PAGE)
    pickem-cards render  --name NAME [--readme README.md]

Layout (all tracked): cards/NAME.csv (you edit), snapshots/NAME.json and snapshots/NAME_odds.csv
(frozen at lock), leaderboards/NAME.json (structured data) and leaderboards/NAME.md (generated).
Scores are computed once per card on fixed sims and never rewritten, so pre-lock numbers stay
pre-lock numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pickem.data.liquipedia import fetch_wikitext, page_url
from pickem.data.stage_file import StageFile, load_stage
from pickem.optimizer.analysis import (
    RECORDS,
    approach_label,
    difference_path,
    overlap,
    record_distribution,
    single_outcome,
    swing_matches,
)
from pickem.optimizer.card import (
    ADVANCE,
    CATEGORY_NAMES,
    N_PICKS,
    OH_THREE,
    TARGET,
    THREE_OH,
    Card,
    Outcomes,
    correct_counts,
    evaluate,
)
from pickem.optimizer.cards_file import CardEntry, append_card, load_cards, picks_by_category
from pickem.optimizer.search import optimize
from pickem.probs.next_stage import replay_from_liquipedia, replay_from_results
from pickem.probs.uncertain import simulate_with_uncertainty
from pickem.swiss import SwissResult, simulate_swiss

BOARD_VERSION = 1
OPTIMIZER_SEED, OPTIMIZER_SIMS = 0, 100_000  # same as pickem-optimize's defaults
SCORE_SEED, SCORE_SIMS = 1, 100_000  # independent of the optimizer's search sims
SWING_SEED, SWING_SIMS = 2, 4000
OPTIMAL_NAME, CHALK_NAME = "Optimizer #1", "Seed chalk"
SHOW_ORDER = (THREE_OH, OH_THREE, ADVANCE)
DASH = chr(0x2013)  # en dash, shown for results that are not in yet
_CATEGORY = {CATEGORY_NAMES[c]: c for c in SHOW_ORDER}


@dataclass(frozen=True)
class Paths:
    root: Path
    name: str

    @property
    def cards(self) -> Path:
        return self.root / "cards" / f"{self.name}.csv"

    @property
    def snapshot(self) -> Path:
        return self.root / "snapshots" / f"{self.name}.json"

    @property
    def odds(self) -> Path:
        return self.root / "snapshots" / f"{self.name}_odds.csv"

    @property
    def board(self) -> Path:
        return self.root / "leaderboards" / f"{self.name}.json"

    @property
    def markdown(self) -> Path:
        return self.root / "leaderboards" / f"{self.name}.md"

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_commit(root: Path | None = None) -> str | None:
    """Short commit of the code doing the scoring (the repo this module lives in, unless ``root``
    is given), with +dirty if tracked files are modified."""
    root = Path(__file__).resolve().parent if root is None else root
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}+dirty" if dirty else head


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def simulate_stage(stage_file: StageFile, n: int, rng: np.random.Generator) -> SwissResult:
    """The stage at its own spread, exactly as pickem-optimize simulates it by default."""
    model = stage_file.strength_model
    if model is None or model.spread == 0:
        return simulate_swiss(stage_file.stage, n, rng)
    return simulate_with_uncertainty(stage_file.stage, model, n, rng)


def optimizer_card(stage_file: StageFile, sims: int = OPTIMIZER_SIMS) -> Card:
    """pickem-optimize's #1 card for this stage file (same seed, sims and search settings)."""
    rng = np.random.default_rng(OPTIMIZER_SEED)
    search = Outcomes.from_result(simulate_stage(stage_file, sims, rng))
    return optimize(search, rng=rng).best.card


def chalk_card(stage_file: StageFile) -> Card:
    """Seeds 1-2 to go 3-0, seeds 15-16 to go 0-3, seeds 3-8 to advance."""
    s = [int(t) for t in stage_file.stage.by_seed]
    return Card(three_oh=(s[0], s[1]), oh_three=(s[14], s[15]), advance=tuple(s[2:8]))


def card_from_picks(picks: dict[str, list[str]], teams: Sequence[str]) -> Card:
    return Card(
        three_oh=tuple(teams.index(t) for t in picks["3-0"]),
        oh_three=tuple(teams.index(t) for t in picks["0-3"]),
        advance=tuple(teams.index(t) for t in picks["advance"]),
    )


# --- freeze -------------------------------------------------------------------------------


def freeze(
    paths: Paths,
    stage_path: Path,
    odds_path: Path | None,
    *,
    add_optimizer: bool = False,
    add_chalk: bool = False,
    optimizer_sims: int = OPTIMIZER_SIMS,
) -> list[str]:
    if paths.snapshot.exists():
        raise FileExistsError(f"{paths.snapshot} already exists; a snapshot is never replaced")
    if odds_path is not None and paths.odds.exists():
        raise FileExistsError(f"{paths.odds} already exists; a snapshot is never replaced")
    stage_file = load_stage(stage_path)
    teams = stage_file.stage.teams
    planned = [OPTIMAL_NAME] * add_optimizer + [CHALK_NAME] * add_chalk
    if planned and paths.cards.exists():
        taken = {e.name.casefold() for e in load_cards(paths.cards, teams)}
        clash = [n for n in planned if n.casefold() in taken]
        if clash:
            raise ValueError(f"{paths.cards} already has {', '.join(clash)}")

    data = json.loads(Path(stage_path).read_text(encoding="utf-8"))
    data["frozen"] = {
        "at": _now(),
        "source": str(stage_path),
        "source_sha256": sha256(Path(stage_path)),
        "odds": paths.relative(paths.odds) if odds_path is not None else None,
        "code_commit": code_commit(),
    }
    paths.snapshot.parent.mkdir(parents=True, exist_ok=True)
    paths.snapshot.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    done = [f"froze {stage_path} -> {paths.relative(paths.snapshot)}"]
    if odds_path is not None:
        shutil.copyfile(odds_path, paths.odds)
        done.append(f"copied {odds_path} -> {paths.relative(paths.odds)}")
    if add_optimizer:
        card = optimizer_card(stage_file, optimizer_sims)
        append_card(paths.cards, CardEntry(OPTIMAL_NAME, "optimizer", "", card), teams)
        done.append(f"added {OPTIMAL_NAME} to {paths.relative(paths.cards)}")
    if add_chalk:
        append_card(
            paths.cards, CardEntry(CHALK_NAME, "seed chalk", "", chalk_card(stage_file)), teams
        )
        done.append(f"added {CHALK_NAME} to {paths.relative(paths.cards)}")
    return done


# --- board --------------------------------------------------------------------------------


def load_board(paths: Paths) -> dict[str, Any]:
    board: dict[str, Any] = json.loads(paths.board.read_text(encoding="utf-8"))
    if board.get("version") != BOARD_VERSION:
        raise ValueError(f"{paths.board}: unknown board version {board.get('version')!r}")
    return board


def save_board(paths: Paths, board: dict[str, Any]) -> None:
    paths.board.parent.mkdir(parents=True, exist_ok=True)
    paths.board.write_text(json.dumps(board, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    paths.markdown.write_text(render_markdown(board), encoding="utf-8")


def _frozen_stage(paths: Paths, board: dict[str, Any] | None) -> StageFile:
    if not paths.snapshot.exists():
        raise FileNotFoundError(f"{paths.snapshot} does not exist; run pickem-cards freeze first")
    if board is not None and board["snapshot"]["sha256"] != sha256(paths.snapshot):
        raise ValueError(
            f"{paths.snapshot} has changed since it was scored; the leaderboard only uses the "
            "snapshot frozen at lock"
        )
    return load_stage(paths.snapshot)


def pre_analysis(
    card: Card, optimal: Card, chalk: Card, outcomes: Outcomes, teams: Sequence[str]
) -> dict[str, Any]:
    stats = evaluate(card, outcomes)
    p = outcomes.probabilities
    identical, other_slot = overlap(card, optimal)
    label, parts = approach_label(card, p, optimal, chalk)
    return {
        "p5": stats.p5,
        "p6": stats.p6,
        "expected": stats.expected,
        "se5": stats.se5,
        "distribution": [int(x) for x in stats.distribution],
        "picks": [
            {"category": CATEGORY_NAMES[c], "team": teams[t], "p_correct": float(p[c][t])}
            for c in SHOW_ORDER
            for t in card.groups[c]
        ],
        "overlap": {"identical": identical, "other_slot": other_slot},
        "differences": [
            {"change": d.change, "gain": d.gain}
            for d in difference_path(card, optimal, outcomes, teams)
        ],
        "label": label,
        "label_parts": list(parts),
    }


def score(
    paths: Paths,
    *,
    sims: int | None = None,
    seed: int | None = None,
    rescore: Sequence[str] = (),
    optimizer_sims: int = OPTIMIZER_SIMS,
) -> list[str]:
    board = load_board(paths) if paths.board.exists() else None
    stage_file = _frozen_stage(paths, board)
    stage = stage_file.stage
    teams = stage.teams
    entries = load_cards(paths.cards, teams)
    notes: list[str] = []
    commit = code_commit()
    model = stage_file.strength_model

    if board is None:
        snapshot = json.loads(paths.snapshot.read_text(encoding="utf-8"))
        board = {
            "version": BOARD_VERSION,
            "name": paths.name,
            "snapshot": {
                "path": paths.relative(paths.snapshot),
                "sha256": sha256(paths.snapshot),
                "frozen": snapshot.get("frozen"),
            },
            "scoring": {
                "sims": sims or SCORE_SIMS,
                "seed": SCORE_SEED if seed is None else seed,
                "spread": model.spread if model else 0.0,
                "sims_per_draw": model.sims_per_draw if model else None,
                "optimizer": {"seed": OPTIMIZER_SEED, "sims": optimizer_sims},
                "code_commit": commit,
                "scored_at": _now(),
            },
            "optimal": None,
            "teams": [],
            "cards": [],
            "results": None,
        }
    else:
        scoring = board["scoring"]
        for label, given, used in (
            ("sims", sims, scoring["sims"]),
            ("seed", seed, scoring["seed"]),
        ):
            if given is not None and given != used:
                raise ValueError(f"this board was scored with {label}={used}; keep it the same")
        if scoring["code_commit"] != commit:
            notes.append(
                f"warning: code is at {commit}, the board was scored at {scoring['code_commit']}; "
                "new cards may not be scored exactly like the old ones"
            )

    scoring = board["scoring"]
    rng = np.random.default_rng(scoring["seed"])
    result = simulate_stage(stage_file, scoring["sims"], rng)
    outcomes = Outcomes.from_result(result)

    if board["optimal"] is None:
        best = optimizer_card(stage_file, scoring["optimizer"]["sims"])
        board["optimal"] = {"name": OPTIMAL_NAME, "picks": picks_by_category(best, teams)}
    optimal = card_from_picks(board["optimal"]["picks"], teams)
    if not board["teams"]:
        p = outcomes.probabilities
        records = record_distribution(result)
        board["teams"] = [
            {
                "team": teams[t],
                "seed": stage.seeds[t],
                "p": {
                    "3-0": float(p[THREE_OH][t]),
                    "advance": float(p[ADVANCE][t]),
                    "0-3": float(p[OH_THREE][t]),
                },
                "records": {r: float(records[t, k]) for k, r in enumerate(RECORDS)},
                "actual_record": None,
                "p_actual_record": None,
            }
            for t in range(len(teams))
        ]

    chalk = chalk_card(stage_file)
    wanted = {name.casefold() for name in rescore}
    position = {c["name"].casefold(): i for i, c in enumerate(board["cards"])}
    for entry in entries:
        key = entry.name.casefold()
        picks = picks_by_category(entry.card, teams)
        if key in position and key not in wanted:
            stored = board["cards"][position[key]]
            if stored["picks"] != picks:
                raise ValueError(
                    f"{entry.name}'s picks changed since it was scored; pre-lock scores are "
                    f"never rewritten silently (use --rescore {entry.name!r} to replace it)"
                )
            continue
        scored = {
            "name": entry.name,
            "source": entry.source,
            "link": entry.link,
            "picks": picks,
            "pre": pre_analysis(entry.card, optimal, chalk, outcomes, teams),
            "post": None,
        }
        if key in position:
            board["cards"][position[key]] = scored
            notes.append(f"rescored {entry.name} (its results, if any, need pickem-cards results)")
        else:
            board["cards"].append(scored)
            notes.append(f"scored {entry.name}")
        if entry.source == "optimizer" and entry.card != optimal:
            notes.append(f"warning: {entry.name} (optimizer) differs from the board's optimal card")
    missing = sorted(set(position) - {e.name.casefold() for e in entries})
    if missing:
        notes.append(
            f"note: {len(missing)} scored card(s) are no longer in the cards file and stay on "
            "the board"
        )
    save_board(paths, board)
    return notes


def add_results(
    paths: Paths,
    *,
    results_csv: Path | None = None,
    liquipedia: str | None = None,
    aliases: dict[str, str] | None = None,
    swing_sims: int = SWING_SIMS,
) -> list[str]:
    board = load_board(paths)
    stage_file = _frozen_stage(paths, board)
    stage = stage_file.stage
    teams = stage.teams
    by_seed = [teams[int(i)] for i in stage.by_seed]
    if results_csv is not None:
        replay = replay_from_results(by_seed, results_csv)
        source = str(results_csv)
    elif liquipedia is not None:
        replay = replay_from_liquipedia(by_seed, fetch_wikitext(liquipedia), aliases)
        source = page_url(liquipedia)
    else:
        raise ValueError("give --results or --liquipedia")

    order = [replay.teams.index(t) for t in teams]
    wins = [replay.wins[i] for i in order]
    losses = [replay.losses[i] for i in order]
    actual = single_outcome(teams, wins, losses)
    for entry in board["teams"]:
        t = teams.index(entry["team"])
        record = f"{wins[t]}-{losses[t]}"
        entry["actual_record"] = record
        entry["p_actual_record"] = entry["records"][record]

    cards = [card_from_picks(c["picks"], teams) for c in board["cards"]]
    counts = [int(correct_counts(card, actual)[0]) for card in cards]
    cleared = [n >= TARGET for n in counts]
    swings = swing_matches(stage, replay.rounds, cards, cleared, swing_sims, SWING_SEED)
    for entry, n, did_clear, card_swings in zip(
        board["cards"], counts, cleared, swings, strict=True
    ):
        pre = entry["pre"]
        dist = pre["distribution"]
        total = sum(dist)
        picks = []
        for pick in pre["picks"]:
            c, t = _CATEGORY[pick["category"]], teams.index(pick["team"])
            hit = bool(actual.hit[c, 0, t])
            picks.append(
                {
                    **pick,
                    "correct": hit,
                    "p_actual": pick["p_correct"] if hit else 1 - pick["p_correct"],
                }
            )
        entry["post"] = {
            "correct": n,
            "cleared": did_clear,
            "picks": picks,
            "p_exact": dist[n] / total,
            "p_at_least": sum(dist[n:]) / total,
            "p_cleared_outcome": pre["p5"] if did_clear else 1 - pre["p5"],
            "swings": [asdict(s) for s in card_swings],
        }
    board["results"] = {
        "source": source,
        "added_at": _now(),
        "swing_sims": swing_sims,
        "swing_seed": SWING_SEED,
    }
    save_board(paths, board)
    return [f"added results from {source} to {len(cards)} card(s)"]


# --- markdown -----------------------------------------------------------------------------


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _card_cell(card: dict[str, Any]) -> str:
    name = _cell(card["name"])
    return f"[{name}]({card['link']})" if card["link"] else name


def _ranked_cards(board: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(board["cards"], key=lambda c: (-c["pre"]["p5"], c["name"].casefold()))


def leaderboard_table(board: dict[str, Any]) -> list[str]:
    lines = [
        "| # | Card | Source | Approach | Overlap | P(≥5) | P(≥6) | E[correct] "
        "| Correct | Cleared |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, card in enumerate(_ranked_cards(board), start=1):
        pre, post = card["pre"], card["post"]
        correct = str(post["correct"]) if post else DASH
        cleared = ("✓" if post["cleared"] else "✗") if post else DASH
        lines.append(
            f"| {rank} | {_card_cell(card)} | {card['source']} | {_cell(pre['label'])} | "
            f"{pre['overlap']['identical']}/{N_PICKS} | {_pct(pre['p5'])} | {_pct(pre['p6'])} | "
            f"{pre['expected']:.2f} | {correct} | {cleared} |"
        )
    return lines


def summary(board: dict[str, Any]) -> list[str]:
    judged = [c for c in board["cards"] if c["post"] is not None]
    if not judged:
        return ["Results are not in yet."]

    def average(cards: list[dict[str, Any]]) -> str:
        return _pct(sum(c["pre"]["p5"] for c in cards) / len(cards)) if cards else "n/a"

    cleared = [c for c in judged if c["post"]["cleared"]]
    missed = [c for c in judged if not c["post"]["cleared"]]
    expected = sum(c["pre"]["p5"] for c in judged)
    brier = sum((c["pre"]["p5"] - c["post"]["cleared"]) ** 2 for c in judged) / len(judged)
    return [
        f"Cards that cleared: {len(cleared)}, average P(≥5) {average(cleared)}. "
        f"Cards that didn't: {len(missed)}, average P(≥5) {average(missed)}.",
        "",
        f"Expected clears (sum of P(≥5)): {expected:.1f}; actual clears: {len(cleared)}. "
        f"Brier score of P(≥5): {brier:.3f}. With this few cards these are anecdotes, not a test "
        "of the model.",
    ]


def render_markdown(board: dict[str, Any]) -> str:
    snapshot, scoring = board["snapshot"], board["scoring"]
    frozen = snapshot.get("frozen") or {}
    spread = scoring["spread"]
    uncertainty = f"strength uncertainty (spread {spread:.2f})" if spread else "fixed probabilities"
    lines = [
        f"## Card leaderboard: {board['name']}",
        "",
        f"Scored before lock on `{snapshot['path']}` (frozen {frozen.get('at', 'unknown')}, "
        f"sha256 {snapshot['sha256'][:12]}), {scoring['sims']:,} sims with {uncertainty}, "
        f"seed {scoring['seed']}, code {scoring['code_commit'] or 'unknown'}. "
        f"Optimal card: {board['optimal']['name']}.",
        "",
    ]
    results = board["results"]
    if results:
        lines += [f"Results from {results['source']} (added {results['added_at']}).", ""]
    lines += [*leaderboard_table(board), "", *summary(board)]

    lines += [
        "",
        "### Teams: model vs actual",
        "",
        "| Seed | Team | P(3-0) | P(3-1/3-2) | P(0-3) | Actual | Model P(actual record) |",
        "|---|---|---|---|---|---|---|",
    ]
    for team in sorted(board["teams"], key=lambda t: t["seed"]):
        p = team["p"]
        actual = team["actual_record"] or DASH
        p_actual = _pct(team["p_actual_record"]) if team["p_actual_record"] is not None else DASH
        lines.append(
            f"| {team['seed']} | {_cell(team['team'])} | {_pct(p['3-0'])} | {_pct(p['advance'])} | "
            f"{_pct(p['0-3'])} | {actual} | {p_actual} |"
        )

    records = {t["team"]: t["actual_record"] for t in board["teams"]}
    lines += ["", "### Cards"]
    for card in _ranked_cards(board):
        lines += ["", *card_section(card, records)]
    return "\n".join(lines) + "\n"


def card_section(card: dict[str, Any], records: dict[str, str | None]) -> list[str]:
    pre, post = card["pre"], card["post"]
    link = f", [link]({card['link']})" if card["link"] else ""
    other = pre["overlap"]["other_slot"]
    lines = [
        f"#### {_cell(card['name'])} ({card['source']}{link})",
        "",
        f"**Approach:** {pre['label']}. **Overlap with the optimal card:** "
        f"{pre['overlap']['identical']}/{N_PICKS} identical"
        + (f", plus {other} team(s) in a different slot." if other else "."),
        "",
        "| Slot | Team | P(correct) | Result | Model P(that result) |",
        "|---|---|---|---|---|",
    ]
    post_picks = post["picks"] if post else [None] * len(pre["picks"])
    for pick, outcome in zip(pre["picks"], post_picks, strict=True):
        if outcome is None:
            result, p_actual = DASH, DASH
        else:
            mark = "✓" if outcome["correct"] else "✗"
            result, p_actual = f"{records[pick['team']]} {mark}", _pct(outcome["p_actual"])
        lines.append(
            f"| {pick['category']} | {_cell(pick['team'])} | {_pct(pick['p_correct'])} | "
            f"{result} | {p_actual} |"
        )

    lines.append("")
    if pre["differences"]:
        gap = sum(d["gain"] for d in pre["differences"])
        lines.append(
            "**Differences from the optimal card**, fixed one at a time. Each step makes the "
            "change that leaves the best card at that point; because picks interact, a later "
            "step can gain more than an earlier one. The gains add up to the full gap of "
            f"{100 * gap:+.1f} points of P(≥5):"
        )
        lines += [
            f"{i}. {_cell(d['change'])}: {100 * d['gain']:+.1f} points"
            for i, d in enumerate(pre["differences"], start=1)
        ]
    else:
        lines.append("**Identical to the optimal card.**")

    lines += [
        "",
        f"**Predicted:** P(≥5) {_pct(pre['p5'])} ± {100 * pre['se5']:.2f}, "
        f"P(≥6) {_pct(pre['p6'])}, E[correct] {pre['expected']:.2f}.",
    ]
    if post is None:
        return lines
    outcome = "cleared" if post["cleared"] else "did not clear"
    lines[-1] += (
        f" **Actual:** {post['correct']} correct, {outcome}. The model gave exactly "
        f"{post['correct']} correct a {_pct(post['p_exact'])} chance, and {post['correct']} or "
        f"more {_pct(post['p_at_least'])}."
    )
    lines += [
        "",
        "**Swing matches** (each real result reversed on its own; teams that really met keep "
        "their real result, new pairings use the model):",
    ]
    if not post["swings"]:
        lines.append("None: no single result would have changed whether this card cleared.")
    for s in post["swings"]:
        share = _pct(s["p_clear_if_flipped"])
        if post["cleared"]:
            effect = (
                "it would not have cleared"
                if s["certain"]
                else f"it clears in only {share} of replays"
            )
        else:
            effect = "it would have cleared" if s["certain"] else f"it clears in {share} of replays"
        lines.append(
            f"- Round {s['round']}: {_cell(s['team1'])} v {_cell(s['team2'])} "
            f"({_cell(s['winner'])} won). Reversed, {effect}."
        )
    return lines


def update_readme(readme: Path, board: dict[str, Any], paths: Paths) -> None:
    start, end = f"<!-- leaderboard:{board['name']} -->", f"<!-- /leaderboard:{board['name']} -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text or end not in text or text.index(start) > text.index(end):
        raise ValueError(
            f"{readme} needs the markers {start} and {end} where the leaderboard should go"
        )
    try:
        link = paths.markdown.resolve().relative_to(readme.resolve().parent).as_posix()
    except ValueError:
        link = paths.markdown.as_posix()
    body = [
        *leaderboard_table(board),
        "",
        *summary(board),
        "",
        f"Full analysis: [{link}]({link})",
    ]
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    readme.write_text(
        before + start + "\n" + "\n".join(body) + "\n" + end + after, encoding="utf-8"
    )


# --- CLI ----------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pickem-cards", description=(__doc__ or "").splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", required=True, help="stage name, e.g. singapore_stage1")
        p.add_argument(
            "--root", default=".", help="folder holding cards/, snapshots/, leaderboards/"
        )

    p = sub.add_parser("freeze", help="freeze the stage file (and odds sheet) at lock")
    common(p)
    p.add_argument("--stage", required=True, help="stage JSON to freeze, e.g. data/stage1.json")
    p.add_argument("--odds", help="odds sheet it was fitted from (copied alongside)")
    p.add_argument(
        "--add-optimizer", action="store_true", help=f"add '{OPTIMAL_NAME}' to the cards"
    )
    p.add_argument("--add-chalk", action="store_true", help=f"add '{CHALK_NAME}' to the cards")
    p.add_argument("--optimizer-sims", type=int, default=OPTIMIZER_SIMS, help=argparse.SUPPRESS)

    p = sub.add_parser("score", help="score new cards on the frozen snapshot")
    common(p)
    p.add_argument(
        "--sims", type=int, help=f"scoring sims (default {SCORE_SIMS:,}, fixed per board)"
    )
    p.add_argument("--seed", type=int, help=f"scoring seed (default {SCORE_SEED}, fixed per board)")
    p.add_argument(
        "--rescore", action="append", default=[], metavar="CARD", help="re-score this card"
    )
    p.add_argument("--optimizer-sims", type=int, default=OPTIMIZER_SIMS, help=argparse.SUPPRESS)

    p = sub.add_parser("results", help="add the actual results and post-stage analysis")
    common(p)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", help="results CSV (round,team1,team2,winner)")
    source.add_argument("--liquipedia", metavar="PAGE", help="e.g. PGL/2026/Singapore/Stage_1")
    p.add_argument("--alias", action="append", default=[], metavar="FROM=TO")
    p.add_argument("--swing-sims", type=int, default=SWING_SIMS, help="replays per reversed result")

    p = sub.add_parser("render", help="rewrite the markdown section (and README table)")
    common(p)
    p.add_argument("--readme", help="README to update between the leaderboard markers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = Paths(Path(args.root), args.name)
    try:
        if args.command == "freeze":
            notes = freeze(
                paths,
                Path(args.stage),
                Path(args.odds) if args.odds else None,
                add_optimizer=args.add_optimizer,
                add_chalk=args.add_chalk,
                optimizer_sims=args.optimizer_sims,
            )
        elif args.command == "score":
            notes = score(
                paths,
                sims=args.sims,
                seed=args.seed,
                rescore=args.rescore,
                optimizer_sims=args.optimizer_sims,
            )
        elif args.command == "results":
            notes = add_results(
                paths,
                results_csv=Path(args.results) if args.results else None,
                liquipedia=args.liquipedia,
                aliases=dict(item.partition("=")[::2] for item in args.alias),
                swing_sims=args.swing_sims,
            )
        else:
            board = load_board(paths)
            paths.markdown.write_text(render_markdown(board), encoding="utf-8")
            notes = [f"wrote {paths.relative(paths.markdown)}"]
            if args.readme:
                update_readme(Path(args.readme), board, paths)
                notes.append(f"updated the leaderboard table in {args.readme}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"pickem-cards: error: {exc}", file=sys.stderr)
        return 2
    print("\n".join(notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
