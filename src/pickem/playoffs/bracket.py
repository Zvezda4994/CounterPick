"""The 8-team single-elimination playoffs, seeded from Stage 3, evaluated exactly.

Rulebook (pinned commit, L204-L211 and L253-L254): playoff seeds are the final Stage 3 mid-stage
seeds; half A is 1 v 8 and 4 v 5, half B is 2 v 7 and 3 v 6, and the half winners meet in the
final. The rulebook text makes every match BO3 with no third-place match; PGL's Singapore 2026
playoffs (Liquipedia PGL/2026/Singapore/Playoffs) play a BO5 grand final and a BO3 third-place
decider, so the format is a parameter.

A bracket has 8 matches (4 QF, 2 SF, the final, the third-place match), so all 256 outcomes are
enumerated with exact probabilities. Strength uncertainty is handled by averaging the outcome
probabilities over draws of the teams' strengths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
from numpy.typing import NDArray

from pickem.data.stage_file import StageFile
from pickem.probs.next_stage import replay_from_liquipedia, replay_from_results
from pickem.probs.strength import StrengthModel, bo1_matrix, bo3_from_bo1, bo5_from_bo1

N_TEAMS = 8
QF_SEEDS = ((1, 8), (4, 5), (2, 7), (3, 6))
"""Quarterfinals in bracket order: QF 0 and 1 feed semifinal 0 (half A), QF 2 and 3 feed
semifinal 1 (half B)."""
HALVES = ((0, 7, 3, 4), (1, 6, 2, 5))
"""Bracket indices (seed - 1) of the teams in each half."""
QF_MATCHES, SF_MATCHES, FINAL, THIRD = (0, 1, 2, 3), (4, 5), 6, 7
N_MATCHES, N_OUTCOMES = 8, 256


@dataclass(frozen=True)
class PlayoffFormat:
    name: str
    final_best_of: int
    third_place: bool
    description: str

    def best_of(self, match: int) -> int:
        return self.final_best_of if match == FINAL else 3


FORMATS = {
    "singapore": PlayoffFormat(
        "singapore",
        5,
        True,
        "QF, SF and third-place match BO3, grand final BO5 (PGL Singapore 2026, per Liquipedia; "
        "the rulebook text has all BO3 and no third-place match)",
    ),
    "rulebook": PlayoffFormat(
        "rulebook", 3, False, "every match BO3, no third-place match (rulebook L204-L211)"
    ),
}


# --- outcomes ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutcomeTable:
    """Who plays and who wins in each of the 256 outcomes; teams are bracket indices."""

    bits: NDArray[np.bool_]  # (256, 8): the first-listed team won match k
    first: NDArray[np.intp]  # (256, 8)
    second: NDArray[np.intp]  # (256, 8)
    winner: NDArray[np.intp]  # (256, 8)
    loser: NDArray[np.intp]  # (256, 8)

    @property
    def qf_winners(self) -> NDArray[np.intp]:
        return self.winner[:, list(QF_MATCHES)]

    @property
    def sf_winners(self) -> NDArray[np.intp]:
        return self.winner[:, list(SF_MATCHES)]

    @property
    def champion(self) -> NDArray[np.intp]:
        return self.winner[:, FINAL]


@cache
def outcome_table() -> OutcomeTable:
    shape = (N_OUTCOMES, N_MATCHES)
    bits = np.zeros(shape, dtype=np.bool_)
    first = np.zeros(shape, dtype=np.intp)
    second = np.zeros(shape, dtype=np.intp)
    winner = np.zeros(shape, dtype=np.intp)
    loser = np.zeros(shape, dtype=np.intp)
    for o in range(N_OUTCOMES):
        for k in range(N_MATCHES):
            if k in QF_MATCHES:
                a, b = (s - 1 for s in QF_SEEDS[k])
            elif k in SF_MATCHES:
                feed = 2 * (k - SF_MATCHES[0])
                a, b = winner[o, feed], winner[o, feed + 1]
            elif k == FINAL:
                a, b = winner[o, SF_MATCHES[0]], winner[o, SF_MATCHES[1]]
            else:
                a, b = loser[o, SF_MATCHES[0]], loser[o, SF_MATCHES[1]]
            won = bool((o >> k) & 1)
            bits[o, k], first[o, k], second[o, k] = won, a, b
            winner[o, k], loser[o, k] = (a, b) if won else (b, a)
    return OutcomeTable(bits, first, second, winner, loser)


def outcome_index(bits: Sequence[bool]) -> int:
    return sum(int(b) << k for k, b in enumerate(bits))


# --- match probabilities ------------------------------------------------------------------


@dataclass(frozen=True)
class MatchProbabilities:
    """Per-draw win probabilities between the 8 bracket teams (row beats column)."""

    bo3: NDArray[np.float64]  # (draws, 8, 8)
    bo5: NDArray[np.float64]  # (draws, 8, 8)

    @property
    def n_draws(self) -> int:
        return int(self.bo3.shape[0])

    def for_match(self, fmt: PlayoffFormat, match: int) -> NDArray[np.float64]:
        return self.bo5 if fmt.best_of(match) == 5 else self.bo3

    @classmethod
    def from_maps(cls, p_map: NDArray[np.float64]) -> MatchProbabilities:
        """From per-map probabilities of shape (draws, 8, 8)."""
        return cls(bo3=bo3_from_bo1(p_map), bo5=bo5_from_bo1(p_map))


def playoff_seeds(
    stage_file: StageFile,
    *,
    results_csv: str | None = None,
    wikitext: str | None = None,
    aliases: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """Replay Stage 3's results; its final seeds 1-8 are the playoff seeds (rulebook L253-L254)."""
    stage = stage_file.stage
    by_seed = [stage.teams[int(i)] for i in stage.by_seed]
    if results_csv is not None:
        replay = replay_from_results(by_seed, results_csv)
    elif wikitext is not None:
        replay = replay_from_liquipedia(by_seed, wikitext, aliases)
    else:
        raise ValueError("give the Stage 3 results (CSV or Liquipedia)")
    return replay.advancing


def bracket_indices(stage_file: StageFile, bracket: Sequence[str]) -> list[int]:
    teams = stage_file.stage.teams
    missing = [t for t in bracket if t not in teams]
    if missing:
        raise ValueError(f"not in the stage file: {', '.join(missing)}")
    if len(bracket) != N_TEAMS or len(set(bracket)) != N_TEAMS:
        raise ValueError(f"the bracket needs {N_TEAMS} different teams")
    return [teams.index(t) for t in bracket]


def match_probabilities(
    stage_file: StageFile,
    bracket: Sequence[str],
    *,
    spread: float | None = None,
    draws: int = 20_000,
    rng: np.random.Generator | None = None,
) -> MatchProbabilities:
    """Match probabilities for the bracket teams, drawing strengths from the stage file's fit.

    Without a strength model (or at spread 0) there is one draw: per-map probabilities are the
    file's BO1 matrix and BO3 is the file's BO3 matrix.
    """
    idx = bracket_indices(stage_file, bracket)
    grid = np.ix_(idx, idx)
    model = stage_file.strength_model
    spread = (model.spread if model else 0.0) if spread is None else spread
    if spread < 0:
        raise ValueError("spread must be 0 or more")
    if model is None or spread == 0:
        if spread > 0:
            raise ValueError("the stage file has no strength model, so only spread 0 is possible")
        p_map = stage_file.stage.p_bo1[grid][None]
        return MatchProbabilities(bo3=stage_file.stage.p_bo3[grid][None], bo5=bo5_from_bo1(p_map))
    sub = StrengthModel(model.strengths[idx], model.covariance[grid], spread=spread)
    rng = np.random.default_rng(0) if rng is None else rng
    factor = sub.factor()
    strengths = np.stack([sub.sample(rng, spread, factor) for _ in range(draws)])
    p_map = np.stack([bo1_matrix(s) for s in strengths])
    return MatchProbabilities.from_maps(p_map)


def outcome_probabilities(probs: MatchProbabilities, fmt: PlayoffFormat) -> NDArray[np.float64]:
    """(draws, 256): exact probability of each outcome under each draw."""
    table = outcome_table()
    result = np.ones((probs.n_draws, N_OUTCOMES))
    for k in range(N_MATCHES):
        if k == THIRD and not fmt.third_place:
            result *= ~table.bits[:, k]  # no third-place match: only the 'not played' outcome
            continue
        p = probs.for_match(fmt, k)[:, table.first[:, k], table.second[:, k]]
        result *= np.where(table.bits[:, k], p, 1.0 - p)
    return result


@dataclass(frozen=True)
class TeamOdds:
    win_qf: NDArray[np.float64]
    reach_final: NDArray[np.float64]
    champion: NDArray[np.float64]
    third: NDArray[np.float64]
    """Third place if there is a third-place match, else losing in the semifinals."""


def team_odds(outcome_probs: NDArray[np.float64], fmt: PlayoffFormat) -> TeamOdds:
    """Per bracket team, from averaged outcome probabilities of shape (256,)."""
    table = outcome_table()
    teams = np.arange(N_TEAMS)

    def share(holders: NDArray[np.intp]) -> NDArray[np.float64]:
        hit = (holders[:, :, None] == teams[None, None, :]).any(axis=1)  # (256, 8)
        result: NDArray[np.float64] = outcome_probs @ hit
        return result

    third = table.winner[:, [THIRD]] if fmt.third_place else table.loser[:, list(SF_MATCHES)]
    return TeamOdds(
        win_qf=share(table.qf_winners),
        reach_final=share(table.sf_winners),
        champion=share(table.champion[:, None]),
        third=share(third),
    )


def simulate_playoffs(
    probs: MatchProbabilities, fmt: PlayoffFormat, n_sims: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Play the bracket match by match; returns each sim's outcome index (0..255).

    Each sim uses a random draw of strengths; the exact enumeration should match its frequencies.
    """
    draw = rng.integers(probs.n_draws, size=n_sims)
    winners = np.zeros((n_sims, N_MATCHES), dtype=np.intp)
    losers = np.zeros((n_sims, N_MATCHES), dtype=np.intp)
    index = np.zeros(n_sims, dtype=np.intp)
    for k in range(N_MATCHES):
        if k in QF_MATCHES:
            a = np.full(n_sims, QF_SEEDS[k][0] - 1)
            b = np.full(n_sims, QF_SEEDS[k][1] - 1)
        elif k in SF_MATCHES:
            feed = 2 * (k - SF_MATCHES[0])
            a, b = winners[:, feed], winners[:, feed + 1]
        elif k == FINAL:
            a, b = winners[:, SF_MATCHES[0]], winners[:, SF_MATCHES[1]]
        else:
            if not fmt.third_place:
                continue
            a, b = losers[:, SF_MATCHES[0]], losers[:, SF_MATCHES[1]]
        first_won = rng.random(n_sims) < probs.for_match(fmt, k)[draw, a, b]
        winners[:, k] = np.where(first_won, a, b)
        losers[:, k] = np.where(first_won, b, a)
        index |= first_won.astype(np.intp) << k
    return index
