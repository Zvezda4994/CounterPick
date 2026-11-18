from itertools import product
from pathlib import Path

import numpy as np
import pytest

from pickem.data.stage_file import StageFile, load_stage, save_stage
from pickem.playoffs import (
    FORMATS,
    MatchProbabilities,
    PlayoffFormat,
    match_probabilities,
    outcome_probabilities,
    outcome_table,
    playoff_seeds,
    simulate_playoffs,
    team_odds,
)
from pickem.probs import StrengthModel, bo1_matrix, stage_from_strengths
from pickem.probs.strength import bo3_from_bo1, bo5_from_bo1
from tests.probs.test_next_stage import NAME, STAGES, results_csv

SINGAPORE, RULEBOOK = FORMATS["singapore"], FORMATS["rulebook"]


def series_win(p: float, wins_needed: int) -> float:
    """Count every map sequence directly."""
    total = 0.0
    for maps in product((True, False), repeat=2 * wins_needed - 1):
        won = lost = 0
        for m in maps:
            won, lost = won + m, lost + (not m)
            if won == wins_needed or lost == wins_needed:
                break
        if won == wins_needed:
            total += p**won * (1 - p) ** lost / 2 ** (2 * wins_needed - 1 - won - lost)
    return total


@pytest.mark.parametrize("p", [0.1, 0.35, 0.5, 0.62, 0.9])
def test_series_formulas(p: float) -> None:
    arr = np.array([p])
    assert bo3_from_bo1(arr)[0] == pytest.approx(series_win(p, 2))
    assert bo5_from_bo1(arr)[0] == pytest.approx(series_win(p, 3))
    assert bo5_from_bo1(arr)[0] + bo5_from_bo1(1 - arr)[0] == pytest.approx(1.0)
    if p > 0.5:
        assert p < bo3_from_bo1(arr)[0] < bo5_from_bo1(arr)[0]


def test_outcome_table_structure() -> None:
    table = outcome_table()
    assert table.bits.shape == (256, 8)
    assert len({tuple(row) for row in table.bits}) == 256
    assert (table.first[:, :4] == np.array([0, 3, 1, 2])).all()  # seeds 1, 4, 2, 3
    assert (table.second[:, :4] == np.array([7, 4, 6, 5])).all()  # seeds 8, 5, 7, 6
    assert (table.first[:, 4] == table.winner[:, 0]).all() and (
        table.second[:, 4] == table.winner[:, 1]
    ).all()
    assert (table.first[:, 6] == table.winner[:, 4]).all() and (
        table.second[:, 6] == table.winner[:, 5]
    ).all()
    assert (table.first[:, 7] == table.loser[:, 4]).all() and (
        table.second[:, 7] == table.loser[:, 5]
    ).all()


def random_probs(draws: int, seed: int) -> MatchProbabilities:
    rng = np.random.default_rng(seed)
    strengths = rng.normal(scale=0.6, size=(draws, 8))
    return MatchProbabilities.from_maps(np.stack([bo1_matrix(s) for s in strengths]))


@pytest.mark.parametrize("fmt", [SINGAPORE, RULEBOOK])
def test_outcome_probabilities_and_team_odds(fmt: PlayoffFormat) -> None:
    assert fmt in (SINGAPORE, RULEBOOK)
    probs = random_probs(40, seed=1)
    per_draw = outcome_probabilities(probs, fmt)
    assert per_draw.sum(axis=1) == pytest.approx(np.ones(40))
    if not fmt.third_place:
        assert (per_draw[:, outcome_table().bits[:, 7]] == 0).all()
    odds = team_odds(per_draw.mean(axis=0), fmt)
    assert odds.win_qf.sum() == pytest.approx(4)
    assert odds.reach_final.sum() == pytest.approx(2)
    assert odds.champion.sum() == pytest.approx(1)
    assert odds.third.sum() == pytest.approx(1 if fmt.third_place else 2)


@pytest.mark.parametrize(("draws", "fmt"), [(1, SINGAPORE), (30, SINGAPORE), (30, RULEBOOK)])
def test_monte_carlo_agrees_with_exact(draws: int, fmt: PlayoffFormat) -> None:
    probs = random_probs(draws, seed=2)
    exact = outcome_probabilities(probs, fmt).mean(axis=0)
    n = 1_000_000
    sims = simulate_playoffs(probs, fmt, n, np.random.default_rng(3))
    counts = np.bincount(sims, minlength=256)
    expected = n * exact
    # Cells with few expected hits break the normal approximation (and 256 cells are checked at
    # once), so: per-cell z-tests where the approximation holds, plus one chi-square over all
    # cells with at least 5 expected hits.
    big = expected >= 100
    z = (counts[big] - expected[big]) / np.sqrt(expected[big] * (1 - exact[big]))
    assert np.abs(z).max() < 5
    usable = expected >= 5
    chi2 = float((((counts[usable] - expected[usable]) ** 2) / expected[usable]).sum())
    df = int(usable.sum()) - 1
    assert chi2 < df + 6 * np.sqrt(2 * df)  # about 6 standard deviations above the mean
    assert counts[expected == 0].sum() == 0  # impossible outcomes never happen


TEAMS = tuple(f"Team {i:02d}" for i in range(1, 17))


def stage_file(spread: float) -> StageFile:
    strengths = np.linspace(0.8, -0.8, 16)
    return StageFile(
        "test",
        stage_from_strengths(TEAMS, tuple(range(1, 17)), strengths),
        StrengthModel(strengths, np.eye(16) * 0.04, spread=spread),
    )


def test_spread_zero_uses_the_file_matrices() -> None:
    sf = stage_file(1.0)
    bracket = TEAMS[:8]
    fixed = match_probabilities(sf, bracket, spread=0)
    assert fixed.n_draws == 1
    assert (fixed.bo3[0] == sf.stage.p_bo3[:8, :8]).all()
    assert fixed.bo5[0] == pytest.approx(bo5_from_bo1(sf.stage.p_bo1[:8, :8]))
    drawn = match_probabilities(sf, bracket, draws=500, rng=np.random.default_rng(0))
    assert drawn.n_draws == 500 and not np.allclose(drawn.bo3[0], drawn.bo3[1])
    with pytest.raises(ValueError, match="different teams"):
        match_probabilities(sf, [*TEAMS[:7], TEAMS[0]])


# --- Austin 2025: seeds from Stage 3 and the real bracket -----------------------------------
# Liquipedia BLAST/Major/2025/Austin/Playoffs: QF Spirit-MOUZ, Vitality-NaVi, FaZe-MongolZ,
# paiN-FURIA; SF MOUZ-Vitality, MongolZ-paiN; final Vitality-MongolZ (Vitality won).


def test_austin_stage3_seeds_and_real_bracket(tmp_path: Path) -> None:
    teams = tuple(NAME[a] for a in STAGES[3]["seeds"])
    path = tmp_path / "stage3.json"
    save_stage(
        path,
        StageFile("Austin Stage 3", stage_from_strengths(teams, tuple(range(1, 17)), np.zeros(16))),
    )
    bracket = playoff_seeds(load_stage(path), results_csv=str(results_csv(tmp_path, 3)))
    assert bracket == (
        "Spirit", "FURIA", "FaZe", "Natus Vincere", "Vitality", "The MongolZ", "paiN", "MOUZ",
    )  # fmt: skip

    table = outcome_table()
    quarterfinals = {
        frozenset((bracket[table.first[0, k]], bracket[table.second[0, k]])) for k in range(4)
    }
    assert quarterfinals == {
        frozenset(p)
        for p in (
            ("Spirit", "MOUZ"),
            ("Vitality", "Natus Vincere"),
            ("FaZe", "The MongolZ"),
            ("paiN", "FURIA"),
        )
    }
    # Real results: every QF and SF went to the second-listed team, the final to the first.
    real = 1 << 6
    names = [bracket[t] for t in table.winner[real]]
    assert names[:4] == ["MOUZ", "Vitality", "paiN", "The MongolZ"]
    assert {bracket[table.first[real, 4]], bracket[table.second[real, 4]]} == {"MOUZ", "Vitality"}
    assert {bracket[table.first[real, 5]], bracket[table.second[real, 5]]} == {
        "paiN",
        "The MongolZ",
    }
    assert bracket[table.champion[real]] == "Vitality"
    assert bracket[table.loser[real, 6]] == "The MongolZ"
    assert {bracket[table.loser[real, 4]], bracket[table.loser[real, 5]]} == {"MOUZ", "paiN"}
