import numpy as np
import pytest

from pickem.playoffs import (
    FORMATS,
    MatchProbabilities,
    Objective,
    PlayoffPick,
    best_picks,
    consistent_picks,
    favourites_pick,
    independent_picks,
    outcome_probabilities,
    parse_picks,
    score_picks,
)
from pickem.playoffs.picks import MODES, correct_counts
from pickem.probs import bo1_matrix

SINGAPORE = FORMATS["singapore"]
BRACKET = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")  # bracket index = seed - 1
CHALK = PlayoffPick(qf=(0, 3, 1, 2), sf=(0, 1), gf=0)


def probs_from(strengths: np.ndarray) -> np.ndarray:
    p = MatchProbabilities.from_maps(bo1_matrix(strengths)[None])
    result: np.ndarray = outcome_probabilities(p, SINGAPORE).mean(axis=0)
    return result


def test_pick_sets() -> None:
    consistent, independent = consistent_picks(), independent_picks()
    assert len(consistent) == len(set(consistent)) == 128
    assert len(independent) == len(set(independent)) == 2048
    assert all(p.consistent for p in consistent)
    assert set(consistent) <= set(independent)
    assert sum(p.consistent for p in independent) == 128


def test_correct_counts_on_a_known_outcome() -> None:
    real = 1 << 6  # every QF and SF to the lower seed, final to the first-listed team
    wrong_way = PlayoffPick(qf=(7, 4, 6, 5), sf=(4, 5), gf=4)
    counts = correct_counts([CHALK, wrong_way])
    assert (
        counts.qf[:, real].tolist(),
        counts.sf[:, real].tolist(),
        counts.gf[:, real].tolist(),
    ) == (
        [0, 4],
        [0, 2],
        [0, 1],
    )


def test_objective_thresholds() -> None:
    certain = np.zeros(256)
    certain[1 << 6] = 1.0  # one outcome for sure
    pick = PlayoffPick(qf=(7, 4, 0, 0), sf=(4, 1), gf=4)  # 2 QF right, 1 SF right, GF right
    pick = PlayoffPick(qf=(7, 4, 1, 2), sf=(4, 1), gf=4)
    (score,) = score_picks([pick], certain, Objective())
    assert score.p_objective == 1.0 and score.expected_qf == 2 and score.expected_sf == 1
    (strict,) = score_picks([pick], certain, Objective(min_qf=3))
    assert strict.p_objective == 0.0
    with pytest.raises(ValueError, match="thresholds"):
        Objective(min_sf=3)


def test_higher_seed_always_winning_makes_chalk_certain() -> None:
    strengths = np.linspace(40, -40, 8)  # bracket index 0 is seed 1
    probs = probs_from(strengths)
    for mode in MODES:
        best = best_picks(mode, probs, probs, Objective())[0]
        assert best.pick == CHALK and best.p_objective == pytest.approx(1.0)
        assert favourites_pick(mode, probs) == CHALK


def test_independent_is_never_worse_and_favourites_follow_the_rules() -> None:
    rng = np.random.default_rng(9)
    for _ in range(5):
        probs = probs_from(rng.normal(scale=0.7, size=8))
        consistent = best_picks("consistent", probs, probs, Objective(), top=1)[0]
        independent = best_picks("independent", probs, probs, Objective(), top=1)[0]
        assert independent.p_objective >= consistent.p_objective - 1e-12
        assert favourites_pick("consistent", probs).consistent
        all_consistent = score_picks(consistent_picks(), probs, Objective())
        assert consistent.p_objective == pytest.approx(max(s.p_objective for s in all_consistent))


def test_standard_error_over_draws() -> None:
    rng = np.random.default_rng(4)
    draws = np.stack(
        [
            outcome_probabilities(
                MatchProbabilities.from_maps(bo1_matrix(rng.normal(size=8))[None]), SINGAPORE
            )[0]
            for _ in range(50)
        ]
    )
    (with_se,) = score_picks([CHALK], draws.mean(axis=0), Objective(), draws)
    (without,) = score_picks([CHALK], draws.mean(axis=0), Objective())
    assert with_se.se > 0 and without.se == 0 and with_se.p_objective == without.p_objective


def test_parse_picks() -> None:
    pick = parse_picks("qf: S1, S4, S7, S3; sf: S4, S3; gf: s4", BRACKET)
    assert pick == PlayoffPick(qf=(0, 3, 6, 2), sf=(3, 2), gf=3) and pick.consistent
    loose = parse_picks("gf: S8; sf: S5, S6; qf: S1, S4, S2, S3", BRACKET)
    assert not loose.consistent


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("qf: S1, S8, S2, S3; sf: S1, S2; gf: S1", "same quarterfinal"),
        ("qf: S1, S4, S2, S3; sf: S1, S4; gf: S1", "same half"),
        ("qf: S1, S4, S2; sf: S1, S2; gf: S1", "each of the 4 quarterfinals"),
        ("qf: S1, S4, S2, S3; sf: S1, S2", "missing GF"),
        ("qf: S1, S4, S2, S3; sf: S1, S2; gf: Vitality", "unknown team"),
        ("qf: S1, S4, S2, S3; sf: S1, S2; gf: S1, S2", "exactly one grand final"),
    ],
)
def test_parse_picks_errors(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_picks(text, BRACKET)
