import numpy as np
import pytest

from pickem.probs import (
    StrengthModel,
    bo1_matrix,
    bo3_from_bo1,
    simulate_with_uncertainty,
    stage_from_strengths,
)
from pickem.swiss import StageInput, simulate_swiss

TEAMS = tuple(f"T{i:02d}" for i in range(16))
SEEDS = tuple(range(1, 17))
MEAN = np.linspace(0.6, -0.6, 16)


def model(spread: float = 1.0, per: int = 1000, sd: float = 0.2) -> StrengthModel:
    return StrengthModel(MEAN, np.eye(16) * sd**2, spread=spread, sims_per_draw=per)


def stage() -> StageInput:
    return stage_from_strengths(TEAMS, SEEDS, MEAN)


def test_matrices_follow_the_formulas() -> None:
    p1 = bo1_matrix(MEAN)
    p3 = bo3_from_bo1(p1)
    off = ~np.eye(16, dtype=bool)
    assert (p1 + p1.T)[off] == pytest.approx(np.ones(240))
    assert (p3 + p3.T)[off] == pytest.approx(np.ones(240))
    assert p1[0, 15] == pytest.approx(1 / (1 + np.exp(-1.2)))
    assert p3[0, 15] == pytest.approx(p1[0, 15] ** 2 * (3 - 2 * p1[0, 15]))
    assert np.all(np.abs(p3 - 0.5)[off] >= np.abs(p1 - 0.5)[off])  # BO3 favours the favourite


def test_spread_zero_is_exactly_the_fixed_simulator() -> None:
    a = simulate_with_uncertainty(stage(), model(), 3_000, np.random.default_rng(1), spread=0)
    b = simulate_swiss(stage(), 3_000, np.random.default_rng(1))
    assert (a.wins == b.wins).all() and (a.losses == b.losses).all()


def test_fresh_strengths_every_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    draws: list[np.ndarray] = []
    real = stage_from_strengths

    def spy(teams: tuple[str, ...], seeds: tuple[int, ...], strengths: np.ndarray) -> StageInput:
        draws.append(np.array(strengths))
        return real(teams, seeds, strengths)

    monkeypatch.setattr("pickem.probs.uncertain.stage_from_strengths", spy)
    result = simulate_with_uncertainty(stage(), model(per=10), 95, np.random.default_rng(2))
    assert result.n_sims == 95
    assert len(draws) == 10  # 9 batches of 10 and one of 5
    assert len({d.tobytes() for d in draws}) == 10


def test_swiss_invariants_hold_under_uncertainty() -> None:
    result = simulate_with_uncertainty(stage(), model(per=50), 2_000, np.random.default_rng(3))
    assert ((result.wins == 3).sum(axis=1) == 8).all()
    assert (result.wins.sum(axis=1) == 33).all()
    assert (((result.wins == 3) & (result.losses == 0)).sum(axis=1) == 2).all()


def test_uncertainty_flattens_the_favourite() -> None:
    """Averaging over uncertain strengths pulls extreme outcomes toward the middle."""
    n = 40_000
    fixed = simulate_with_uncertainty(stage(), model(), n, np.random.default_rng(4), spread=0)
    wide = simulate_with_uncertainty(
        stage(), model(sd=0.8), n, np.random.default_rng(4), spread=1.0
    )
    p_fixed = ((fixed.wins == 3) & (fixed.losses == 0)).mean(axis=0)
    p_wide = ((wide.wins == 3) & (wide.losses == 0)).mean(axis=0)
    assert p_wide[0] < p_fixed[0] - 0.01  # top seed goes 3-0 less often
    assert p_wide[15] > p_fixed[15] + 0.005  # bottom seed more often


def test_samples_match_the_model() -> None:
    cov = np.array([[0.04, 0.02], [0.02, 0.09]])
    m = StrengthModel(np.array([0.3, -0.3]), cov)
    rng = np.random.default_rng(5)
    draws = np.array([m.sample(rng, 1.0) for _ in range(20_000)])
    assert draws.mean(axis=0) == pytest.approx([0.3, -0.3], abs=0.01)
    assert np.cov(draws.T) == pytest.approx(cov, abs=0.005)
    f = m.factor()
    assert f @ f.T == pytest.approx(cov)
    assert m.sample(rng, 0.0).tolist() == [0.3, -0.3]


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.eye(3), "matching square"),
        (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "semi-definite"),
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), "finite"),
    ],
)
def test_strength_model_validation(covariance: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StrengthModel(np.zeros(2), covariance)


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="spread"):
        StrengthModel(np.zeros(2), np.eye(2), spread=-1)
    with pytest.raises(ValueError, match="sims_per_draw"):
        simulate_with_uncertainty(stage(), model(), 10, np.random.default_rng(), sims_per_draw=0)
    with pytest.raises(ValueError, match="different numbers"):
        simulate_with_uncertainty(
            stage(), StrengthModel(np.zeros(2), np.eye(2)), 10, np.random.default_rng()
        )
