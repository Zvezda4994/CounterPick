"""Team strengths and the win-probability matrices they imply.

A strength is in logit units of one map: P(i beats j in a BO1) = sigmoid(s_i - s_j). A BO3 is
modelled as independent maps at that same probability: p^2 (3 - 2p).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from pickem.swiss import StageInput


def bo1_matrix(strengths: ArrayLike) -> NDArray[np.float64]:
    s = np.asarray(strengths, dtype=np.float64)
    return np.asarray(expit(s[:, None] - s[None, :]), dtype=np.float64)


def bo3_from_bo1(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Win a best-of-three when each map is won independently with probability p."""
    result: NDArray[np.float64] = p * p * (3.0 - 2.0 * p)
    return result


def stage_from_strengths(
    teams: tuple[str, ...], seeds: tuple[int, ...], strengths: ArrayLike
) -> StageInput:
    p1 = bo1_matrix(strengths)
    return StageInput(teams=teams, seeds=seeds, p_bo1=p1, p_bo3=bo3_from_bo1(p1))


@dataclass(frozen=True)
class StrengthModel:
    """Fitted strengths with their uncertainty, in the stage's team order."""

    strengths: NDArray[np.float64]
    covariance: NDArray[np.float64]
    spread: float = 1.0
    """Default multiplier on the fitted standard deviations when sampling (0 = fixed)."""
    sims_per_draw: int = 1000
    """Simulations that share one draw of strengths."""

    def __post_init__(self) -> None:
        s = np.array(self.strengths, dtype=np.float64)
        cov = np.array(self.covariance, dtype=np.float64)
        n = s.shape[0]
        if s.ndim != 1 or cov.shape != (n, n):
            raise ValueError("strengths must be a vector and covariance a matching square matrix")
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(cov))):
            raise ValueError("strengths and covariance must be finite")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        if float(np.min(np.linalg.eigvalsh(cov))) < -1e-9:
            raise ValueError("covariance must be positive semi-definite")
        if self.spread < 0:
            raise ValueError("spread must be 0 or more")
        if self.sims_per_draw < 1:
            raise ValueError("sims_per_draw must be at least 1")
        s.setflags(write=False)
        cov.setflags(write=False)
        object.__setattr__(self, "strengths", s)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "spread", float(self.spread))
        object.__setattr__(self, "sims_per_draw", int(self.sims_per_draw))

    @property
    def sd(self) -> NDArray[np.float64]:
        result: NDArray[np.float64] = np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))
        return result

    def factor(self) -> NDArray[np.float64]:
        """L with L @ L.T == covariance (eigen-based, so semi-definite matrices work too)."""
        values, vectors = np.linalg.eigh(self.covariance)
        result: NDArray[np.float64] = vectors * np.sqrt(np.clip(values, 0.0, None))
        return result

    def sample(
        self, rng: np.random.Generator, spread: float, factor: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """One draw of strengths: mean + spread * N(0, covariance)."""
        if spread == 0:
            return self.strengths.copy()
        f = self.factor() if factor is None else factor
        draw: NDArray[np.float64] = self.strengths + spread * (
            f @ rng.standard_normal(len(self.strengths))
        )
        return draw
