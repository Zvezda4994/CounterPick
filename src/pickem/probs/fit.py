"""Joint Bradley-Terry fit of team strengths from odds, with VRS points as a weak prior.

Round-1 pairs are disjoint, so match odds only fix each pair's internal gap; outrights and the
VRS prior tie the pairs together. Everything is one penalized least-squares problem (a MAP fit)
on the log-odds scale:

    round 1   logit p_hat(i beats j)  ~ s_i - s_j                       sd sigma_match
    qualify   logit q_hat_i           ~ logit Q_i(s)                    sd sigma_qualify
    win Major log share_i             ~ beta s_i - logsumexp(beta s)    sd sigma_major
    prior     s_i ~ N(b * (VRS_i - mean) / 100, prior_sd^2), b ~ N(.4, .25^2), beta ~ N(3, 2^2)

Q_i(s), the chance to reach 3 wins in this Swiss, has no closed form. The fit uses a smooth
stand-in (each team plays average opponents until 3 wins or 3 losses), measures the real Q_i by
simulating the stage at the fitted strengths, shifts the stand-in by the gap and refits, until
the shifts settle. Uncertainty is the Laplace approximation (J^T J)^-1 at the optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import expit, logit, logsumexp

from pickem.probs.margin import METHODS, Method
from pickem.probs.odds_csv import (
    N_TEAMS,
    MarketProbabilities,
    OddsSheet,
    fair_probabilities,
    r1_opponent,
)
from pickem.probs.strength import StrengthModel, stage_from_strengths
from pickem.swiss import simulate_swiss

_EPS = 1e-4


@dataclass(frozen=True)
class FitConfig:
    method: Method = "power"
    sigma_match: float = 0.15
    sigma_qualify: float = 0.25
    sigma_major: float = 0.5
    prior_sd: float = 0.5
    vrs_slope_mean: float = 0.4
    """Prior mean of b, in logit per 100 VRS points."""
    vrs_slope_sd: float = 0.25
    beta_mean: float = 3.0
    beta_sd: float = 2.0
    fit_sims: int = 20_000
    """Sims per qualify-correction round (same random numbers every round)."""
    max_rounds: int = 8
    tolerance: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        positive = {
            "sigma_match": self.sigma_match,
            "sigma_qualify": self.sigma_qualify,
            "sigma_major": self.sigma_major,
            "prior_sd": self.prior_sd,
            "vrs_slope_sd": self.vrs_slope_sd,
            "beta_sd": self.beta_sd,
            "tolerance": self.tolerance,
        }
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        if self.fit_sims < 100 or self.max_rounds < 1:
            raise ValueError("fit_sims must be at least 100 and max_rounds at least 1")


@dataclass(frozen=True)
class StrengthFit:
    """Fit results in seed order (``teams[i]`` is seed ``i + 1``)."""

    teams: tuple[str, ...]
    vrs_points: NDArray[np.float64]
    strengths: NDArray[np.float64]
    covariance: NDArray[np.float64]
    vrs_slope: float
    vrs_slope_sd: float
    beta: float | None
    beta_sd: float | None
    market: MarketProbabilities
    model_r1: NDArray[np.float64]
    """P(win round 1) under the fit."""
    model_qualify: NDArray[np.float64]
    """P(qualify) from simulating the stage at the fitted strengths."""
    model_win_share: NDArray[np.float64]
    """Share of P(win the Major) among teams with win odds (NaN for the rest)."""
    correction_rounds: int
    converged: bool
    residual_rms: dict[str, float]
    """Root-mean-square residual per market, in units of that market's SD."""

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(1, N_TEAMS + 1))

    @property
    def sd(self) -> NDArray[np.float64]:
        result: NDArray[np.float64] = np.sqrt(np.diag(self.covariance))
        return result

    def strength_model(self, spread: float = 1.0, sims_per_draw: int = 1000) -> StrengthModel:
        return StrengthModel(self.strengths, self.covariance, spread, sims_per_draw)


def surrogate_qualify(s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth stand-in for P(3 wins before 3 losses) against the average other team."""
    others = (s.sum() - s) / (len(s) - 1)
    p = expit(s - others)
    q = 1.0 - p
    result: NDArray[np.float64] = p**3 * (1.0 + 3.0 * q + 6.0 * q**2)
    return result


def simulated_qualify(
    teams: tuple[str, ...], strengths: NDArray[np.float64], n_sims: int, seed: int
) -> NDArray[np.float64]:
    stage = stage_from_strengths(teams, tuple(range(1, len(teams) + 1)), strengths)
    result = simulate_swiss(stage, n_sims, np.random.default_rng(seed))
    q: NDArray[np.float64] = (result.wins == 3).mean(axis=0, dtype=np.float64)
    return q


def fit_strengths(sheet: OddsSheet, config: FitConfig | None = None) -> StrengthFit:
    config = FitConfig() if config is None else config
    market = fair_probabilities(sheet, config.method)
    z = (sheet.vrs_points - sheet.vrs_points.mean()) / 100.0

    pairs = np.array([i for i in range(N_TEAMS // 2) if not np.isnan(market.r1[i])], dtype=int)
    opponents = np.array([r1_opponent(i) for i in pairs], dtype=int)
    q_idx = np.flatnonzero(~np.isnan(market.qualify))
    w_idx = np.flatnonzero(~np.isnan(market.win_share))
    use_beta = len(w_idx) >= 2

    r1_target = logit(np.clip(market.r1[pairs], _EPS, 1 - _EPS))
    q_target = logit(np.clip(market.qualify[q_idx], _EPS, 1 - _EPS))
    w_target = np.log(market.win_share[w_idx])

    def unpack(theta: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float]:
        return theta[:N_TEAMS], float(theta[N_TEAMS]), float(theta[-1]) if use_beta else 0.0

    def residual_parts(
        theta: NDArray[np.float64], qualify_logit: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        s, b, beta = unpack(theta)
        parts = {
            "round 1": (r1_target - (s[pairs] - s[opponents])) / config.sigma_match,
            "qualify": (q_target - qualify_logit[q_idx]) / config.sigma_qualify,
            "prior": np.concatenate(
                [(s - b * z) / config.prior_sd, [(b - config.vrs_slope_mean) / config.vrs_slope_sd]]
            ),
        }
        if use_beta:
            scaled = beta * s[w_idx]
            parts["win Major"] = (w_target - (scaled - logsumexp(scaled))) / config.sigma_major
            parts["prior"] = np.append(parts["prior"], (beta - config.beta_mean) / config.beta_sd)
        return parts

    def residuals(theta: NDArray[np.float64], shift: NDArray[np.float64]) -> NDArray[np.float64]:
        s = theta[:N_TEAMS]
        qualify_logit = logit(surrogate_qualify(s)) + shift
        return np.concatenate(list(residual_parts(theta, qualify_logit).values()))

    x = np.concatenate(
        [config.vrs_slope_mean * z, [config.vrs_slope_mean]]
        + ([[config.beta_mean]] if use_beta else [])
    )
    shift = np.zeros(N_TEAMS)
    rounds, converged = 0, True
    while True:
        solution = least_squares(
            residuals, x, args=(shift,), method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12
        )
        x = solution.x
        rounds += 1
        s = x[:N_TEAMS]
        q_sim = simulated_qualify(sheet.teams, s, config.fit_sims, config.seed)
        if q_idx.size == 0:
            break
        clipped = np.clip(q_sim, 0.5 / config.fit_sims, 1 - 0.5 / config.fit_sims)
        new_shift = logit(clipped) - logit(surrogate_qualify(s))
        change = float(np.max(np.abs(new_shift - shift)[q_idx]))
        shift = new_shift
        if change < config.tolerance:
            break
        if rounds >= config.max_rounds:
            converged = False
            break

    jac = np.asarray(solution.jac, dtype=np.float64)  # dense: 18 parameters at most
    covariance_full = np.linalg.inv(jac.T @ jac)
    s, b, beta = unpack(x)
    clipped = np.clip(q_sim, 0.5 / config.fit_sims, 1 - 0.5 / config.fit_sims)
    final_parts = residual_parts(x, logit(clipped))
    rms = {
        name: float(np.sqrt(np.mean(values**2)))
        for name, values in final_parts.items()
        if name != "prior" and values.size
    }

    win_share = np.full(N_TEAMS, np.nan)
    if use_beta:
        scaled = beta * s[w_idx]
        win_share[w_idx] = np.exp(scaled - logsumexp(scaled))
    cov_s = covariance_full[:N_TEAMS, :N_TEAMS]
    return StrengthFit(
        teams=sheet.teams,
        vrs_points=sheet.vrs_points,
        strengths=s.copy(),
        covariance=(cov_s + cov_s.T) / 2.0,
        vrs_slope=b,
        vrs_slope_sd=float(np.sqrt(covariance_full[N_TEAMS, N_TEAMS])),
        beta=beta if use_beta else None,
        beta_sd=float(np.sqrt(covariance_full[-1, -1])) if use_beta else None,
        market=market,
        model_r1=np.asarray(
            expit(s - s[[r1_opponent(i) for i in range(N_TEAMS)]]), dtype=np.float64
        ),
        model_qualify=q_sim,
        model_win_share=win_share,
        correction_rounds=rounds,
        converged=converged,
        residual_rms=rms,
    )
