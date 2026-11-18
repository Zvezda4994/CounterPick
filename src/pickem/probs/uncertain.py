"""Swiss simulations that carry uncertainty about the win probabilities themselves."""

from __future__ import annotations

import numpy as np

from pickem.probs.strength import StrengthModel, stage_from_strengths
from pickem.swiss import StageInput, SwissResult, concat_results, simulate_swiss


def simulate_with_uncertainty(
    stage: StageInput,
    model: StrengthModel,
    n_sims: int,
    rng: np.random.Generator,
    *,
    spread: float | None = None,
    sims_per_draw: int | None = None,
    record_matches: bool = False,
) -> SwissResult:
    """Simulate ``n_sims`` stages, drawing fresh team strengths every ``sims_per_draw`` sims.

    Each batch draws s ~ N(mean, spread^2 * covariance) and builds its own BO1/BO3 matrices, so
    the results mix uncertainty about the probabilities with match randomness. With
    ``spread == 0`` this is exactly ``simulate_swiss`` on ``stage``'s fixed matrices.
    """
    spread = model.spread if spread is None else spread
    per = model.sims_per_draw if sims_per_draw is None else sims_per_draw
    if spread < 0:
        raise ValueError("spread must be 0 or more")
    if per < 1:
        raise ValueError("sims_per_draw must be at least 1")
    if len(model.strengths) != len(stage.teams):
        raise ValueError("strength model and stage have different numbers of teams")
    if spread == 0:
        return simulate_swiss(stage, n_sims, rng, record_matches=record_matches)
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")

    factor = model.factor()
    batches = []
    for start in range(0, n_sims, per):
        strengths = model.sample(rng, spread, factor)
        draw = stage_from_strengths(stage.teams, stage.seeds, strengths)
        batches.append(
            simulate_swiss(draw, min(per, n_sims - start), rng, record_matches=record_matches)
        )
    return concat_results(batches)
