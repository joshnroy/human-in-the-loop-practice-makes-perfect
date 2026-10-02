"""Our grid engine against Tom's notebook, executed from a vendored copy of its cells.

`data/competence_models_notebook_cells.txt` holds the notebook's model, engine and toy
data cells verbatim, with the notebook's sha256 in its header. The notebook runs
unpatched: our grid now discretizes exactly as it does -- densities evaluated at the
grid points and normalized (Model A's cycle table, Model B's prior), Gaussian CDF bin
masses for Model B's transitions -- so the filtered posteriors themselves, and the
filtering, extrapolation, zero-example forecast and smoothing means, agree to rounding.
"""

from typing import Any

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.competence_inference import (
    CompetenceModel,
    create_bayesian_prior,
    smooth_history,
)

HORIZON = 10
SKILLS = ("Always fails", "Always succeeds", "Noisy, not improving", "Noisy, improving")
TOLERANCE = 1e-12


def _notebook_model(*, notebook: dict[str, Any], model: CompetenceModel) -> Any:
    return notebook["GlobalCurveModel" if model == "global_curve" else "LocalTrendModel"]()


def _notebook_run(
    *,
    notebook: dict[str, Any],
    model: CompetenceModel,
    cycles: list[tuple[int, int, int]],
) -> dict[str, np.ndarray]:
    engine = notebook["GridEngine"](_notebook_model(notebook=notebook, model=model))
    filtered, extrapolated, zero_forecast, posteriors = [], [], [], []
    for m, n, s in cycles:
        engine.update(m, n, s)
        posteriors.append(engine.belief)
        values, weights = engine.estimate()
        filtered.append(float(values @ weights))
        values, weights = engine.extrapolate(m + n + HORIZON)
        extrapolated.append(float(values @ weights))
        values, weights = engine.extrapolate(m)
        zero_forecast.append(float(values @ weights))
    smoothed = [float(values @ weights) for values, weights in engine.history()]
    return {
        "filtered": np.array(filtered),
        "extrapolated": np.array(extrapolated),
        "zero_forecast": np.array(zero_forecast),
        "smoothed": np.array(smoothed),
        "posterior": np.array(posteriors),
    }


def _our_run(
    *, model: CompetenceModel, cycles: list[tuple[int, int, int]]
) -> dict[str, np.ndarray]:
    belief = create_bayesian_prior(model=model, engine="grid", seed=0, num_particles=1)
    filtered, extrapolated, zero_forecast, history, posteriors = [], [], [], [], []
    for m, n, s in cycles:
        assert belief.total_training_examples == m
        for index in range(n):
            belief = belief.condition_outcome(success=index < s)
        posteriors.append(belief.arrays()[1])
        filtered.append(belief.mean_competence())
        extrapolated.append(belief.refit(training_examples=n + HORIZON).mean_competence())
        zero_forecast.append(belief.refit(training_examples=0).mean_competence())
        history.append(belief)
        belief = belief.advance_cycle(training_examples=n)
    smoothed = [cycle.smoothed_competence for cycle in smooth_history(history=history)]
    return {
        "filtered": np.array(filtered),
        "extrapolated": np.array(extrapolated),
        "zero_forecast": np.array(zero_forecast),
        "smoothed": np.array(smoothed),
        "posterior": np.array(posteriors),
    }


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("skill", SKILLS)
def test_grid_engine_matches_the_notebook(
    *, notebook: dict[str, Any], model: CompetenceModel, skill: str
) -> None:
    cycles = notebook["DATA"][skill][0]
    expected = _notebook_run(notebook=notebook, model=model, cycles=cycles)
    actual = _our_run(model=model, cycles=cycles)
    for key, values in expected.items():
        np.testing.assert_allclose(actual[key], values, rtol=0, atol=TOLERANCE, err_msg=key)
