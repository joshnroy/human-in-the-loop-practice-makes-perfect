"""Our grid engine against Tom's notebook, executed from a vendored copy of its cells.

`data/competence_models_notebook_cells.txt` holds the notebook's model, engine and toy
data cells verbatim, with the notebook's sha256 in its header. The one deliberate
deviation, exact Beta bin integrals where the notebook evaluates pdfs at grid points
(Model A's cycle table, Model B's prior), is patched into the notebook so that
everything else -- filtering, extrapolation, the zero-example forecast and smoothing --
is compared exactly. `test_unpatched_discretization_gap_is_bounded` keeps that deviation
visible instead of silently absorbing it.
"""

from typing import Any

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.competence_inference import (
    CompetenceModel,
    InferenceConfig,
    create_bayesian_prior,
    smooth_history,
)
from hitl_pmp.methods.belief_space.competence_inference.models import (
    curve_cycle_mass,
    grid_prior,
)

HORIZON = 10
SKILLS = ("Always fails", "Always succeeds", "Noisy, not improving", "Noisy, improving")
TOLERANCE = 1e-9


def _notebook_model(*, notebook: dict[str, Any], model: CompetenceModel, patched: bool) -> Any:
    config = InferenceConfig()
    if model == "global_curve":
        instance = notebook["GlobalCurveModel"]()
        if patched:
            instance._cycle_prior_table = lambda m: curve_cycle_mass(  # noqa: SLF001
                config=config, training_examples=int(m)
            )
        return instance
    instance = notebook["LocalTrendModel"]()
    if patched:
        instance.grid_initial = lambda: grid_prior(model="local_trend", config=config)
    return instance


def _notebook_run(
    *,
    notebook: dict[str, Any],
    model: CompetenceModel,
    cycles: list[tuple[int, int, int]],
    patched: bool,
) -> dict[str, np.ndarray]:
    engine = notebook["GridEngine"](
        _notebook_model(notebook=notebook, model=model, patched=patched)
    )
    filtered, extrapolated, zero_forecast = [], [], []
    for m, n, s in cycles:
        engine.update(m, n, s)
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
    }


def _our_run(
    *, model: CompetenceModel, cycles: list[tuple[int, int, int]]
) -> dict[str, np.ndarray]:
    belief = create_bayesian_prior(model=model, engine="grid", seed=0, num_particles=1)
    filtered, extrapolated, zero_forecast, history = [], [], [], []
    for m, n, s in cycles:
        assert belief.total_training_examples == m
        for index in range(n):
            belief = belief.condition_outcome(success=index < s)
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
    }


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("skill", SKILLS)
def test_grid_engine_matches_the_notebook(
    *, notebook: dict[str, Any], model: CompetenceModel, skill: str
) -> None:
    cycles = notebook["DATA"][skill][0]
    expected = _notebook_run(notebook=notebook, model=model, cycles=cycles, patched=True)
    actual = _our_run(model=model, cycles=cycles)
    for key, values in expected.items():
        np.testing.assert_allclose(actual[key], values, rtol=0, atol=TOLERANCE, err_msg=key)


@pytest.mark.parametrize(("model", "bound"), [("global_curve", 5e-3), ("local_trend", 5e-2)])
def test_unpatched_discretization_gap_is_bounded(
    *, notebook: dict[str, Any], model: CompetenceModel, bound: float
) -> None:
    gaps = []
    for skill in SKILLS:
        cycles = notebook["DATA"][skill][0]
        expected = _notebook_run(notebook=notebook, model=model, cycles=cycles, patched=False)
        actual = _our_run(model=model, cycles=cycles)
        gaps.append(max(float(np.max(np.abs(actual[k] - v))) for k, v in expected.items()))
    assert 1e-6 < max(gaps) < bound
