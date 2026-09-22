"""Exact integration must preserve the deployment objective without sampling noise."""

from itertools import product
from typing import Literal

import numpy as np
import pytest

from hitl_pmp.methods.belief_space.competence_inference import create_bayesian_prior
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_deployment_model import evaluate_deployment_policies
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import SkillHypothesis, WeightedHypothesis
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief


def _belief(*, values: tuple[float, ...]) -> WeightedHypothesisBelief:
    return WeightedHypothesisBelief(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=value, learning_rate=0.07),
                probability=1 / len(values),
            )
            for value in values
        )
    )


def _state() -> Tossing3DBeliefState:
    return Tossing3DBeliefState(
        skill_beliefs={
            PICK_SKILL: _belief(values=(0.0, 0.2, 0.8, 1.0)),
            TOSS_SKILL: _belief(values=(0.1, 0.6, 1.0)),
            OPEN_GRIPPER_SKILL: _belief(values=(0.0, 0.7, 1.0)),
        },
        pending_examples={PICK_SKILL: 2, TOSS_SKILL: 1},
    )


@pytest.mark.parametrize("horizon", range(13))
@pytest.mark.parametrize("cost_lambda", [None, 0.0003])
def test_exact_objective_matches_exhaustive_deployment_models(
    *, horizon: int, cost_lambda: float | None
) -> None:
    model = Tossing3DPracticeModel(deployment_horizon=horizon, linear_cost_lambda=cost_lambda)
    state = _state()
    projected = refit_belief_state(state=state)
    expected = 0.0
    for pick, toss, opened in product(
        projected.skill_beliefs[PICK_SKILL].hypotheses,
        projected.skill_beliefs[TOSS_SKILL].hypotheses,
        projected.skill_beliefs[OPEN_GRIPPER_SKILL].hypotheses,
    ):
        theta = Tossing3DTheta(
            pick=pick.hypothesis, toss=toss.hypothesis, open_gripper=opened.hypothesis
        )
        expected += (
            pick.probability
            * toss.probability
            * opened.probability
            * model.G(policy_value=model.evaluate_policy(sampled_theta=theta), summed_cost=10)
        )

    # A one-sample request must have the same expectation as exhaustive integration.
    assert model.J(belief_state=state, summed_cost=10, num_samples=1) == pytest.approx(
        expected, abs=1e-13
    )


def test_extra_reset_cost_cannot_improve_unchanged_deployment_belief() -> None:
    model = Tossing3DPracticeModel(linear_cost_lambda=0.0003)
    state = _state()
    rng_before = model._rng.bit_generator.state
    original = state.model_dump_json()
    values = [
        model.J(belief_state=state, summed_cost=cost, num_samples=100) for cost in (5, 10, 15)
    ]
    np.testing.assert_allclose(np.diff(values), [-0.0015, -0.0015], atol=1e-14, rtol=0)
    assert model._rng.bit_generator.state == rng_before
    assert state.model_dump_json() == original


def test_exact_objective_retains_hard_cost_budget() -> None:
    assert Tossing3DPracticeModel().J(
        belief_state=_state(), summed_cost=20.001, num_samples=1
    ) == -float("inf")


@pytest.mark.parametrize("model_name", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_experiment_beliefs_match_cartesian_support_integration(
    *, model_name: Literal["global_curve", "local_trend"], engine: Literal["particle", "grid"]
) -> None:
    names = (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)
    beliefs = {
        name: create_bayesian_prior(model=model_name, engine=engine, seed=i, num_particles=32)
        .condition_outcome(success=True)
        .condition_outcome(success=False)
        for i, name in enumerate(names)
    }
    state = Tossing3DBeliefState(skill_beliefs=beliefs, pending_examples={TOSS_SKILL: 2})
    projected = refit_belief_state(state=state)
    values, weights = [], []
    for name in names:
        belief = projected.skill_beliefs[name]
        support, indexes = np.unique(belief.competence_values(), return_inverse=True)
        values.append(support)
        weights.append(np.bincount(indexes, weights=belief.arrays()[1]))
    pick, toss, opened = np.meshgrid(*values, indexing="ij")
    joint_weights = (
        weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
    )
    model = Tossing3DPracticeModel(deployment_horizon=8, linear_cost_lambda=0.0003)
    expected = (
        float(
            np.sum(
                joint_weights
                * evaluate_deployment_policies(
                    pick_competences=pick, toss_competences=toss, open_competences=opened, horizon=8
                )
            )
        )
        - 0.0003 * 15
    )
    assert model.J(belief_state=state, summed_cost=15, num_samples=1) == pytest.approx(
        expected, abs=1e-13
    )
