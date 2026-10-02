"""Imagined S/F branches preserve the represented posterior without resampling noise."""

from typing import Literal

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state

Model = Literal["global_curve", "local_trend"]
Engine = Literal["particle", "grid"]


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
@pytest.mark.parametrize("success", [False, True])
def test_outcome_update_keeps_exact_weights_online_and_in_search(
    *, model: Model, engine: Engine, success: bool
) -> None:
    prior = create_bayesian_prior(
        model=model,
        engine=engine,
        seed=0,
        num_particles=128,
    )
    values, weights = prior.arrays()
    competence = prior.competence_values()
    likelihood = competence if success else 1 - competence
    expected = weights * likelihood
    expected /= expected.sum()
    before = prior.model_dump_json()
    # Online and imagined updates are the same call now: resampling happens only at a
    # real cycle boundary, so neither ever carries resampling noise.
    planned = prior.condition_outcome(success=success)
    assert np.array_equal(planned.arrays()[0], values)
    assert np.array_equal(planned.arrays()[1], expected)
    assert planned.parent_indices == prior.parent_indices
    assert planned.resampling_count == prior.resampling_count
    assert planned.cost_belief == prior.cost_belief
    assert planned.cycle_successes == int(success)
    assert planned.cycle_failures == int(not success)
    assert prior.model_dump_json() == before


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
@pytest.mark.parametrize("engine", ["particle", "grid"])
def test_fixed_controller_value_is_only_its_clock_and_cost_after_low_ess_history(
    *, model: Model, engine: Engine
) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=0,
        pomdp_competence_model=model,
        pomdp_inference_engine=engine,
        pomdp_num_particles=1024,
        pomdp_linear_cost_lambda=0.0003,
    )
    practice_model = method._pomdp_model  # noqa: SLF001
    for name in (PICK_SKILL, OPEN_GRIPPER_SKILL):
        ground = next(g for g in practice_model.ground_skills if g.skill.name == name)
        state = method.pomdp_state
        # This includes the measured B/particle counterexample: after five
        # failures, resampling a hypothetical child invented gain > action cost.
        for _ in range(6):
            outcomes = practice_model.outcomes(
                environment_state=make_tossing3d_search_state(
                    state=state, true_atoms=ground.preconditions
                ),
                state=state,
                action=ground,
            )
            value = sum(
                probability
                * practice_model.J(
                    belief_state=branch, summed_cost=branch.accumulated_cost, num_samples=1
                )
                for probability, branch, _ in outcomes
            )
            cost = sum(probability * branch.accumulated_cost for probability, branch, _ in outcomes)
            # A fixed controller's attempt still carries no S/F value in expectation;
            # its only effect beyond the cost is advancing its own training clock.
            pending = dict(state.pending_examples)
            pending[name] = pending.get(name, 0) + 1
            clock_only = practice_model.J(
                belief_state=state.model_copy(update={"pending_examples": pending}),
                summed_cost=0,
                num_samples=1,
            )
            assert value == pytest.approx(clock_only - 0.0003 * cost, abs=1e-14)
            before = state.skill_beliefs[name]
            assert isinstance(before, BayesianSkillBelief)
            for _, branch, _ in outcomes:
                after = branch.skill_beliefs[name]
                assert isinstance(after, BayesianSkillBelief)
                assert after.resampling_count == before.resampling_count
                assert after.latent_values == before.latent_values
            state = practice_model.observe_outcome(
                state=state, ground_skill=ground, success=False, was_random_exploration=False
            )
    assert env._backend is None  # noqa: SLF001


@pytest.mark.parametrize("model", ["global_curve", "local_trend"])
def test_toss_forecast_also_retains_support_without_resampling(*, model: Model) -> None:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        pomdp_competence_model=model,
        pomdp_num_particles=128,
    )
    practice_model = method._pomdp_model  # noqa: SLF001
    toss = next(g for g in practice_model.ground_skills if g.skill.name == TOSS_SKILL)
    prior = create_bayesian_prior(
        model=model,
        engine="particle",
        seed=0,
        num_particles=128,
    )
    beliefs = dict(method.pomdp_state.skill_beliefs)
    beliefs[TOSS_SKILL] = prior
    state = method.pomdp_state.model_copy(update={"skill_beliefs": beliefs})
    outcomes = practice_model.outcomes(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=toss.preconditions),
        state=state,
        action=toss,
    )
    assert len(outcomes) == 2
    for _, branch, atoms in outcomes:
        success = toss.add_effects <= atoms
        assert branch.skill_beliefs[TOSS_SKILL] == prior.condition_outcome(success=success)
