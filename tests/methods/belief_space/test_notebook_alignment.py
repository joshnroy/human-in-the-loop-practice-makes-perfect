"""Notebook semantics for the training clock, resampling and defaults."""

import argparse

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.competence_inference import (
    BayesianSkillBelief,
    InferenceConfig,
    create_bayesian_prior,
)
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    SKILL_BELIEF_MODELS,
    make_default_tossing3d_belief,
    refit_belief_state,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.practice_makes_perfect.cli import Tossing3DPomdpCli

TOSS_MODEL = SKILL_BELIEF_MODELS[Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS]


def _toss(*, state: Tossing3DBeliefState) -> BayesianSkillBelief:
    belief = state.skill_beliefs[TOSS_SKILL]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


def _train(*, state: Tossing3DBeliefState, labels: list[bool]) -> Tossing3DBeliefState:
    for success in labels:
        state = TOSS_MODEL.observe_training_example(state=state, success=success)
    return state


@pytest.mark.parametrize("advance_cycle", [False, True])
def test_training_clock_counts_every_attempt_including_one_class_cycles(
    *, advance_cycle: bool
) -> None:
    state = make_default_tossing3d_belief(model="global_curve", engine="grid")
    state = _train(state=state, labels=[False, False, False])
    state = refit_belief_state(state=state, advance_cycle=advance_cycle)
    assert _toss(state=state).total_training_examples == 3
    state = _train(state=state, labels=[True, False])
    state = refit_belief_state(state=state, advance_cycle=advance_cycle)
    assert _toss(state=state).total_training_examples == 5


def _particles(*, count: int = 256) -> BayesianSkillBelief:
    return create_bayesian_prior(
        model="local_trend", engine="particle", seed=3, num_particles=count
    )


def test_particle_outcomes_never_resample_within_a_cycle() -> None:
    belief = _particles()
    for _ in range(40):
        belief = belief.condition_outcome(success=False)
    assert belief.resampling_count == 0
    _, weights = belief.arrays()
    assert 1.0 / float(weights @ weights) < 0.5 * belief.state_count


def test_real_boundary_systematically_resamples_once_then_predicts() -> None:
    belief = _particles()
    for success in (True, False, False):
        belief = belief.condition_outcome(success=success)
    values, weights = belief.arrays()
    advanced = belief.advance_cycle(training_examples=2)
    assert advanced.resampling_count == 1
    parents = advanced.ancestors()
    counts = np.bincount(parents, minlength=belief.state_count)
    # Systematic resampling: every count is floor or ceil of N * w.
    assert np.all(np.abs(counts - belief.state_count * weights) < 1.0)
    _, new_weights = advanced.arrays()
    assert np.allclose(new_weights, 1.0 / belief.state_count)
    idle = advanced.advance_cycle(training_examples=0)
    assert idle.resampling_count == 2


def test_search_forecast_does_not_resample_and_is_deterministic() -> None:
    belief = _particles()
    for success in (True, False, False):
        belief = belief.condition_outcome(success=success)
    first = belief.refit(training_examples=0)
    second = belief.refit(training_examples=0)
    assert first == second
    assert first.resampling_count == belief.resampling_count
    assert first.state_weights == belief.state_weights
    assert first.latent_values != belief.latent_values


def test_zero_example_forecast_is_the_notebook_predict_not_the_identity() -> None:
    belief = create_bayesian_prior(model="global_curve", engine="grid", seed=0, num_particles=1)
    for success in (True, True, False):
        belief = belief.condition_outcome(success=success)
    forecast = belief.refit(training_examples=0)
    assert forecast is not belief
    assert forecast.state_weights == belief.advance_cycle(training_examples=0).state_weights
    assert forecast.mean_competence() != pytest.approx(belief.mean_competence())


def test_config_has_no_ess_threshold() -> None:
    assert "resample_ess_fraction" not in InferenceConfig.model_fields


def test_particle_count_default_is_the_notebooks() -> None:
    assert Tossing3DPomdpMethod.model_fields["pomdp_num_particles"].default == 20000


def test_learning_rate_decay_default_is_the_notebooks_per_data_point_rho() -> None:
    assert Tossing3DPomdpMethod.model_fields["pomdp_learning_rate_decay"].default == 0.9
    assert Tossing3DPomdpMethod.model_fields["pomdp_learning_time_scale"].default == 1.0
    assert InferenceConfig().learning_rate_decay == 0.9
    parser = argparse.ArgumentParser()
    Tossing3DPomdpCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    assert args.pomdp_learning_rate_decay == 0.9
    assert args.pomdp_learning_time_scale == 1.0
    assert args.pomdp_num_particles == 20000
