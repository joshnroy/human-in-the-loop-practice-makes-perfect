"""The practice forecast follows the real sampler's fit and label lifecycle."""

from typing import Literal

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.types import Tossing3DState
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief, smooth_history
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.methods.belief_space.tossing3d_transition_model import make_tossing3d_search_state
from hitl_pmp.methods.belief_space.types.belief_state import (
    SamplerTrainingState,
    Tossing3DBeliefState,
)

Model = Literal["global_curve", "local_trend"]
Engine = Literal["particle", "grid"]
ARMS: tuple[tuple[Model, Engine], ...] = (
    ("global_curve", "particle"),
    ("global_curve", "grid"),
    ("local_trend", "particle"),
    ("local_trend", "grid"),
)


def _method(*, model: Model, engine: Engine) -> Tossing3DPomdpMethod:
    env = Tossing3DEnvironment(scene_bg=False)
    method = Tossing3DPomdpMethod(
        env=env,
        skill_provider=Tossing3DSkillProvider(env=env),
        seed=17,
        pomdp_competence_model=model,
        pomdp_inference_engine=engine,
        pomdp_num_particles=256,
        pomdp_grid_competence_bins=9,
        pomdp_grid_learning_rate_bins=5,
        pomdp_linear_cost_lambda=0.0003,
        sampler_max_train_iters=1,
    )
    env.current_state = Tossing3DState(
        data={obj: np.zeros(obj.type.dim) for obj in method.objects()},
        abstract_atoms=frozenset(),
    )
    return method


def _row(*, method: Tossing3DPomdpMethod, success: bool, random: bool = False) -> None:
    toss = next(g for g in method._pomdp_model.ground_skills if g.skill.name == TOSS_SKILL)  # noqa: SLF001
    method.observe_outcome(ground_skill=toss, success=success, was_random_exploration=random)
    method.observe_sampler_outcome(
        skill_name=TOSS_SKILL,
        param_dim=4,
        sampler_input=[1.0, float(success), 2.5, 0.0, 360.0, 500.0],
        success=success,
    )


def _toss_belief(*, state: Tossing3DBeliefState) -> BayesianSkillBelief:
    belief = state.skill_beliefs[TOSS_SKILL]
    assert isinstance(belief, BayesianSkillBelief)
    return belief


@pytest.mark.parametrize(("model", "engine"), ARMS)
@pytest.mark.parametrize("first_label", [False, True])
def test_one_class_cycles_preserve_policy_then_first_mixed_refit_credits_all_data(
    *, model: Model, engine: Engine, first_label: bool
) -> None:
    method = _method(model=model, engine=engine)
    for cycle in range(2):
        _row(method=method, success=first_label)
        _row(method=method, success=first_label)
        before = _toss_belief(state=method.pomdp_state)
        projected = _toss_belief(state=refit_belief_state(state=method.pomdp_state))
        assert projected is before
        method.end_cycle()
        after = _toss_belief(state=method.pomdp_state)
        # The real boundary applies the notebook's n = 0 noise step (the search
        # forecast above stayed the identity); one-class data still earns no
        # learning credit.
        assert (after.latent_values, after.state_weights) != (
            before.latent_values,
            before.state_weights,
        )
        assert after.total_training_examples == 0
        assert after.process_transition_count == cycle + 1
        training = method.pomdp_state.sampler_training[TOSS_SKILL]
        assert training.successes + training.failures == 2 * (cycle + 1)
        assert training.fitted_successes + training.fitted_failures == 2 * (cycle + 1)
        assert not training.fitted_mixed_classes
        assert (
            method.sampler(skill_name=TOSS_SKILL, param_dim=4).score_inputs(
                sampler_inputs=[[1, 0, 2.5, 0, 360, 500], [1, 1, 2.0, 0, 300, 550]]
            )
            == [float(first_label)] * 2
        )

    _row(method=method, success=not first_label, random=True)
    state = method.pomdp_state
    training = state.sampler_training[TOSS_SKILL]
    assert not training.fitted_mixed_classes
    assert training.refit_examples == 5
    projected = refit_belief_state(state=state)
    assert projected.sampler_training[TOSS_SKILL].fitted_mixed_classes
    assert _toss_belief(state=projected).total_training_examples == 5
    method.end_cycle()
    actual = _toss_belief(state=method.pomdp_state)
    predicted = _toss_belief(state=projected)
    assert actual == predicted
    assert actual.process_transition_count == 3
    assert method.sampler(skill_name=TOSS_SKILL, param_dim=4).num_observations == 5

    _row(method=method, success=False)
    method.end_cycle()
    assert _toss_belief(state=method.pomdp_state).total_training_examples == 6
    assert _toss_belief(state=method.pomdp_state).incoming_training_examples == 1
    assert method.env._backend is None  # noqa: SLF001


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_hypothetical_first_success_enables_refit_without_changing_current_sampler_mode(
    *, model: Model, engine: Engine
) -> None:
    method = _method(model=model, engine=engine)
    _row(method=method, success=False)
    method.end_cycle()
    state = method.pomdp_state
    practice_model = method._pomdp_model  # noqa: SLF001
    toss = next(g for g in practice_model.ground_skills if g.skill.name == TOSS_SKILL)
    search_state = make_tossing3d_search_state(state=state, true_atoms=toss.preconditions)
    outcomes = practice_model.outcomes(environment_state=search_state, state=state, action=toss)
    assert len(outcomes) == 2
    assert sum(p for p, _, _ in outcomes) == pytest.approx(1.0)
    prior = _toss_belief(state=state)
    for probability, branch, atoms in outcomes:
        success = toss.add_effects <= atoms
        assert probability == pytest.approx(
            prior.mean_competence() if success else 1 - prior.mean_competence()
        )
        belief = _toss_belief(state=branch)
        assert belief.cycle_successes + belief.cycle_failures == 1
        assert not branch.sampler_training[TOSS_SKILL].fitted_mixed_classes
        assert branch.sampler_training[TOSS_SKILL].refit_examples == (2 if success else 0)
        projected = refit_belief_state(state=branch)
        assert projected.sampler_training[TOSS_SKILL].fitted_mixed_classes == success
        assert _toss_belief(state=projected).total_training_examples == (2 if success else 0)
        if success:
            # The model forecasts a mixed fit only for deployment, not between
            # actions of this practice cycle. The current policy stays frozen.
            next_outcomes = practice_model.outcomes(
                environment_state=make_tossing3d_search_state(
                    state=branch, true_atoms=toss.preconditions
                ),
                state=branch,
                action=toss,
            )
            assert len(next_outcomes) == 2
            refitted_outcomes = practice_model.outcomes(
                environment_state=make_tossing3d_search_state(
                    state=projected, true_atoms=toss.preconditions
                ),
                state=projected,
                action=toss,
            )
            unconditioned_mass = sum(
                p
                for p, result, _ in refitted_outcomes
                if _toss_belief(state=result).cycle_successes
                + _toss_belief(state=result).cycle_failures
                == 0
            )
            assert unconditioned_mass == pytest.approx(method.exploration_epsilon)


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_forecast_j_is_pure_and_matches_real_refit(*, model: Model, engine: Engine) -> None:
    method = _method(model=model, engine=engine)
    _row(method=method, success=False)
    method.end_cycle()
    # A MIXED pending cycle: with effective examples > 0 the search refit and the
    # real boundary run the identical transition off the same noise stream, so the
    # forecast is exact. (At zero effective examples they now deliberately differ:
    # refit keeps the identity while the boundary applies the n = 0 noise step --
    # pinned in test_competence_2x2_integration.)
    _row(method=method, success=True)
    _row(method=method, success=False)
    sampler = method.sampler(skill_name=TOSS_SKILL, param_dim=4)
    before = method.pomdp_state.model_dump_json()
    rows = sampler.observed_inputs()
    scores = sampler.score_inputs(sampler_inputs=rows)
    forecast = method._pomdp_model.J(  # noqa: SLF001
        belief_state=method.pomdp_state, summed_cost=7.0, num_samples=1
    )
    assert method.pomdp_state.model_dump_json() == before
    assert sampler.observed_inputs() == rows
    assert sampler.score_inputs(sampler_inputs=rows) == scores
    method.end_cycle()
    actual = method._pomdp_model.J(  # noqa: SLF001
        belief_state=method.pomdp_state, summed_cost=7.0, num_samples=1
    )
    # The pending toss belief transitions identically on both paths, but the
    # real boundary also noise-steps every IDLE skill belief (search refit
    # keeps those at the zero-example identity -- the carve-out), so J now
    # agrees only to within those skills' process noise.
    assert actual == pytest.approx(forecast, abs=0.01)
    assert method.pomdp_state.accumulated_cost == 0


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_fixed_controllers_condition_sf_and_cost_without_learning_credit(
    *, model: Model, engine: Engine
) -> None:
    method = _method(model=model, engine=engine)
    for name in (PICK_SKILL, OPEN_GRIPPER_SKILL, RESET_SKILL):
        ground = next(g for g in method._pomdp_model.ground_skills if g.skill.name == name)  # noqa: SLF001
        before = method.pomdp_state.skill_beliefs[name]
        method.observe_outcome(ground_skill=ground, success=True)
        after = method.pomdp_state.skill_beliefs[name]
        assert isinstance(before, BayesianSkillBelief)
        assert isinstance(after, BayesianSkillBelief)
        assert after.mean_competence() > before.mean_competence()
        assert after.cost_belief != before.cost_belief
        assert method.pomdp_state.pending_examples.get(name, 0) == 0
    before_refit = dict(method.pomdp_state.skill_beliefs)
    method.end_cycle()
    for name in (PICK_SKILL, OPEN_GRIPPER_SKILL, RESET_SKILL):
        before = before_refit[name]
        after = method.pomdp_state.skill_beliefs[name]
        assert isinstance(before, BayesianSkillBelief)
        assert isinstance(after, BayesianSkillBelief)
        # Real-boundary n = 0 noise step: latents move, learning credit does not.
        assert (after.latent_values, after.state_weights) != (
            before.latent_values,
            before.state_weights,
        )
        assert after.total_training_examples == 0
        assert after.process_transition_count == 1


def test_sampler_lifecycle_is_in_clone_serialization_and_all_search_keys() -> None:
    method = _method(model="local_trend", engine="grid")
    original = method.pomdp_state
    model = method._pomdp_model  # noqa: SLF001
    states = [
        original.model_copy(update={"sampler_training": {TOSS_SKILL: training}})
        for training in (
            SamplerTrainingState(),
            SamplerTrainingState(successes=1),
            SamplerTrainingState(failures=1),
            SamplerTrainingState(successes=1, failures=1),
            SamplerTrainingState(successes=1, failures=1, fitted_successes=1, fitted_failures=1),
        )
    ]
    search_keys = []
    transition_keys = []
    for state in states:
        restored = Tossing3DBeliefState.model_validate_json(state.model_dump_json())
        assert restored == state
        search_state = make_tossing3d_search_state(state=state, true_atoms=frozenset())
        search_keys.append(
            model.search_cache_key(
                environment_state=search_state, belief_state=state, summed_cost=0, horizon=2
            )
        )
        transition_keys.append(model.transition_key(environment_state=search_state, cost=1))
    assert len(set(search_keys)) == len(set(transition_keys)) == len(states)
    assert original.sampler_training[TOSS_SKILL] == SamplerTrainingState()
    historical = original.model_dump(mode="json")
    historical.pop("sampler_training")
    assert Tossing3DBeliefState.model_validate(historical).sampler_training == {}
    with pytest.raises(ValidationError, match="cannot exceed"):
        SamplerTrainingState(successes=0, fitted_successes=1)


@pytest.mark.parametrize(("model", "engine"), ARMS)
def test_smoothing_spans_identity_cycles_and_first_deferred_transition(
    *, model: Model, engine: Engine
) -> None:
    method = _method(model=model, engine=engine)
    for success in (False, False, True, False):
        _row(method=method, success=success)
        method.end_cycle()
    history = method._belief_history[TOSS_SKILL]  # noqa: SLF001
    assert [belief.total_training_examples for belief in history] == [0, 0, 0, 3]
    assert [belief.incoming_training_examples for belief in history] == [0, 0, 0, 3]
    smoothed = smooth_history(history=history)
    assert len(smoothed) == 4
    assert all(np.isfinite(row.smoothed_competence) for row in smoothed)
    assert smoothed[-1].smoothed_competence == pytest.approx(history[-1].mean_competence())
