from itertools import product

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.expectimax import solve_belief_space_expectimax
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    condition_skill_belief,
    make_default_tossing3d_belief,
    mean_competence,
    refit_belief_state,
    refit_skill_belief,
)
from hitl_pmp.methods.belief_space.tossing3d_particle_filter import (
    belief_arrays,
    belief_from_arrays,
    condition_particle_belief,
    make_particle_belief_prior,
    particle_filter_diagnostics,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    make_tossing3d_search_state,
    render_atoms,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.search_state import Tossing3DSearchState
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)
from hitl_pmp.methods.belief_space.types.stop_action import STOP_ACTION
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta
from hitl_pmp.planning.grounding import SkillGrounder


def _domain_model(**kwargs: object) -> Tossing3DPracticeModel:
    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env)
    skills = provider.skills()
    reset_cost = kwargs.pop("reset_cost", None)
    if reset_cost is not None:
        reset = provider.human_cube_bin_reset_skill()
        assert reset is not None
        skills = (*skills, reset.skill.model_copy(update={"practice_cost": reset_cost}))
    ground_skills = SkillGrounder.applicable_ground_skills(
        skills=skills,
        objects=provider.objects(),
        true_atoms=SkillGrounder.all_possible_ground_atoms(
            objects=provider.objects(), predicates=provider.predicates()
        ),
    )
    return Tossing3DPracticeModel(ground_skills=tuple(ground_skills), **kwargs)


def _ground_skill(*, model: Tossing3DPracticeModel, name: str):
    return next(skill for skill in model.ground_skills if skill.skill.name == name)


def _belief(*, state: Tossing3DBeliefState, skill_name: str) -> SkillBelief:
    return state.skill_beliefs[skill_name]


def _pending_examples(*, state: Tossing3DBeliefState, skill_name: str) -> int:
    return state.pending_examples.get(skill_name, 0)


def _point_state(
    *, toss: float, pick: float, open_gripper: float, accumulated_cost: float = 0.0
) -> Tossing3DBeliefState:
    return Tossing3DBeliefState(
        skill_beliefs={
            TOSS_SKILL: _point_belief(competence=toss, learning_rate=0.0),
            PICK_SKILL: _point_belief(competence=pick, learning_rate=0.0),
            OPEN_GRIPPER_SKILL: _point_belief(competence=open_gripper, learning_rate=0.0),
        },
        accumulated_cost=accumulated_cost,
    )


def _search_state(
    *, model: Tossing3DPracticeModel, state: Tossing3DBeliefState, action_name: str
) -> Tossing3DSearchState:
    return make_tossing3d_search_state(
        state=state, true_atoms=_ground_skill(model=model, name=action_name).preconditions
    )


def _action(
    *, model: Tossing3DPracticeModel, search_state: Tossing3DSearchState, name: str
) -> GroundSkill:
    return next(
        action
        for action in model.get_valid_actions(environment_state=search_state)
        if action.skill.name == name
    )


def _outcomes(*, model: Tossing3DPracticeModel, state: Tossing3DBeliefState, name: str):
    search_state = _search_state(model=model, state=state, action_name=name)
    return model.outcomes(
        environment_state=search_state,
        state=state,
        action=_action(model=model, search_state=search_state, name=name),
    )


def _expected_stop_value(*, model: Tossing3DPracticeModel, state: Tossing3DBeliefState) -> float:
    projected = refit_belief_state(state=state)
    deployment_value = sum(
        pick.probability
        * toss.probability
        * opened.probability
        * model.evaluate_policy(
            sampled_theta=Tossing3DTheta(
                pick=pick.hypothesis,
                toss=toss.hypothesis,
                open_gripper=opened.hypothesis,
            )
        )
        for pick, toss, opened in product(
            _belief(state=projected, skill_name=PICK_SKILL).hypotheses,
            _belief(state=projected, skill_name=TOSS_SKILL).hypotheses,
            _belief(state=projected, skill_name=OPEN_GRIPPER_SKILL).hypotheses,
        )
    )
    return model.G(policy_value=deployment_value, summed_cost=state.accumulated_cost)


def test_G_returns_policy_value_within_hard_budget() -> None:
    model = Tossing3DPracticeModel()
    assert model.G(policy_value=2.5, summed_cost=150.0) == pytest.approx(2.5)


def test_G_returns_negative_infinity_beyond_hard_budget() -> None:
    model = Tossing3DPracticeModel()
    assert model.G(policy_value=2.5, summed_cost=150.001) == -float("inf")


def test_budget_does_not_change_environment_action_applicability() -> None:
    model = _domain_model(reset_cost=1.0)
    state = make_default_tossing3d_belief().model_copy(update={"accumulated_cost": 151.0})
    search_state = _search_state(model=model, state=state, action_name=PICK_SKILL)
    assert model.get_valid_actions(environment_state=search_state)


def test_search_prunes_state_beyond_hard_budget() -> None:
    model = _domain_model(reset_cost=1.0)
    state = make_default_tossing3d_belief().model_copy(update={"accumulated_cost": 151.0})
    search_state = _search_state(model=model, state=state, action_name=PICK_SKILL)
    value, action = solve_belief_space_expectimax(
        environment_state=search_state,
        belief_state=state,
        summed_cost=state.accumulated_cost,
        horizon=3,
        model=model,
    )
    assert value == -float("inf")
    assert action == STOP_ACTION


def test_search_protocol_charges_accumulated_cost_once() -> None:
    state = _point_state(toss=0.8, pick=0.5, open_gripper=1.0, accumulated_cost=3.0)
    model = Tossing3DPracticeModel()
    value, action = solve_belief_space_expectimax(
        environment_state=make_tossing3d_search_state(state=state, true_atoms=frozenset()),
        belief_state=state,
        summed_cost=state.accumulated_cost,
        horizon=3,
        model=model,
    )
    assert value == pytest.approx(_expected_stop_value(model=model, state=state))
    assert action == STOP_ACTION


def test_search_protocol_merges_identical_exploration_successors() -> None:
    state = make_default_tossing3d_belief()
    state = state.model_copy(
        update={
            "skill_beliefs": {
                **state.skill_beliefs,
                TOSS_SKILL: _point_belief(competence=0.5),
            }
        }
    )
    model = _domain_model(random_toss_competence=0.5)
    environment_state = _search_state(model=model, state=state, action_name=TOSS_SKILL)
    action = _action(model=model, search_state=environment_state, name=TOSS_SKILL)
    successors = model.sample_next_states(
        environment_state=environment_state,
        practice_action=action,
        belief_state=state,
    )
    assert len(successors) == 2
    probabilities = [
        model.transition_probability(
            potential_next_environment_state=successor,
            sampled_cost=cost,
            environment_state=environment_state,
            practice_action=action,
            belief_state=state,
        )
        for successor, cost in successors
    ]
    assert probabilities == pytest.approx([0.5, 0.5])


def _point_belief(*, competence: float, learning_rate: float = 0.1) -> SkillBelief:
    return SkillBelief(
        hypotheses=(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=competence, learning_rate=learning_rate),
                probability=1.0,
            ),
        )
    )


def test_particle_prior_is_continuous_seeded_and_normalized() -> None:
    first = make_particle_belief_prior(num_particles=128, seed=7)
    second = make_particle_belief_prior(num_particles=128, seed=7)

    assert first == second
    assert first.estimator == "particle_filter"
    parameters, weights = belief_arrays(belief=first)
    assert parameters.shape == (128, 2)
    assert weights.sum() == pytest.approx(1.0)
    assert np.any(~np.isin(parameters[:, 1], (0.0, 0.1)))


def test_particle_belief_round_trips_through_json_without_expanding_particles() -> None:
    prior = make_particle_belief_prior(num_particles=128, seed=8)

    restored = SkillBelief.model_validate_json(prior.model_dump_json())

    assert restored == prior
    assert restored.hypotheses == ()
    actual_parameters, actual_weights = belief_arrays(belief=restored)
    expected_parameters, expected_weights = belief_arrays(belief=prior)
    np.testing.assert_array_equal(actual_parameters, expected_parameters)
    np.testing.assert_array_equal(actual_weights, expected_weights)


def test_particle_filter_conditions_with_bernoulli_likelihood() -> None:
    prior = make_particle_belief_prior(num_particles=1_000, seed=9)
    posterior = condition_particle_belief(belief=prior, success=True)

    assert mean_competence(belief=posterior) > mean_competence(belief=prior)
    assert particle_filter_diagnostics(belief=posterior).effective_sample_size > 0.0


def test_particle_filter_resampling_is_seeded_and_reports_diagnostics() -> None:
    prior = make_particle_belief_prior(num_particles=128, seed=11)
    first = prior
    second = prior
    for _ in range(8):
        first = condition_particle_belief(belief=first, success=True)
        second = condition_particle_belief(belief=second, success=True)

    assert first == second
    diagnostics = particle_filter_diagnostics(belief=first)
    assert diagnostics.resampling_count > 0
    assert diagnostics.num_particles == 128


def test_particle_filter_resampling_rejuvenates_learning_rate_particles() -> None:
    prior = make_particle_belief_prior(num_particles=256, seed=15)
    posterior = prior
    for _ in range(12):
        posterior = condition_particle_belief(belief=posterior, success=True)

    parameters, _ = belief_arrays(belief=posterior)
    learning_rates = parameters[:, 1]
    assert particle_filter_diagnostics(belief=posterior).resampling_count > 0
    assert np.all((learning_rates >= 0.0) & (learning_rates <= 1.0))
    assert np.unique(learning_rates).size > 1


def test_particle_filter_rejuvenation_preserves_joint_parameter_correlation() -> None:
    prior = make_particle_belief_prior(num_particles=2_000, seed=16)
    parameters, weights = belief_arrays(belief=prior)
    parameters = parameters.copy()
    parameters[:, 1] = parameters[:, 0] * 0.1
    correlated = belief_from_arrays(belief=prior, parameters=parameters, weights=weights)
    posterior = correlated
    for _ in range(8):
        posterior = condition_particle_belief(belief=posterior, success=True)

    parameters, _ = belief_arrays(belief=posterior)
    assert np.corrcoef(parameters.T)[0, 1] > 0.95


def test_particle_filter_accepts_observation_after_boundary_rejuvenation() -> None:
    prior = make_particle_belief_prior(num_particles=256, seed=17)
    posterior = prior
    for _ in range(30):
        posterior = condition_particle_belief(belief=posterior, success=True)

    posterior = condition_particle_belief(belief=posterior, success=False)

    _, weights = belief_arrays(belief=posterior)
    assert weights.sum() == pytest.approx(1.0)


def test_particle_prediction_advances_competence_but_not_learning_rate() -> None:
    prior = make_particle_belief_prior(num_particles=128, seed=13)
    predicted = refit_skill_belief(belief=prior, training_examples=2)

    assert predicted.estimator == "particle_filter"
    before, _ = belief_arrays(belief=prior)
    after, _ = belief_arrays(belief=predicted)
    np.testing.assert_array_equal(after[:, 1], before[:, 1])
    assert np.all(after[:, 0] >= before[:, 0])


def test_only_physically_applicable_actions_are_returned() -> None:
    model = _domain_model(reset_cost=0.2)
    belief = make_default_tossing3d_belief()
    ready = _search_state(model=model, state=belief, action_name=PICK_SKILL)
    assert {action.skill.name for action in model.get_valid_actions(environment_state=ready)} == {
        PICK_SKILL,
        OPEN_GRIPPER_SKILL,
        RESET_SKILL,
    }
    carrying = _search_state(model=model, state=belief, action_name=TOSS_SKILL)
    assert {
        action.skill.name for action in model.get_valid_actions(environment_state=carrying)
    } == {
        TOSS_SKILL,
        OPEN_GRIPPER_SKILL,
        RESET_SKILL,
    }


def test_search_state_serializes_ees_atoms_without_serializing_predicate_functions() -> None:
    model = _domain_model()
    state = _search_state(
        model=model, state=make_default_tossing3d_belief(), action_name=PICK_SKILL
    )
    serialized = state.model_dump(mode="json")
    assert serialized["atoms"] == sorted(str(atom) for atom in state.true_atoms)
    assert "true_atoms" not in serialized
    assert "state" not in serialized


def test_search_state_reuses_rendered_atoms() -> None:
    model = _domain_model()
    belief = make_default_tossing3d_belief()
    true_atoms = _ground_skill(model=model, name=PICK_SKILL).preconditions
    render_atoms.cache_clear()

    first = make_tossing3d_search_state(state=belief, true_atoms=true_atoms)
    cache_after_first = render_atoms.cache_info()
    second = make_tossing3d_search_state(state=belief, true_atoms=true_atoms)

    assert first.atoms == second.atoms
    assert render_atoms.cache_info().hits == cache_after_first.hits + 1


def test_batched_sampling_and_evaluation_matches_individual_theta_path() -> None:
    belief = make_default_tossing3d_belief()
    individual_model = Tossing3DPracticeModel(seed=123)
    batched_model = Tossing3DPracticeModel(seed=123)
    samples = individual_model.sample_thetas_from_belief(belief_state=belief, num_samples=100)

    assert batched_model.sample_policy_values_from_belief(
        belief_state=belief, num_samples=100
    ) == pytest.approx([
        individual_model.evaluate_policy(sampled_theta=sample) for sample in samples
    ])


@pytest.mark.parametrize("action_name", [PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL])
def test_human_reset_uses_unchanged_ees_empty_preconditions(*, action_name: str) -> None:
    model = _domain_model(reset_cost=1.0)
    state = make_default_tossing3d_belief()
    search_state = _search_state(model=model, state=state, action_name=action_name)
    assert RESET_SKILL in {
        action.skill.name for action in model.get_valid_actions(environment_state=search_state)
    }


def test_disabling_human_reset_removes_only_that_ees_skill() -> None:
    with_reset = _domain_model(reset_cost=1.0)
    without_reset = _domain_model(reset_cost=None)
    state = make_default_tossing3d_belief()
    with_state = _search_state(model=with_reset, state=state, action_name=PICK_SKILL)
    without_state = _search_state(model=without_reset, state=state, action_name=PICK_SKILL)
    assert {
        action.skill.name for action in with_reset.get_valid_actions(environment_state=with_state)
    } - {
        action.skill.name
        for action in without_reset.get_valid_actions(environment_state=without_state)
    } == {RESET_SKILL}


def test_human_skill_uses_default_noop_belief_observers() -> None:
    model = _domain_model(reset_cost=0.25)
    state = make_default_tossing3d_belief()
    reset = _ground_skill(model=model, name=RESET_SKILL)
    assert reset.evaluate_practice_cost() == 0.25
    assert (
        model.observe_outcome(
            state=state,
            ground_skill=reset,
            success=True,
            was_random_exploration=False,
        )
        == state
    )
    assert model.observe_training_example(state=state, skill_name=RESET_SKILL) == state


def test_pick_outcomes_update_only_its_own_posterior() -> None:
    state = make_default_tossing3d_belief()
    model = _domain_model()
    search_state = _search_state(model=model, state=state, action_name=PICK_SKILL)
    outcomes = _outcomes(model=model, state=state, name=PICK_SKILL)
    assert [outcome[0] for outcome in outcomes] == pytest.approx([0.5, 0.5])
    assert all(
        _belief(state=outcome[1], skill_name=TOSS_SKILL)
        == _belief(state=state, skill_name=TOSS_SKILL)
        for outcome in outcomes
    )
    assert all(
        _pending_examples(state=outcome[1], skill_name=TOSS_SKILL) == 0 for outcome in outcomes
    )
    assert all(
        _pending_examples(state=outcome[1], skill_name=PICK_SKILL) == 1 for outcome in outcomes
    )
    assert mean_competence(
        belief=_belief(state=outcomes[0][1], skill_name=PICK_SKILL)
    ) > mean_competence(belief=_belief(state=state, skill_name=PICK_SKILL))
    assert mean_competence(
        belief=_belief(state=outcomes[1][1], skill_name=PICK_SKILL)
    ) < mean_competence(belief=_belief(state=state, skill_name=PICK_SKILL))
    assert outcomes[1][2] == search_state.true_atoms


@pytest.mark.parametrize("skill_name", [PICK_SKILL, OPEN_GRIPPER_SKILL])
def test_stationary_data_favors_zero_improvement(*, skill_name: str) -> None:
    model = _domain_model()
    skill = _ground_skill(model=model, name=skill_name)
    state = make_default_tossing3d_belief()
    for _ in range(20):
        for success in [True, False] * 5:
            state = model.observe_outcome(
                state=state,
                ground_skill=skill,
                success=success,
                was_random_exploration=False,
            )
        state = refit_belief_state(state=state)
    belief = _belief(state=state, skill_name=skill_name)
    stationary_mass = sum(
        item.probability for item in belief.hypotheses if item.hypothesis.learning_rate == 0
    )
    assert stationary_mass > 0.99
    assert mean_competence(belief=belief) == pytest.approx(0.5, abs=0.01)


def test_first_session_cannot_identify_learning_rate() -> None:
    model = _domain_model()
    pick = _ground_skill(model=model, name=PICK_SKILL)
    state = make_default_tossing3d_belief()
    for success in [True, False] * 50:
        state = model.observe_outcome(
            state=state,
            ground_skill=pick,
            success=success,
            was_random_exploration=False,
        )
    stationary_mass = sum(
        item.probability
        for item in _belief(state=state, skill_name=PICK_SKILL).hypotheses
        if item.hypothesis.learning_rate == 0
    )
    assert stationary_mass == pytest.approx(0.5)


def test_open_gripper_success_is_inferred_not_assumed() -> None:
    state = make_default_tossing3d_belief()
    model = _domain_model()
    closed_atoms = frozenset(
        atom
        for atom in _ground_skill(model=model, name=PICK_SKILL).preconditions
        if atom.predicate.name != "HandEmpty"
    )
    search_state = make_tossing3d_search_state(state=state, true_atoms=closed_atoms)
    outcomes = model.outcomes(
        environment_state=search_state,
        state=state,
        action=_action(model=model, search_state=search_state, name=OPEN_GRIPPER_SKILL),
    )
    assert [o[0] for o in outcomes] == pytest.approx([0.5, 0.5])
    assert outcomes[1][2] == closed_atoms
    open_gripper = _ground_skill(model=model, name=OPEN_GRIPPER_SKILL)
    for _ in range(100):
        state = model.observe_outcome(
            state=state,
            ground_skill=open_gripper,
            success=True,
            was_random_exploration=False,
        )
    open_gripper_belief = _belief(state=state, skill_name=OPEN_GRIPPER_SKILL)
    assert mean_competence(belief=open_gripper_belief) > 0.99
    projected = refit_skill_belief(belief=open_gripper_belief, training_examples=1)
    assert mean_competence(belief=projected) - mean_competence(belief=open_gripper_belief) < 0.001


def test_pending_examples_predict_improvement_without_changing_current_competence() -> None:
    model = _domain_model()
    pick = _ground_skill(model=model, name=PICK_SKILL)
    state = make_default_tossing3d_belief()
    observed = model.observe_outcome(
        state=state,
        ground_skill=pick,
        success=True,
        was_random_exploration=False,
    )
    assert _belief(state=observed, skill_name=PICK_SKILL) == condition_skill_belief(
        belief=_belief(state=state, skill_name=PICK_SKILL), success=True
    )
    refit = refit_belief_state(state=observed)
    assert mean_competence(belief=_belief(state=refit, skill_name=PICK_SKILL)) > mean_competence(
        belief=_belief(state=observed, skill_name=PICK_SKILL)
    )
    assert _pending_examples(state=refit, skill_name=PICK_SKILL) == 0


@pytest.mark.parametrize("skill_name", [PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL])
def test_random_exploration_does_not_update_any_skill_belief(*, skill_name: str) -> None:
    model = _domain_model()
    state = make_default_tossing3d_belief()
    observed = model.observe_outcome(
        state=state,
        ground_skill=_ground_skill(model=model, name=skill_name),
        success=True,
        was_random_exploration=True,
    )
    assert observed == state


def test_toss_random_exploration_does_not_condition_policy_belief() -> None:
    state = make_default_tossing3d_belief()
    outcomes = _outcomes(
        model=_domain_model(exploration_epsilon=0.5, random_toss_competence=0.2),
        state=state,
        name=TOSS_SKILL,
    )
    assert sum(outcome[0] for outcome in outcomes) == pytest.approx(1.0)
    unchanged = [
        outcome
        for outcome in outcomes
        if _belief(state=outcome[1], skill_name=TOSS_SKILL)
        == _belief(state=state, skill_name=TOSS_SKILL)
    ]
    assert sum(outcome[0] for outcome in unchanged) == pytest.approx(0.5)
    assert all(
        _pending_examples(state=outcome[1], skill_name=TOSS_SKILL) == 1 for outcome in outcomes
    )


def test_refit_is_deferred_until_cycle_boundary() -> None:
    state = make_default_tossing3d_belief()
    state = state.model_copy(
        update={
            "skill_beliefs": {
                **state.skill_beliefs,
                TOSS_SKILL: _point_belief(competence=0.5),
            },
            "pending_examples": {TOSS_SKILL: 2},
        },
    )
    assert mean_competence(belief=_belief(state=state, skill_name=TOSS_SKILL)) == pytest.approx(0.5)
    refit = refit_belief_state(state=state)
    assert mean_competence(belief=_belief(state=refit, skill_name=TOSS_SKILL)) == pytest.approx(
        0.595
    )
    assert _pending_examples(state=refit, skill_name=TOSS_SKILL) == 0


def test_stop_value_solves_deployment_chain_within_hard_budget() -> None:
    state = _point_state(toss=0.8, pick=0.5, open_gripper=1.0, accumulated_cost=3.0)
    model = Tossing3DPracticeModel()
    assert _expected_stop_value(model=model, state=state) == pytest.approx((0.5 + 0.5 * 0.5) * 0.8)


def test_partial_reset_does_not_open_a_closed_gripper() -> None:
    state = make_default_tossing3d_belief()
    model = _domain_model(reset_cost=0.01)
    reset = _outcomes(model=model, state=state, name=RESET_SKILL)[0]
    assert "HandEmpty" not in {atom.predicate.name for atom in reset[2]}
    opened = _outcomes(model=model, state=state, name=OPEN_GRIPPER_SKILL)[0]
    assert "HandEmpty" in {atom.predicate.name for atom in opened[2]}


def test_invalid_configuration_is_rejected_early() -> None:
    with pytest.raises(ValidationError):
        Tossing3DBeliefState(
            skill_beliefs={TOSS_SKILL: _point_belief(competence=0.5)},
            accumulated_cost=float("nan"),
        )
