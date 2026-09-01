import pytest
from pydantic import ValidationError

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.belief_space.expectimax import solve_belief_space_expectimax
from hitl_pmp.methods.belief_space.tossing3d_model import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
    SkillBelief,
    SkillHypothesis,
    Tossing3DAction,
    Tossing3DBeliefState,
    Tossing3DEnvironmentState,
    Tossing3DPracticeModel,
    Tossing3DSearchState,
    WeightedHypothesis,
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types import STOP_ACTION
from hitl_pmp.planning.grounding import SkillGrounder


def _domain_model(**kwargs: object) -> Tossing3DPracticeModel:
    env = Tossing3DEnvironment(scene_bg=False)
    provider = Tossing3DSkillProvider(env=env)
    skills = provider.skills()
    if kwargs.get("reset_cost") is not None:
        reset = provider.human_cube_bin_reset_skill()
        assert reset is not None
        skills = (*skills, reset.skill)
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


def _search_state(
    *, model: Tossing3DPracticeModel, state: Tossing3DBeliefState, action_name: str
) -> Tossing3DSearchState:
    return Tossing3DSearchState(
        state=state, true_atoms=_ground_skill(model=model, name=action_name).preconditions
    )


def _action(
    *, model: Tossing3DPracticeModel, search_state: Tossing3DSearchState, name: str
) -> Tossing3DAction:
    return next(
        action
        for action in model.get_valid_actions(environment_state=search_state)
        if isinstance(action, Tossing3DAction) and action.name == name
    )


def _outcomes(*, model: Tossing3DPracticeModel, state: Tossing3DBeliefState, name: str):
    search_state = _search_state(model=model, state=state, action_name=name)
    return model.outcomes(
        environment_state=search_state,
        state=state,
        action=_action(model=model, search_state=search_state, name=name),
    )


def test_hard_budget_filters_unaffordable_actions() -> None:
    model = _domain_model(hard_budget=2, reset_cost=1)
    state = make_default_tossing3d_belief()
    state = state.model_copy(update={"accumulated_cost": 1.0})
    search_state = _search_state(model=model, state=state, action_name=PICK_SKILL)
    actions = model.get_valid_actions(environment_state=search_state)
    assert {action.name for action in actions} == {PICK_SKILL, OPEN_GRIPPER_SKILL, RESET_SKILL}
    exhausted = state.model_copy(update={"accumulated_cost": 2.0})
    assert (
        model.get_valid_actions(
            environment_state=search_state.model_copy(update={"state": exhausted})
        )
        == []
    )


def test_hard_budget_has_no_linear_cost_penalty() -> None:
    model = Tossing3DPracticeModel(hard_budget=2, practice_cost=0.9)
    assert (
        model.score_pomdp_value_from_policy_value_and_cost(policy_value=0.5, summed_cost=2) == 0.5
    )
    state = make_default_tossing3d_belief()
    assert model.stop_value(state) == model.stop_value(
        state.model_copy(update={"accumulated_cost": 2.0})
    )


def test_zero_budget_stops_even_with_search_horizon_remaining() -> None:
    state = make_default_tossing3d_belief()
    model = _domain_model(hard_budget=0, reset_cost=1)
    _, action = solve_belief_space_expectimax(
        environment_state=_search_state(model=model, state=state, action_name=PICK_SKILL),
        belief_state=state,
        summed_cost=0,
        horizon=10,
        model=model,
    )
    assert action == STOP_ACTION


def test_search_protocol_charges_accumulated_cost_once() -> None:
    state = Tossing3DBeliefState(
        environment_state=Tossing3DEnvironmentState.READY,
        toss_belief=_point_belief(competence=0.8, learning_rate=0.0),
        pick_belief=_point_belief(competence=0.5, learning_rate=0.0),
        open_gripper_belief=_point_belief(competence=1.0, learning_rate=0.0),
        accumulated_cost=3.0,
    )
    model = Tossing3DPracticeModel(practice_cost=0.1)
    value, action = solve_belief_space_expectimax(
        environment_state=Tossing3DSearchState(state=state, true_atoms=frozenset()),
        belief_state=state,
        summed_cost=state.accumulated_cost,
        horizon=3,
        model=model,
    )
    assert value == pytest.approx(model.stop_value(state))
    assert action == STOP_ACTION


def test_search_protocol_merges_identical_exploration_successors() -> None:
    state = Tossing3DBeliefState(
        environment_state=Tossing3DEnvironmentState.HOLDING,
        toss_belief=_point_belief(competence=0.5),
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


def test_only_physically_applicable_actions_are_returned() -> None:
    model = _domain_model(reset_cost=0.2)
    belief = make_default_tossing3d_belief()
    ready = _search_state(model=model, state=belief, action_name=PICK_SKILL)
    assert {action.name for action in model.get_valid_actions(environment_state=ready)} == {
        PICK_SKILL,
        OPEN_GRIPPER_SKILL,
        RESET_SKILL,
    }
    holding = belief.model_copy(update={"environment_state": Tossing3DEnvironmentState.HOLDING})
    carrying = _search_state(model=model, state=holding, action_name=TOSS_SKILL)
    assert {action.name for action in model.get_valid_actions(environment_state=carrying)} == {
        TOSS_SKILL,
        OPEN_GRIPPER_SKILL,
        RESET_SKILL,
    }


@pytest.mark.parametrize("action_name", [PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL])
def test_human_reset_uses_unchanged_ees_empty_preconditions(*, action_name: str) -> None:
    model = _domain_model(reset_cost=1.0)
    state = make_default_tossing3d_belief()
    search_state = _search_state(model=model, state=state, action_name=action_name)
    assert RESET_SKILL in {
        action.name for action in model.get_valid_actions(environment_state=search_state)
    }


def test_disabling_human_reset_removes_only_that_ees_skill() -> None:
    with_reset = _domain_model(reset_cost=1.0)
    without_reset = _domain_model(reset_cost=None)
    state = make_default_tossing3d_belief()
    with_state = _search_state(model=with_reset, state=state, action_name=PICK_SKILL)
    without_state = _search_state(model=without_reset, state=state, action_name=PICK_SKILL)
    assert {
        action.name for action in with_reset.get_valid_actions(environment_state=with_state)
    } - {
        action.name for action in without_reset.get_valid_actions(environment_state=without_state)
    } == {RESET_SKILL}


def test_pick_outcomes_update_only_its_own_posterior() -> None:
    state = make_default_tossing3d_belief()
    outcomes = _outcomes(model=_domain_model(), state=state, name=PICK_SKILL)
    assert [outcome.probability for outcome in outcomes] == pytest.approx([0.5, 0.5])
    assert all(outcome.next_state.toss_belief == state.toss_belief for outcome in outcomes)
    assert all(outcome.next_state.pending_training_examples == 0 for outcome in outcomes)
    assert all(outcome.next_state.pending_pick_examples == 1 for outcome in outcomes)
    assert outcomes[0].next_state.pick_belief.mean_competence > state.pick_belief.mean_competence
    assert outcomes[1].next_state.pick_belief.mean_competence < state.pick_belief.mean_competence


@pytest.mark.parametrize("skill_name", [PICK_SKILL, OPEN_GRIPPER_SKILL])
def test_stationary_data_favors_zero_improvement(*, skill_name: str) -> None:
    state = make_default_tossing3d_belief()
    for _ in range(20):
        for success in [True, False] * 5:
            state = Tossing3DPracticeModel.observe_robot_skill(
                state=state, skill_name=skill_name, success=success
            )
        state = state.after_refit()
    belief = state.pick_belief if skill_name == PICK_SKILL else state.open_gripper_belief
    stationary_mass = sum(
        item.probability for item in belief.hypotheses if item.hypothesis.learning_rate == 0
    )
    assert stationary_mass > 0.99
    assert belief.mean_competence == pytest.approx(0.5, abs=0.01)


def test_first_session_cannot_identify_learning_rate() -> None:
    state = make_default_tossing3d_belief()
    for success in [True, False] * 50:
        state = Tossing3DPracticeModel.observe_robot_skill(
            state=state, skill_name=PICK_SKILL, success=success
        )
    stationary_mass = sum(
        item.probability
        for item in state.pick_belief.hypotheses
        if item.hypothesis.learning_rate == 0
    )
    assert stationary_mass == pytest.approx(0.5)


def test_open_gripper_success_is_inferred_not_assumed() -> None:
    state = make_default_tossing3d_belief(
        environment_state=Tossing3DEnvironmentState.GRIPPER_CLOSED
    )
    model = _domain_model()
    closed_atoms = frozenset(
        atom
        for atom in _ground_skill(model=model, name=PICK_SKILL).preconditions
        if atom.predicate.name != "HandEmpty"
    )
    search_state = Tossing3DSearchState(state=state, true_atoms=closed_atoms)
    outcomes = model.outcomes(
        environment_state=search_state,
        state=state,
        action=_action(model=model, search_state=search_state, name=OPEN_GRIPPER_SKILL),
    )
    assert [o.probability for o in outcomes] == pytest.approx([0.5, 0.5])
    assert outcomes[1].next_state.environment_state == state.environment_state
    for _ in range(100):
        state = Tossing3DPracticeModel.observe_robot_skill(
            state=state, skill_name=OPEN_GRIPPER_SKILL, success=True
        )
    assert state.open_gripper_belief.mean_competence > 0.99
    projected = state.open_gripper_belief.after_refit(training_examples=1)
    assert projected.mean_competence - state.open_gripper_belief.mean_competence < 0.001


def test_pending_examples_predict_improvement_without_changing_current_competence() -> None:
    state = make_default_tossing3d_belief()
    observed = Tossing3DPracticeModel.observe_robot_skill(
        state=state, skill_name=PICK_SKILL, success=True
    )
    assert observed.pick_belief == state.pick_belief.condition(success=True)
    assert observed.after_refit().pick_belief.mean_competence > observed.pick_belief.mean_competence
    assert observed.after_refit().pending_pick_examples == 0


def test_toss_random_exploration_does_not_condition_policy_belief() -> None:
    state = make_default_tossing3d_belief(environment_state=Tossing3DEnvironmentState.HOLDING)
    outcomes = _outcomes(
        model=_domain_model(exploration_epsilon=0.5, random_toss_competence=0.2),
        state=state,
        name=TOSS_SKILL,
    )
    assert sum(outcome.probability for outcome in outcomes) == pytest.approx(1.0)
    unchanged = [o for o in outcomes if o.next_state.toss_belief == state.toss_belief]
    assert sum(outcome.probability for outcome in unchanged) == pytest.approx(0.5)
    assert all(outcome.next_state.pending_training_examples == 1 for outcome in outcomes)


def test_refit_is_deferred_until_cycle_boundary() -> None:
    state = Tossing3DBeliefState(
        environment_state=Tossing3DEnvironmentState.HOLDING,
        toss_belief=_point_belief(competence=0.5),
        pending_training_examples=2,
    )
    assert state.toss_belief.mean_competence == pytest.approx(0.5)
    refit = state.after_refit()
    assert refit.toss_belief.mean_competence == pytest.approx(0.595)
    assert refit.pending_training_examples == 0


def test_stop_value_solves_deployment_chain_and_charges_cost() -> None:
    state = Tossing3DBeliefState(
        environment_state=Tossing3DEnvironmentState.STRANDED,
        toss_belief=_point_belief(competence=0.8, learning_rate=0.0),
        pick_belief=_point_belief(competence=0.5, learning_rate=0.0),
        open_gripper_belief=_point_belief(competence=1.0, learning_rate=0.0),
        accumulated_cost=3.0,
    )
    model = Tossing3DPracticeModel(practice_cost=0.1)
    assert model.stop_value(state) == pytest.approx((0.5 + 0.5 * 0.5) * 0.8 - 0.3)


def test_partial_reset_does_not_open_a_closed_gripper() -> None:
    state = make_default_tossing3d_belief(
        environment_state=Tossing3DEnvironmentState.CLOSED_STRANDED
    )
    model = _domain_model(reset_cost=0.01)
    reset = _outcomes(model=model, state=state, name=RESET_SKILL)[0].next_state
    assert reset.environment_state is Tossing3DEnvironmentState.GRIPPER_CLOSED
    opened = _outcomes(model=model, state=state, name=OPEN_GRIPPER_SKILL)[0].next_state
    assert opened.environment_state is Tossing3DEnvironmentState.STRANDED


def test_invalid_configuration_is_rejected_early() -> None:
    with pytest.raises(ValidationError):
        Tossing3DPracticeModel(practice_cost=-0.1)
    with pytest.raises(ValidationError):
        Tossing3DBeliefState(
            environment_state=Tossing3DEnvironmentState.READY,
            toss_belief=_point_belief(competence=0.5),
            accumulated_cost=float("nan"),
        )
