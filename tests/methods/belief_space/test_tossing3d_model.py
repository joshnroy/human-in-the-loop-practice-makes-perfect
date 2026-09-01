from itertools import product

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
    observe_robot_skill,
    refit_belief_state,
    refit_skill_belief,
)
from hitl_pmp.methods.belief_space.tossing3d_transition_model import (
    make_tossing3d_search_state,
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
            projected.pick_belief.hypotheses,
            projected.toss_belief.hypotheses,
            projected.open_gripper_belief.hypotheses,
        )
    )
    return model.G(policy_value=deployment_value, summed_cost=state.accumulated_cost)


def test_G_subtracts_accumulated_per_skill_cost() -> None:
    model = Tossing3DPracticeModel()
    assert model.G(policy_value=2.5, summed_cost=2.0) == pytest.approx(0.5)


def test_search_protocol_charges_accumulated_cost_once() -> None:
    state = Tossing3DBeliefState(
        toss_belief=_point_belief(competence=0.8, learning_rate=0.0),
        pick_belief=_point_belief(competence=0.5, learning_rate=0.0),
        open_gripper_belief=_point_belief(competence=1.0, learning_rate=0.0),
        accumulated_cost=3.0,
    )
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
    state = make_default_tossing3d_belief().model_copy(
        update={"toss_belief": _point_belief(competence=0.5)},
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
    assert all(outcome[1].toss_belief == state.toss_belief for outcome in outcomes)
    assert all(outcome[1].pending_training_examples == 0 for outcome in outcomes)
    assert all(outcome[1].pending_pick_examples == 1 for outcome in outcomes)
    assert mean_competence(belief=outcomes[0][1].pick_belief) > mean_competence(
        belief=state.pick_belief
    )
    assert mean_competence(belief=outcomes[1][1].pick_belief) < mean_competence(
        belief=state.pick_belief
    )
    assert outcomes[1][2] == search_state.true_atoms


@pytest.mark.parametrize("skill_name", [PICK_SKILL, OPEN_GRIPPER_SKILL])
def test_stationary_data_favors_zero_improvement(*, skill_name: str) -> None:
    state = make_default_tossing3d_belief()
    for _ in range(20):
        for success in [True, False] * 5:
            state = observe_robot_skill(state=state, skill_name=skill_name, success=success)
        state = refit_belief_state(state=state)
    belief = state.pick_belief if skill_name == PICK_SKILL else state.open_gripper_belief
    stationary_mass = sum(
        item.probability for item in belief.hypotheses if item.hypothesis.learning_rate == 0
    )
    assert stationary_mass > 0.99
    assert mean_competence(belief=belief) == pytest.approx(0.5, abs=0.01)


def test_first_session_cannot_identify_learning_rate() -> None:
    state = make_default_tossing3d_belief()
    for success in [True, False] * 50:
        state = observe_robot_skill(state=state, skill_name=PICK_SKILL, success=success)
    stationary_mass = sum(
        item.probability
        for item in state.pick_belief.hypotheses
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
    for _ in range(100):
        state = observe_robot_skill(state=state, skill_name=OPEN_GRIPPER_SKILL, success=True)
    assert mean_competence(belief=state.open_gripper_belief) > 0.99
    projected = refit_skill_belief(belief=state.open_gripper_belief, training_examples=1)
    assert (
        mean_competence(belief=projected) - mean_competence(belief=state.open_gripper_belief)
        < 0.001
    )


def test_pending_examples_predict_improvement_without_changing_current_competence() -> None:
    state = make_default_tossing3d_belief()
    observed = observe_robot_skill(state=state, skill_name=PICK_SKILL, success=True)
    assert observed.pick_belief == condition_skill_belief(belief=state.pick_belief, success=True)
    refit = refit_belief_state(state=observed)
    assert mean_competence(belief=refit.pick_belief) > mean_competence(belief=observed.pick_belief)
    assert refit.pending_pick_examples == 0


def test_toss_random_exploration_does_not_condition_policy_belief() -> None:
    state = make_default_tossing3d_belief()
    outcomes = _outcomes(
        model=_domain_model(exploration_epsilon=0.5, random_toss_competence=0.2),
        state=state,
        name=TOSS_SKILL,
    )
    assert sum(outcome[0] for outcome in outcomes) == pytest.approx(1.0)
    unchanged = [o for o in outcomes if o[1].toss_belief == state.toss_belief]
    assert sum(outcome[0] for outcome in unchanged) == pytest.approx(0.5)
    assert all(outcome[1].pending_training_examples == 1 for outcome in outcomes)


def test_refit_is_deferred_until_cycle_boundary() -> None:
    state = make_default_tossing3d_belief().model_copy(
        update={
            "toss_belief": _point_belief(competence=0.5),
            "pending_training_examples": 2,
        },
    )
    assert mean_competence(belief=state.toss_belief) == pytest.approx(0.5)
    refit = refit_belief_state(state=state)
    assert mean_competence(belief=refit.toss_belief) == pytest.approx(0.595)
    assert refit.pending_training_examples == 0


def test_stop_value_solves_deployment_chain_and_charges_cost() -> None:
    state = Tossing3DBeliefState(
        toss_belief=_point_belief(competence=0.8, learning_rate=0.0),
        pick_belief=_point_belief(competence=0.5, learning_rate=0.0),
        open_gripper_belief=_point_belief(competence=1.0, learning_rate=0.0),
        accumulated_cost=3.0,
    )
    model = Tossing3DPracticeModel()
    assert _expected_stop_value(model=model, state=state) == pytest.approx(
        (0.5 + 0.5 * 0.5) * 0.8 - 3.0
    )


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
            toss_belief=_point_belief(competence=0.5),
            accumulated_cost=float("nan"),
        )
