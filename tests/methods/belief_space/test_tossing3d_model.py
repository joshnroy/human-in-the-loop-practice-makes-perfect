import pytest
from pydantic import ValidationError

from hitl_pmp.methods.belief_space.expectimax import solve_belief_space_expectimax
from hitl_pmp.methods.belief_space.tossing3d_model import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
    SkillBelief,
    SkillHypothesis,
    Tossing3DBeliefState,
    Tossing3DEnvironmentState,
    Tossing3DPracticeModel,
    Tossing3DSearchState,
    WeightedHypothesis,
    make_default_tossing3d_belief,
)
from hitl_pmp.methods.belief_space.types import STOP_ACTION


@pytest.mark.parametrize("environment_state", list(Tossing3DEnvironmentState))
def test_hard_budget_filters_unaffordable_actions(
    *, environment_state: Tossing3DEnvironmentState
) -> None:
    model = Tossing3DPracticeModel(hard_budget=2, reset_cost=1)
    state = make_default_tossing3d_belief(environment_state=environment_state)
    state = state.model_copy(update={"accumulated_cost": 1.0})
    actions = model.get_valid_actions(environment_state=Tossing3DSearchState(state=state))
    assert tuple(action.name for action in actions) == model.actions(state)
    exhausted = state.model_copy(update={"accumulated_cost": 2.0})
    assert model.get_valid_actions(environment_state=Tossing3DSearchState(state=exhausted)) == []


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
    _, action = solve_belief_space_expectimax(
        environment_state=Tossing3DSearchState(state=state),
        belief_state=state,
        summed_cost=0,
        horizon=10,
        model=Tossing3DPracticeModel(hard_budget=0, reset_cost=1),
    )
    assert action == STOP_ACTION


def test_search_protocol_charges_accumulated_cost_once() -> None:
    state = Tossing3DBeliefState(
        environment_state=Tossing3DEnvironmentState.READY,
        toss_belief=_point_belief(competence=0.8, learning_rate=0.0),
        accumulated_cost=3.0,
    )
    model = Tossing3DPracticeModel(pick_competence=0.5, practice_cost=0.1)
    value, action = solve_belief_space_expectimax(
        environment_state=Tossing3DSearchState(state=state),
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
    model = Tossing3DPracticeModel(random_toss_competence=0.5)
    environment_state = Tossing3DSearchState(state=state)
    action = model.get_valid_actions(environment_state=environment_state)[0]
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
    model = Tossing3DPracticeModel(reset_cost=0.2)
    belief = make_default_tossing3d_belief()
    assert tuple(model.actions(belief)) == (PICK_SKILL,)
    holding = belief.model_copy(update={"environment_state": Tossing3DEnvironmentState.HOLDING})
    assert tuple(model.actions(holding)) == (TOSS_SKILL,)
    closed = belief.model_copy(
        update={"environment_state": Tossing3DEnvironmentState.GRIPPER_CLOSED}
    )
    assert tuple(model.actions(closed)) == (OPEN_GRIPPER_SKILL,)


def test_pick_is_a_fixed_environment_transition_not_a_learnable_arm() -> None:
    state = make_default_tossing3d_belief()
    outcomes = Tossing3DPracticeModel(pick_competence=0.75).outcomes(state, PICK_SKILL)
    assert [outcome.probability for outcome in outcomes] == pytest.approx([0.75, 0.25])
    assert all(outcome.next_state.toss_belief == state.toss_belief for outcome in outcomes)
    assert all(outcome.next_state.pending_training_examples == 0 for outcome in outcomes)


def test_toss_random_exploration_does_not_condition_policy_belief() -> None:
    state = make_default_tossing3d_belief(environment_state=Tossing3DEnvironmentState.HOLDING)
    outcomes = Tossing3DPracticeModel(exploration_epsilon=0.5, random_toss_competence=0.2).outcomes(
        state, TOSS_SKILL
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
        accumulated_cost=3.0,
    )
    model = Tossing3DPracticeModel(pick_competence=0.5, practice_cost=0.1)
    assert model.stop_value(state) == pytest.approx((0.5 + 0.5 * 0.5) * 0.8 - 0.3)


def test_partial_reset_does_not_open_a_closed_gripper() -> None:
    state = make_default_tossing3d_belief(
        environment_state=Tossing3DEnvironmentState.CLOSED_STRANDED
    )
    model = Tossing3DPracticeModel(reset_cost=0.01)
    reset = model.outcomes(state, RESET_SKILL)[0].next_state
    assert reset.environment_state is Tossing3DEnvironmentState.GRIPPER_CLOSED
    assert tuple(model.actions(reset)) == (OPEN_GRIPPER_SKILL,)
    opened = model.outcomes(state, OPEN_GRIPPER_SKILL)[0].next_state
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
