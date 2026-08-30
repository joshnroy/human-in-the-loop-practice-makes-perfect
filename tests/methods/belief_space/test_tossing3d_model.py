import pytest
from pydantic import ValidationError

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
    WeightedHypothesis,
    make_default_tossing3d_belief,
)


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
