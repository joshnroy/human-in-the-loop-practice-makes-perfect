"""Belief initialization and observation updates for Tossing3D skills."""

from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)


def make_skill_belief_prior() -> SkillBelief:
    return SkillBelief(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(competence=competence, learning_rate=learning_rate),
                probability=0.1,
            )
            for competence in (0.0, 0.25, 0.5, 0.75, 1.0)
            for learning_rate in (0.0, 0.1)
        )
    )


def make_default_tossing3d_belief() -> Tossing3DBeliefState:
    """Independent broad priors for all robot skills; human reset is known."""
    return Tossing3DBeliefState(
        toss_belief=make_skill_belief_prior(),
        pick_belief=make_skill_belief_prior(),
        open_gripper_belief=make_skill_belief_prior(),
    )


def mean_competence(*, belief: SkillBelief) -> float:
    return sum(item.probability * item.hypothesis.competence for item in belief.hypotheses)


def condition_skill_belief(*, belief: SkillBelief, success: bool) -> SkillBelief:
    """Condition on a greedy-policy outcome without pretending a refit occurred."""
    weighted: list[tuple[SkillHypothesis, float]] = []
    for item in belief.hypotheses:
        likelihood = item.hypothesis.competence if success else 1.0 - item.hypothesis.competence
        mass = item.probability * likelihood
        if mass > 0.0:
            weighted.append((item.hypothesis, mass))
    normalizer = sum(mass for _hypothesis, mass in weighted)
    if normalizer <= 0.0:
        raise ValueError(f"observation success={success} has zero probability")
    return SkillBelief(
        hypotheses=tuple(
            WeightedHypothesis(hypothesis=hypothesis, probability=mass / normalizer)
            for hypothesis, mass in weighted
        )
    )


def refit_skill_belief(*, belief: SkillBelief, training_examples: int) -> SkillBelief:
    assert training_examples >= 0
    return SkillBelief(
        hypotheses=tuple(
            WeightedHypothesis(
                hypothesis=SkillHypothesis(
                    competence=1.0
                    - (1.0 - item.hypothesis.competence)
                    * (1.0 - item.hypothesis.learning_rate) ** training_examples,
                    learning_rate=item.hypothesis.learning_rate,
                ),
                probability=item.probability,
            )
            for item in belief.hypotheses
        )
    )


def refit_belief_state(*, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
    return state.model_copy(
        update={
            "toss_belief": refit_skill_belief(
                belief=state.toss_belief, training_examples=state.pending_training_examples
            ),
            "pick_belief": refit_skill_belief(
                belief=state.pick_belief, training_examples=state.pending_pick_examples
            ),
            "open_gripper_belief": refit_skill_belief(
                belief=state.open_gripper_belief,
                training_examples=state.pending_open_gripper_examples,
            ),
            "pending_training_examples": 0,
            "pending_pick_examples": 0,
            "pending_open_gripper_examples": 0,
        }
    )


def observe_robot_skill(
    *, state: Tossing3DBeliefState, skill_name: str, success: bool
) -> Tossing3DBeliefState:
    belief_field, count_field = {
        PICK_SKILL: ("pick_belief", "pending_pick_examples"),
        OPEN_GRIPPER_SKILL: ("open_gripper_belief", "pending_open_gripper_examples"),
    }[skill_name]
    belief = getattr(state, belief_field)
    return state.model_copy(
        update={
            belief_field: condition_skill_belief(belief=belief, success=success),
            count_field: getattr(state, count_field) + 1,
        }
    )


def observe_toss(
    *, state: Tossing3DBeliefState, success: bool, was_random_exploration: bool
) -> Tossing3DBeliefState:
    belief = state.toss_belief
    if not was_random_exploration:
        belief = condition_skill_belief(belief=belief, success=success)
    return state.model_copy(update={"toss_belief": belief})


def record_training_example(*, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
    return state.model_copy(
        update={"pending_training_examples": state.pending_training_examples + 1}
    )
