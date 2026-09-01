"""Belief initialization and observation updates for Tossing3D skills."""

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
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


class SkillBeliefModel:
    """Belief updates contributed by one lifted practice skill."""

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        del success, was_random_exploration
        return state

    def observe_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        return state


class RobotSkillBeliefModel(SkillBeliefModel):
    def __init__(self, *, skill_name: str) -> None:
        self.skill_name = skill_name

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        del was_random_exploration
        return observe_robot_skill(state=state, skill_name=self.skill_name, success=success)


class TossSkillBeliefModel(SkillBeliefModel):
    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        return observe_toss(
            state=state,
            success=success,
            was_random_exploration=was_random_exploration,
        )

    def observe_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        return record_training_example(state=state)


SKILL_BELIEF_MODELS: dict[Skill, SkillBeliefModel] = {
    Tossing3DSkills.PICK_CUBE: RobotSkillBeliefModel(skill_name=PICK_SKILL),
    Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS: TossSkillBeliefModel(),
    Tossing3DSkills.OPEN_GRIPPER: RobotSkillBeliefModel(skill_name=OPEN_GRIPPER_SKILL),
}


def make_skill_belief_models(
    *, ground_skills: tuple[GroundSkill, ...]
) -> tuple[dict[GroundSkill, SkillBeliefModel], dict[str, SkillBeliefModel]]:
    """Associate every practice skill with explicit updates or the default no-op."""
    by_ground_skill = {
        ground_skill: SKILL_BELIEF_MODELS.get(ground_skill.skill, SkillBeliefModel())
        for ground_skill in ground_skills
    }
    by_name = {ground_skill.skill.name: model for ground_skill, model in by_ground_skill.items()}
    return by_ground_skill, by_name


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
