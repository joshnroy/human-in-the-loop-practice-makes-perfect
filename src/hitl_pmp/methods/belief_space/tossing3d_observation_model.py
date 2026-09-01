"""Belief initialization and observation updates for Tossing3D skills."""

from typing import Literal

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)

BeliefField = Literal["toss_belief", "pick_belief", "open_gripper_belief"]
ExampleCountField = Literal[
    "pending_training_examples",
    "pending_pick_examples",
    "pending_open_gripper_examples",
]


class SkillBeliefModel:
    """Configurable belief updates contributed by one lifted practice skill."""

    def __init__(
        self,
        *,
        belief_field: BeliefField | None = None,
        outcome_example_field: ExampleCountField | None = None,
        sampler_example_field: ExampleCountField | None = None,
    ) -> None:
        self.belief_field = belief_field
        self.outcome_example_field = outcome_example_field
        self.sampler_example_field = sampler_example_field

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        if self.belief_field is None or was_random_exploration:
            return state
        belief = condition_skill_belief(belief=getattr(state, self.belief_field), success=success)
        update: dict[str, object] = {self.belief_field: belief}
        if self.outcome_example_field is not None:
            update[self.outcome_example_field] = getattr(state, self.outcome_example_field) + 1
        return state.model_copy(update=update)

    def observe_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        if self.sampler_example_field is None:
            return state
        return state.model_copy(
            update={
                self.sampler_example_field: getattr(state, self.sampler_example_field) + 1
            }
        )


SKILL_BELIEF_MODELS: dict[Skill, SkillBeliefModel] = {
    Tossing3DSkills.PICK_CUBE: SkillBeliefModel(
        belief_field="pick_belief", outcome_example_field="pending_pick_examples"
    ),
    Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS: SkillBeliefModel(
        belief_field="toss_belief", sampler_example_field="pending_training_examples"
    ),
    Tossing3DSkills.OPEN_GRIPPER: SkillBeliefModel(
        belief_field="open_gripper_belief",
        outcome_example_field="pending_open_gripper_examples",
    ),
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
