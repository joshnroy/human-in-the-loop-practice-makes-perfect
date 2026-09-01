"""Belief initialization and observation updates for Tossing3D skills."""

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.skill_belief import (
    SkillBelief,
    SkillHypothesis,
    WeightedHypothesis,
)


class SkillBeliefModel:
    """Configurable belief updates contributed by one lifted practice skill."""

    def __init__(
        self,
        *,
        skill: Skill | None = None,
        examples_from_outcomes: bool = False,
        examples_from_sampler: bool = False,
    ) -> None:
        self.skill = skill
        self.examples_from_outcomes = examples_from_outcomes
        self.examples_from_sampler = examples_from_sampler

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
    ) -> Tossing3DBeliefState:
        if self.skill is None or was_random_exploration:
            return state
        skill_name = self.skill.name
        skill_beliefs = dict(state.skill_beliefs)
        skill_beliefs[skill_name] = condition_skill_belief(
            belief=skill_beliefs[skill_name], success=success
        )
        pending_examples = dict(state.pending_examples)
        if self.examples_from_outcomes:
            pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        return state.model_copy(
            update={"skill_beliefs": skill_beliefs, "pending_examples": pending_examples}
        )

    def observe_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        if self.skill is None or not self.examples_from_sampler:
            return state
        pending_examples = dict(state.pending_examples)
        skill_name = self.skill.name
        pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        return state.model_copy(update={"pending_examples": pending_examples})


SKILL_BELIEF_MODELS: dict[Skill, SkillBeliefModel] = {
    Tossing3DSkills.PICK_CUBE: SkillBeliefModel(
        skill=Tossing3DSkills.PICK_CUBE, examples_from_outcomes=True
    ),
    Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS: SkillBeliefModel(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        examples_from_sampler=True,
    ),
    Tossing3DSkills.OPEN_GRIPPER: SkillBeliefModel(
        skill=Tossing3DSkills.OPEN_GRIPPER,
        examples_from_outcomes=True,
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
        skill_beliefs={skill.name: make_skill_belief_prior() for skill in SKILL_BELIEF_MODELS}
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
            "skill_beliefs": {
                skill_name: refit_skill_belief(
                    belief=belief,
                    training_examples=state.pending_examples.get(skill_name, 0),
                )
                for skill_name, belief in state.skill_beliefs.items()
            },
            "pending_examples": {},
        }
    )
