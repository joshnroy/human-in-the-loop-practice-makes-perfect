"""Belief initialization and observation updates for Tossing3D skills."""

from enum import Enum

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.types.belief_state import (
    ConcreteSkillBelief,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    ParticleFilterBelief,
    create_broad_particle_prior,
    create_fixed_performance_cost_prior,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief

from .tossing3d_constants import RESET_SKILL


class PracticeExampleSource(Enum):
    OUTCOME = "outcome"
    SAMPLER = "sampler"


class SkillBeliefModel:
    """Configurable belief updates contributed by one lifted practice skill."""

    def __init__(
        self,
        *,
        skill: Skill | None = None,
        example_source: PracticeExampleSource | None = None,
    ) -> None:
        self.skill = skill
        self.example_source = example_source

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
        observed_cost: float | None = None,
    ) -> Tossing3DBeliefState:
        if self.skill is None:
            return state
        skill_name = self.skill.name
        if skill_name not in state.skill_beliefs:
            return state
        skill_beliefs = dict(state.skill_beliefs)
        belief = skill_beliefs[skill_name]
        if observed_cost is not None and isinstance(belief, ParticleFilterBelief):
            belief = (
                belief.condition_cost(observed_cost=observed_cost)
                if was_random_exploration
                else belief.condition_execution(success=success, observed_cost=observed_cost)
            )
        elif not was_random_exploration:
            belief = condition_skill_belief(belief=belief, success=success)
        skill_beliefs[skill_name] = belief
        if was_random_exploration:
            return state.model_copy(update={"skill_beliefs": skill_beliefs})
        pending_examples = dict(state.pending_examples)
        if self.example_source == PracticeExampleSource.OUTCOME:
            pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        return state.model_copy(
            update={"skill_beliefs": skill_beliefs, "pending_examples": pending_examples}
        )

    def observe_training_example(self, *, state: Tossing3DBeliefState) -> Tossing3DBeliefState:
        if self.skill is None or self.example_source != PracticeExampleSource.SAMPLER:
            return state
        pending_examples = dict(state.pending_examples)
        skill_name = self.skill.name
        pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        return state.model_copy(update={"pending_examples": pending_examples})


SKILL_BELIEF_MODELS: dict[Skill, SkillBeliefModel] = {
    Tossing3DSkills.PICK_CUBE: SkillBeliefModel(
        skill=Tossing3DSkills.PICK_CUBE,
        example_source=PracticeExampleSource.OUTCOME,
    ),
    Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS: SkillBeliefModel(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        example_source=PracticeExampleSource.SAMPLER,
    ),
    Tossing3DSkills.OPEN_GRIPPER: SkillBeliefModel(
        skill=Tossing3DSkills.OPEN_GRIPPER,
        example_source=PracticeExampleSource.OUTCOME,
    ),
}


def make_skill_belief_models(
    *, ground_skills: tuple[GroundSkill, ...]
) -> tuple[dict[GroundSkill, SkillBeliefModel], dict[str, SkillBeliefModel]]:
    """Associate every practice skill with explicit updates or the default no-op."""
    by_ground_skill = {
        ground_skill: SKILL_BELIEF_MODELS.get(
            ground_skill.skill,
            SkillBeliefModel(skill=ground_skill.skill),
        )
        for ground_skill in ground_skills
    }
    by_name = {ground_skill.skill.name: model for ground_skill, model in by_ground_skill.items()}
    return by_ground_skill, by_name


def make_skill_belief_prior() -> WeightedHypothesisBelief:
    return WeightedHypothesisBelief.broad_prior()


def make_default_tossing3d_belief(
    *,
    num_particles: int = 256,
    seed: int = 0,
    include_human_reset: bool = False,
) -> Tossing3DBeliefState:
    """Independent cost priors; human-reset performance remains known."""
    beliefs: dict[str, ConcreteSkillBelief] = {
        skill.name: create_broad_particle_prior(
            num_particles=num_particles,
            seed=seed + index,
        )
        for index, skill in enumerate(SKILL_BELIEF_MODELS)
    }
    if include_human_reset:
        beliefs[RESET_SKILL] = create_fixed_performance_cost_prior(
            num_particles=num_particles,
            seed=seed + len(beliefs),
            competence=1.0,
            learning_rate=0.0,
        )
    return Tossing3DBeliefState(skill_beliefs=beliefs)


def mean_competence(*, belief: SkillBelief) -> float:
    return belief.mean_competence()


def mean_learning_rate(*, belief: SkillBelief) -> float:
    return belief.mean_learning_rate()


def mean_cost(*, belief: ParticleFilterBelief) -> float:
    return belief.mean_cost()


def condition_skill_belief(*, belief: ConcreteSkillBelief, success: bool) -> ConcreteSkillBelief:
    """Condition on a greedy-policy outcome without pretending a refit occurred."""
    return belief.condition_outcome(success=success)


def refit_skill_belief(
    *, belief: ConcreteSkillBelief, training_examples: int
) -> ConcreteSkillBelief:
    """Advance competence along a locally linear learning curve.

    ``learning_rate`` is the first derivative of competence with respect to the
    number of training examples.  Competence is a probability, so the linear
    extrapolation is capped at one.
    """
    return belief.refit(training_examples=training_examples)


def observed_learning_rate(
    *, competence_before: float, competence_after: float, training_examples: int
) -> float | None:
    """Return the nonnegative competence increase per example for one cycle."""
    assert training_examples >= 0
    if training_examples == 0:
        return None
    return max(0.0, (competence_after - competence_before) / training_examples)


def refit_observed_skill_belief(
    *,
    belief: ConcreteSkillBelief,
    training_examples: int,
    cycle_start_competence: float | None,
) -> ConcreteSkillBelief:
    if cycle_start_competence is not None:
        rate = observed_learning_rate(
            competence_before=cycle_start_competence,
            competence_after=mean_competence(belief=belief),
            training_examples=training_examples,
        )
        if rate is not None:
            belief = belief.condition_learning_rate(observed_learning_rate=rate)
    return refit_skill_belief(belief=belief, training_examples=training_examples)


def refit_belief_state(
    *,
    state: Tossing3DBeliefState,
    cycle_start_competences: dict[str, float] | None = None,
) -> Tossing3DBeliefState:
    start_competences = cycle_start_competences or {}
    return state.model_copy(
        update={
            "skill_beliefs": {
                skill_name: refit_observed_skill_belief(
                    belief=belief,
                    training_examples=state.pending_examples.get(skill_name, 0),
                    cycle_start_competence=start_competences.get(skill_name),
                )
                for skill_name, belief in state.skill_beliefs.items()
            },
            "pending_examples": {},
        }
    )
