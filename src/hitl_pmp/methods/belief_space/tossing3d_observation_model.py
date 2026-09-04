"""Belief initialization and observation updates for Tossing3D skills."""

from enum import Enum
from typing import Literal

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.types.belief_state import (
    ConcreteSkillBelief,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import ParticleFilterBelief
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief

from .tossing3d_constants import RESET_SKILL

BeliefEstimator = Literal["finite_grid", "particle_filter"]


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
        if self.skill is None or was_random_exploration:
            return state
        skill_name = self.skill.name
        if skill_name not in state.skill_beliefs:
            return state
        skill_beliefs = dict(state.skill_beliefs)
        belief = skill_beliefs[skill_name]
        skill_beliefs[skill_name] = (
            belief.condition_execution(success=success, observed_cost=observed_cost)
            if observed_cost is not None and isinstance(belief, ParticleFilterBelief)
            else condition_skill_belief(belief=belief, success=success)
        )
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
    estimator: BeliefEstimator = "particle_filter",
    num_particles: int = 256,
    seed: int = 0,
    cost_min: float = 0.0,
    cost_max: float = 0.01,
    cost_observation_scale: float = 0.0001,
    include_human_reset: bool = False,
) -> Tossing3DBeliefState:
    """Independent cost priors; human-reset performance remains known."""
    beliefs: dict[str, ConcreteSkillBelief]
    if estimator == "finite_grid":
        beliefs = {skill.name: make_skill_belief_prior() for skill in SKILL_BELIEF_MODELS}
    else:
        beliefs = {
            skill.name: ParticleFilterBelief.broad_prior(
                num_particles=num_particles,
                seed=seed + index,
                cost_min=cost_min,
                cost_max=cost_max,
                cost_observation_scale=cost_observation_scale,
            )
            for index, skill in enumerate(SKILL_BELIEF_MODELS)
        }
    if include_human_reset and estimator == "particle_filter":
        beliefs[RESET_SKILL] = ParticleFilterBelief.fixed_performance_cost_prior(
            num_particles=num_particles,
            seed=seed + len(beliefs),
            competence=1.0,
            learning_rate=0.0,
            cost_min=cost_min,
            cost_max=cost_max,
            cost_observation_scale=cost_observation_scale,
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
