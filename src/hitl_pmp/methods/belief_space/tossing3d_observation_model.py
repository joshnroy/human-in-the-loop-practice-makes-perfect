"""Belief initialization and observation updates for Tossing3D skills."""

from enum import Enum
from typing import Literal

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.environments.tossing3d.skills import Tossing3DSkills
from hitl_pmp.methods.belief_space.types.belief_state import (
    ConcreteSkillBelief,
    SamplerTrainingState,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    ParticleFilterBelief,
    create_broad_particle_prior,
)
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief

from .competence_inference import BayesianSkillBelief, InferenceConfig, create_bayesian_prior
from .tossing3d_constants import OPEN_GRIPPER_SKILL, PICK_SKILLS, RESET_SKILLS, TOSS_SKILL


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
        resample: bool = True,
    ) -> Tossing3DBeliefState:
        assert resample or observed_cost is None, "imagined updates do not observe execution costs"
        if self.skill is None:
            return state
        skill_name = self.skill.name
        if skill_name not in state.skill_beliefs:
            return state
        skill_beliefs = dict(state.skill_beliefs)
        belief = skill_beliefs[skill_name]
        if observed_cost is not None and isinstance(
            belief, (ParticleFilterBelief, BayesianSkillBelief)
        ):
            belief = (
                belief.condition_cost(observed_cost=observed_cost)
                if was_random_exploration
                else belief.condition_execution(success=success, observed_cost=observed_cost)
            )
        elif not was_random_exploration:
            belief = condition_skill_belief(belief=belief, success=success, resample=resample)
        skill_beliefs[skill_name] = belief
        if was_random_exploration:
            return state.model_copy(update={"skill_beliefs": skill_beliefs})
        pending_examples = dict(state.pending_examples)
        if self.example_source == PracticeExampleSource.OUTCOME:
            pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        return state.model_copy(
            update={"skill_beliefs": skill_beliefs, "pending_examples": pending_examples}
        )

    def observe_training_example(
        self, *, state: Tossing3DBeliefState, success: bool
    ) -> Tossing3DBeliefState:
        if self.skill is None or self.example_source != PracticeExampleSource.SAMPLER:
            return state
        pending_examples = dict(state.pending_examples)
        skill_name = self.skill.name
        pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        training = dict(state.sampler_training)
        if skill_name in training:
            training[skill_name] = training[skill_name].observe(success=success)
        return state.model_copy(
            update={"pending_examples": pending_examples, "sampler_training": training}
        )


SKILL_BELIEF_MODELS: dict[Skill, SkillBeliefModel] = {
    Tossing3DSkills.PICK_CUBE: SkillBeliefModel(
        skill=Tossing3DSkills.PICK_CUBE,
    ),
    Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS: SkillBeliefModel(
        skill=Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS,
        example_source=PracticeExampleSource.SAMPLER,
    ),
    Tossing3DSkills.OPEN_GRIPPER: SkillBeliefModel(
        skill=Tossing3DSkills.OPEN_GRIPPER,
    ),
}


def skill_belief_model(*, ground_skill: GroundSkill) -> SkillBeliefModel:
    """Return the update contract for one provider-supplied ground skill."""
    if ground_skill.skill.name in PICK_SKILLS:
        return SKILL_BELIEF_MODELS[Tossing3DSkills.PICK_CUBE]
    if ground_skill.skill.name == TOSS_SKILL:
        return SKILL_BELIEF_MODELS[Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS]
    if ground_skill.skill.name == OPEN_GRIPPER_SKILL:
        return SKILL_BELIEF_MODELS[Tossing3DSkills.OPEN_GRIPPER]
    if ground_skill.skill.name in RESET_SKILLS:
        return SkillBeliefModel(skill=ground_skill.skill)
    return SKILL_BELIEF_MODELS.get(
        ground_skill.skill,
        SkillBeliefModel(skill=ground_skill.skill),
    )


def make_skill_belief_models(
    *, ground_skills: tuple[GroundSkill, ...]
) -> tuple[dict[GroundSkill, SkillBeliefModel], dict[str, SkillBeliefModel]]:
    """Associate every practice skill with explicit updates or the default no-op."""
    by_ground_skill = {
        ground_skill: skill_belief_model(ground_skill=ground_skill)
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
    additional_skill_names: tuple[str, ...] = (),
    model: Literal["global_curve", "local_trend"] | None = None,
    engine: Literal["particle", "grid"] = "particle",
    inference_config: InferenceConfig | None = None,
) -> Tossing3DBeliefState:
    """Independent priors for every modeled practice skill.

    Omitting model retains the legacy prior for standalone numerical consumers;
    the POMDP method selects its configured Bayesian model explicitly.
    """
    names = [skill.name for skill in SKILL_BELIEF_MODELS]
    names.extend(additional_skill_names)
    if include_human_reset:
        from .tossing3d_constants import RESET_SKILL

        names.append(RESET_SKILL)
    beliefs: dict[str, ConcreteSkillBelief] = {}
    for index, name in enumerate(dict.fromkeys(names)):
        beliefs[name] = (
            create_broad_particle_prior(num_particles=num_particles, seed=seed + index)
            if model is None
            else create_bayesian_prior(
                model=model,
                engine=engine,
                seed=seed + index,
                num_particles=num_particles,
                config=inference_config or InferenceConfig(),
            )
        )
    return Tossing3DBeliefState(
        skill_beliefs=beliefs, sampler_training={TOSS_SKILL: SamplerTrainingState()}
    )


def mean_competence(*, belief: SkillBelief) -> float:
    return belief.mean_competence()


def mean_learning_rate(*, belief: SkillBelief) -> float:
    return belief.mean_learning_rate()


def mean_cost(*, belief: ParticleFilterBelief | BayesianSkillBelief) -> float:
    return belief.mean_cost()


def condition_skill_belief(
    *, belief: ConcreteSkillBelief, success: bool, resample: bool = True
) -> ConcreteSkillBelief:
    """Condition on a greedy-policy outcome without pretending a refit occurred."""
    if isinstance(belief, BayesianSkillBelief):
        return belief.condition_outcome(success=success, resample=resample)
    return belief.condition_outcome(success=success)


def refit_skill_belief(
    *, belief: ConcreteSkillBelief, training_examples: int
) -> ConcreteSkillBelief:
    """Forecast after training without modifying the observed belief."""
    return belief.refit(training_examples=training_examples)


def refit_belief_state(
    *,
    state: Tossing3DBeliefState,
    learning_rate_process_noise_std: float = 0.0,
    advance_cycle: bool = False,
) -> Tossing3DBeliefState:
    """Apply the configured transition; S/F is the only competence evidence.

    Real cycle boundaries also reset evidence and ancestry bookkeeping when no
    training occurred. A hypothetical zero-example forecast remains an identity.
    Tracked one-class samplers defer learning credit until both labels exist;
    their first mixed-class refit uses the whole retained training set.
    """
    assert learning_rate_process_noise_std >= 0.0
    refitted_beliefs: dict[str, ConcreteSkillBelief] = {}
    for skill_name, belief in state.skill_beliefs.items():
        count = state.pending_examples.get(skill_name, 0)
        training = state.sampler_training.get(skill_name)
        if training is not None:
            count = training.refit_examples
        refitted: ConcreteSkillBelief
        if isinstance(belief, BayesianSkillBelief):
            refitted = (
                belief.advance_cycle(training_examples=count)
                if advance_cycle
                else belief.refit(training_examples=count)
            )
        else:
            refitted = belief.refit(training_examples=count).advance_learning_rate(
                process_noise_std=learning_rate_process_noise_std
            )
        refitted_beliefs[skill_name] = refitted
    return state.model_copy(
        update={
            "skill_beliefs": refitted_beliefs,
            "pending_examples": {},
            "sampler_training": {
                name: training.refitted() for name, training in state.sampler_training.items()
            },
        }
    )
