"""Belief initialization and observation updates for Tossing3D skills."""

from typing import ClassVar, Literal

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.methods.belief_space.types.belief_state import (
    ConcreteSkillBelief,
    SamplerTrainingState,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.belief_space.types.particle_filter_belief import (
    ParticleFilterBelief,
    create_broad_particle_prior,
)
from hitl_pmp.methods.belief_space.types.practice_example_source import PracticeExampleSource
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief
from hitl_pmp.methods.belief_space.types.weighted_hypothesis_belief import WeightedHypothesisBelief

from .belief_digest import BeliefDigest
from .bounded_memo import BoundedMemo
from .competence_inference import BayesianSkillBelief, InferenceConfig, create_bayesian_prior
from .practice_domain import TOSSING3D_DOMAIN, PracticeDomain


class SearchForecastCache:
    """Memoised refits and S/F conditionings of competence posteriors.

    Search forecasts the same few hundred (posterior, count) pairs over and over: J
    refits every skill's belief at every node, and every node's children condition the
    same parent belief. Both are pure functions of the belief and the count (or the
    outcome), so each is keyed on the belief's digest -- which reads every field of a
    `BayesianSkillBelief` -- and the count or outcome. The digest names the type, so no
    other representation can collide with one. Only that exact type is cached: another
    representation's signature need not read every field, and a subclass may add one.
    A hit returns the very object computed on the miss: beliefs are frozen.
    """

    MAX_ENTRIES: ClassVar[int] = 16384
    MAX_BYTES: ClassVar[int] = 256 * 1024 * 1024
    _refits: ClassVar[BoundedMemo[tuple[bytes, int], BayesianSkillBelief]] = BoundedMemo(
        max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES
    )
    _conditioned: ClassVar[BoundedMemo[tuple[bytes, bool], BayesianSkillBelief]] = BoundedMemo(
        max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES
    )

    @staticmethod
    def refit(*, belief: ConcreteSkillBelief, training_examples: int) -> ConcreteSkillBelief:
        if type(belief) is not BayesianSkillBelief:
            return belief.refit(training_examples=training_examples)
        key = (BeliefDigest.of(belief=belief), training_examples)
        refitted = SearchForecastCache._refits.get(key=key)
        if refitted is None:
            refitted = belief.refit(training_examples=training_examples)
            SearchForecastCache._refits.put(
                key=key, value=refitted, nbytes=SearchForecastCache._nbytes(belief=refitted)
            )
        return refitted

    @staticmethod
    def condition(*, belief: ConcreteSkillBelief, success: bool) -> ConcreteSkillBelief:
        if type(belief) is not BayesianSkillBelief:
            return belief.condition_outcome(success=success)
        key = (BeliefDigest.of(belief=belief), success)
        conditioned = SearchForecastCache._conditioned.get(key=key)
        if conditioned is None:
            conditioned = belief.condition_outcome(success=success)
            SearchForecastCache._conditioned.put(
                key=key,
                value=conditioned,
                nbytes=SearchForecastCache._nbytes(belief=conditioned),
            )
        return conditioned

    @staticmethod
    def clear() -> None:
        SearchForecastCache._refits.clear()
        SearchForecastCache._conditioned.clear()

    @staticmethod
    def _nbytes(*, belief: BayesianSkillBelief) -> int:
        # An upper bound: buffers a result shares with its input are counted again.
        return len(belief.latent_values) + len(belief.state_weights) + len(belief.parent_indices)


class SkillBeliefModel:
    """Configurable belief updates contributed by one lifted practice skill."""

    def __init__(
        self,
        *,
        skill: Skill | None = None,
        example_source: PracticeExampleSource | None = None,
        belief_name: str | None = None,
    ) -> None:
        self.skill = skill
        self.example_source = example_source
        # The belief this skill updates, when it shares one under another name.
        self.belief_name = belief_name or (None if skill is None else skill.name)

    def observe_outcome(
        self,
        *,
        state: Tossing3DBeliefState,
        success: bool,
        was_random_exploration: bool,
        observed_cost: float | None = None,
        resample: bool = True,
        condition_competence: bool | None = None,
    ) -> Tossing3DBeliefState:
        """`condition_competence` is the `CompetenceEvidence` decision for this
        attempt; left as None it falls back to excluding epsilon-random attempts."""
        assert resample or observed_cost is None, "imagined updates do not observe execution costs"
        if self.skill is None:
            return state
        skill_name = self.belief_name
        assert skill_name is not None
        if skill_name not in state.skill_beliefs:
            return state
        if condition_competence is None:
            condition_competence = not was_random_exploration
        skill_beliefs = dict(state.skill_beliefs)
        belief = skill_beliefs[skill_name]
        if observed_cost is not None and isinstance(
            belief, (ParticleFilterBelief, BayesianSkillBelief)
        ):
            belief = (
                belief.condition_execution(success=success, observed_cost=observed_cost)
                if condition_competence
                else belief.condition_cost(observed_cost=observed_cost)
            )
        elif condition_competence:
            belief = condition_skill_belief(belief=belief, success=success)
        skill_beliefs[skill_name] = belief
        pending_examples = dict(state.pending_examples)
        if self.example_source == PracticeExampleSource.OUTCOME:
            # The notebook's m: every attempt is a data point, success or failure.
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
        skill_name = self.belief_name
        assert skill_name is not None
        pending_examples[skill_name] = pending_examples.get(skill_name, 0) + 1
        training = dict(state.sampler_training)
        # Training rows belong to the acting skill's parameter sampler.
        sampler_name = self.skill.name
        if sampler_name in training:
            training[sampler_name] = training[sampler_name].observe(success=success)
        return state.model_copy(
            update={"pending_examples": pending_examples, "sampler_training": training}
        )


def skill_belief_model(
    *, ground_skill: GroundSkill, domain: PracticeDomain = TOSSING3D_DOMAIN
) -> SkillBeliefModel:
    """Return the update contract for one provider-supplied ground skill: the belief it
    updates (its own, or the one it shares) and what advances that belief's training
    clock. A skill with no belief gets the no-op contract."""
    belief_name = domain.belief_name(skill_name=ground_skill.skill.name)
    if belief_name is None:
        return SkillBeliefModel(skill=ground_skill.skill)
    return SkillBeliefModel(
        skill=ground_skill.skill,
        example_source=domain.example_source(belief_name=belief_name),
        belief_name=belief_name,
    )


def make_skill_belief_models(
    *, ground_skills: tuple[GroundSkill, ...], domain: PracticeDomain = TOSSING3D_DOMAIN
) -> tuple[dict[GroundSkill, SkillBeliefModel], dict[str, SkillBeliefModel]]:
    """Associate every practice skill with explicit updates or the default no-op."""
    by_ground_skill = {
        ground_skill: skill_belief_model(ground_skill=ground_skill, domain=domain)
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
    domain: PracticeDomain = TOSSING3D_DOMAIN,
) -> Tossing3DBeliefState:
    """Independent priors for every modeled practice skill.

    Omitting model retains the legacy prior for standalone numerical consumers;
    the POMDP method selects its configured Bayesian model explicitly.
    """
    names = [name for name, _ in domain.belief_sources]
    names.extend(additional_skill_names)
    if include_human_reset:
        from .tossing3d_constants import RESET_OPPOSITE_SIDE_SKILL, RESET_SAME_SIDE_SKILL

        names.extend((RESET_SAME_SIDE_SKILL, RESET_OPPOSITE_SIDE_SKILL))
    if domain.initial_competence_alpha is not None or domain.initial_competence_beta is not None:
        settings = inference_config or InferenceConfig()
        inference_config = settings.model_copy(
            update={
                key: value
                for key, value in (
                    ("initial_competence_alpha", domain.initial_competence_alpha),
                    ("initial_competence_beta", domain.initial_competence_beta),
                )
                if value is not None
            }
        )
    beliefs: dict[str, ConcreteSkillBelief] = {}
    seed_offsets = {
        name: index for index, name in enumerate(dict.fromkeys((*domain.prior_seed_order, *names)))
    }
    for name in dict.fromkeys(names):
        index = seed_offsets[name]
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
        skill_beliefs=beliefs,
        sampler_training={name: SamplerTrainingState() for name in domain.sampler_skills},
    )


def mean_competence(*, belief: SkillBelief) -> float:
    return belief.mean_competence()


def mean_learning_rate(*, belief: SkillBelief) -> float:
    return belief.mean_learning_rate()


def mean_cost(*, belief: ParticleFilterBelief | BayesianSkillBelief) -> float:
    return belief.mean_cost()


def condition_skill_belief(*, belief: ConcreteSkillBelief, success: bool) -> ConcreteSkillBelief:
    """Condition on a greedy-policy outcome without pretending a refit occurred."""
    return SearchForecastCache.condition(belief=belief, success=success)


def refit_skill_belief(
    *, belief: ConcreteSkillBelief, training_examples: int
) -> ConcreteSkillBelief:
    """Forecast after training without modifying the observed belief."""
    return belief.refit(training_examples=training_examples)


def refit_skill_after_example(
    *,
    state: Tossing3DBeliefState,
    belief_name: str,
    sampler_name: str,
    learning_rate_process_noise_std: float = 0.0,
    advance_cycle: bool = False,
) -> Tossing3DBeliefState:
    """`refit_belief_state` for the one skill whose sampler just refitted on an example.

    Every other skill's belief, pending count and sampler bookkeeping is untouched:
    their samplers did not train. A skill with nothing pending did not train either.
    """
    count = state.pending_examples.get(belief_name, 0)
    if count == 0:
        return state
    stepped = refit_belief_state(
        state=state.model_copy(
            update={
                "skill_beliefs": {belief_name: state.skill_beliefs[belief_name]},
                "pending_examples": {belief_name: count},
                "sampler_training": {
                    name: training
                    for name, training in state.sampler_training.items()
                    if name == sampler_name
                },
            }
        ),
        learning_rate_process_noise_std=learning_rate_process_noise_std,
        advance_cycle=advance_cycle,
    )
    pending_examples = dict(state.pending_examples)
    del pending_examples[belief_name]
    return state.model_copy(
        update={
            "skill_beliefs": {**state.skill_beliefs, **stepped.skill_beliefs},
            "pending_examples": pending_examples,
            "sampler_training": {**state.sampler_training, **stepped.sampler_training},
        }
    )


def refit_belief_state(
    *,
    state: Tossing3DBeliefState,
    learning_rate_process_noise_std: float = 0.0,
    advance_cycle: bool = False,
) -> Tossing3DBeliefState:
    """Apply the configured transition; S/F is the only competence evidence.

    The training clock is the notebook's m: every attempt that entered the
    sampler's training data advances it, whether or not the fit was one-class.
    Real cycle boundaries also reset evidence and ancestry bookkeeping when no
    training occurred, and a zero-example forecast is still a predict step.
    """
    assert learning_rate_process_noise_std >= 0.0
    refitted_beliefs: dict[str, ConcreteSkillBelief] = {}
    for skill_name, belief in state.skill_beliefs.items():
        count = state.pending_examples.get(skill_name, 0)
        refitted: ConcreteSkillBelief
        if isinstance(belief, BayesianSkillBelief):
            refitted = (
                belief.advance_cycle(training_examples=count)
                if advance_cycle
                else SearchForecastCache.refit(belief=belief, training_examples=count)
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
