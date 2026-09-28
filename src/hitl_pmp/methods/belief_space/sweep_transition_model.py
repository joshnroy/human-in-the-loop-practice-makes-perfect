"""Provider-owned effects and learned failure effects for Sweep practice."""

from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import GroundAtom

from .failure_effect_model import EmpiricalFailureEffects
from .sweep_observation_model import SweepBeliefs
from .tossing3d_transition_model import (
    TransitionBranch,
    apply_success_effects,
    estimated_action_cost,
)
from .types.belief_state import Tossing3DBeliefState
from .types.competence_evidence import CompetenceEvidence
from .types.failure_effects import FailureEffectCount
from .types.search_state import Tossing3DSearchState


class SweepTransitions:
    @staticmethod
    def outcomes(
        *,
        environment_state: Tossing3DSearchState,
        state: Tossing3DBeliefState,
        action: GroundSkill,
        ground_skills: tuple[GroundSkill, ...],
        effects: dict[
            GroundSkill, tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]]
        ],
        exploration_epsilon: float,
        trainable_skill_names: tuple[str, ...],
        human_skill_names: tuple[str, ...],
        random_competences: dict[str, float],
        failure_effect_counts: tuple[FailureEffectCount, ...] = (),
        competence_evidence: CompetenceEvidence = CompetenceEvidence.NON_EPSILON,
    ) -> tuple[TransitionBranch, ...]:
        assert action in ground_skills and action.preconditions <= environment_state.true_atoms
        name = action.skill.name
        cost = estimated_action_cost(state=state, action=action)
        charged = state.model_copy(update={"accumulated_cost": state.accumulated_cost + cost})
        if name in human_skill_names:
            return (
                (
                    1.0,
                    charged,
                    apply_success_effects(
                        true_atoms=environment_state.true_atoms,
                        ground_skill=action,
                        effects=effects,
                    ),
                ),
            )
        trainable = name in trainable_skill_names
        training = state.sampler_training.get(name)
        mixed = training is None or training.fitted_mixed_classes
        epsilon = exploration_epsilon if trainable and mixed else 0.0
        greedy = (
            (SamplerConsultation.INFORMED if mixed else SamplerConsultation.UNINFORMATIVE)
            if trainable
            else SamplerConsultation.NO_SAMPLER
        )
        model = SweepBeliefs.models(
            ground_skills=(action,), trainable_skill_names=trainable_skill_names
        )[0][action]
        branches: list[TransitionBranch] = []
        choices = [(False, 1 - epsilon, state.skill_beliefs[name].mean_competence(), greedy)]
        if epsilon:
            choices.append((
                True,
                epsilon,
                random_competences[name],
                SamplerConsultation.EPSILON_RANDOM,
            ))
        for random, choice_probability, competence, consultation in choices:
            for success, outcome_probability in ((True, competence), (False, 1 - competence)):
                probability = choice_probability * outcome_probability
                if probability <= 0:
                    continue
                next_state = model.observe_outcome(
                    state=charged,
                    success=success,
                    was_random_exploration=random,
                    resample=False,
                    condition_competence=competence_evidence.admits(consultation=consultation),
                )
                if trainable:
                    next_state = model.observe_training_example(state=next_state, success=success)
                successors = (
                    (
                        (
                            1.0,
                            apply_success_effects(
                                true_atoms=environment_state.true_atoms,
                                ground_skill=action,
                                effects=effects,
                            ),
                        ),
                    )
                    if success
                    else EmpiricalFailureEffects.outcomes(
                        counts=failure_effect_counts,
                        ground_skill=action,
                        true_atoms=environment_state.true_atoms,
                        was_random_exploration=random,
                    )
                )
                branches.extend(
                    (probability * weight, next_state, atoms) for weight, atoms in successors
                )
        return tuple(branches)
