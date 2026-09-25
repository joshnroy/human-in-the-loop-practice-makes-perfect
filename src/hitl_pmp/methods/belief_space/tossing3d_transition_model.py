"""Symbolic practice transitions for Tossing3D."""

from functools import cache

from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    PICK_SKILLS,
    RESET_SKILLS,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    condition_skill_belief,
    mean_competence,
    skill_belief_model,
)
from hitl_pmp.methods.belief_space.types.belief_state import (
    ConcreteSkillBelief,
    Tossing3DBeliefState,
)
from hitl_pmp.methods.belief_space.types.competence_evidence import CompetenceEvidence
from hitl_pmp.methods.belief_space.types.failure_effects import FailureEffectCount
from hitl_pmp.methods.belief_space.types.particle_filter_belief import ParticleFilterBelief
from hitl_pmp.methods.belief_space.types.search_state import Tossing3DSearchState

TransitionBranch = tuple[float, Tossing3DBeliefState, frozenset[GroundAtom]]


def estimated_action_cost(*, state: Tossing3DBeliefState, action: GroundSkill) -> float:
    """Use a certainty-equivalent posterior cost during tractable tree search.

    Real executions update the full cost posterior. Expanding cost-observation
    branches here would multiply the already exponential outcome tree by the
    particle count at every depth, so planning deliberately uses its posterior
    mean and holds that estimate fixed within one search.
    """
    belief = state.skill_beliefs.get(action.skill.name)
    if isinstance(belief, (ParticleFilterBelief, BayesianSkillBelief)):
        return belief.mean_cost()
    return action.evaluate_practice_cost()


@cache
def render_atoms(*, true_atoms: frozenset[GroundAtom]) -> tuple[str, ...]:
    """Render each immutable symbolic state once for diagnostics."""
    return tuple(sorted(str(atom) for atom in true_atoms))


def make_tossing3d_search_state(
    *, state: Tossing3DBeliefState, true_atoms: frozenset[GroundAtom]
) -> Tossing3DSearchState:
    return Tossing3DSearchState(
        state=state,
        true_atoms=true_atoms,
        atoms=render_atoms(true_atoms=true_atoms),
    )


def transition_belief_state(
    *,
    state: Tossing3DBeliefState,
    added_cost: float,
    toss_belief: ConcreteSkillBelief | None = None,
    added_training_examples: int = 0,
) -> Tossing3DBeliefState:
    skill_beliefs = dict(state.skill_beliefs)
    pending_examples = dict(state.pending_examples)
    if toss_belief is not None:
        skill_beliefs[TOSS_SKILL] = toss_belief
    if added_training_examples:
        pending_examples[TOSS_SKILL] = pending_examples.get(TOSS_SKILL, 0) + added_training_examples
    return state.model_copy(
        update={
            "skill_beliefs": skill_beliefs,
            "pending_examples": pending_examples,
            "accumulated_cost": state.accumulated_cost + added_cost,
        }
    )


def apply_success_effects(
    *,
    true_atoms: frozenset[GroundAtom],
    ground_skill: GroundSkill,
    effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ],
) -> frozenset[GroundAtom]:
    add_effects, delete_effects, ignore_effects = effects[ground_skill]
    kept = {
        atom
        for atom in true_atoms
        if atom.predicate not in ignore_effects and atom not in delete_effects
    }
    conditional_additions = {
        atom
        for effect in ground_skill.conditional_add_effects
        if effect.conditions <= true_atoms
        for atom in effect.add_effects
    }
    return frozenset(kept | set(add_effects) | conditional_additions)


def transition_outcomes(
    *,
    environment_state: Tossing3DSearchState,
    state: Tossing3DBeliefState,
    action: GroundSkill,
    ground_skills: tuple[GroundSkill, ...],
    effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ],
    exploration_epsilon: float,
    random_toss_competence: float,
    failure_effect_counts: tuple[FailureEffectCount, ...] = (),
    competence_evidence: CompetenceEvidence = CompetenceEvidence.NON_EPSILON,
) -> tuple[TransitionBranch, ...]:
    assert action in ground_skills
    assert action.preconditions <= environment_state.true_atoms
    cost = estimated_action_cost(state=state, action=action)
    if action.skill.name in PICK_SKILLS:
        return binary_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            probability=mean_competence(belief=state.skill_beliefs[PICK_SKILL]),
            cost=cost,
            effects=effects,
            failure_effect_counts=failure_effect_counts,
        )
    if action.skill.name == OPEN_GRIPPER_SKILL:
        return binary_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            probability=mean_competence(belief=state.skill_beliefs[OPEN_GRIPPER_SKILL]),
            cost=cost,
            effects=effects,
            failure_effect_counts=failure_effect_counts,
        )
    if action.skill.name in RESET_SKILLS:
        # The reset API either completes successfully or raises and aborts the
        # run. Its empirical performance telemetry is not uncertain dynamics.
        # Real completions still update S/F, costs and refits, but a forecast must
        # not invent evidence about a known mechanism or move its cost posterior.
        return (
            (
                1.0,
                transition_belief_state(state=state, added_cost=cost),
                apply_success_effects(
                    true_atoms=environment_state.true_atoms,
                    ground_skill=action,
                    effects=effects,
                ),
            ),
        )
    assert action.skill.name == TOSS_SKILL
    return toss_outcomes(
        state=state,
        true_atoms=environment_state.true_atoms,
        ground_skill=action,
        toss_cost=cost,
        exploration_epsilon=exploration_epsilon,
        random_toss_competence=random_toss_competence,
        effects=effects,
        failure_effect_counts=failure_effect_counts,
        competence_evidence=competence_evidence,
    )


def binary_outcomes(
    *,
    state: Tossing3DBeliefState,
    true_atoms: frozenset[GroundAtom],
    ground_skill: GroundSkill,
    probability: float,
    cost: float,
    effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ],
    failure_effect_counts: tuple[FailureEffectCount, ...] = (),
) -> tuple[TransitionBranch, ...]:
    outcomes: list[TransitionBranch] = []
    for success, branch_probability in ((True, probability), (False, 1.0 - probability)):
        if branch_probability <= 0.0:
            continue
        effect_outcomes = (
            (
                (
                    1.0,
                    apply_success_effects(
                        true_atoms=true_atoms, ground_skill=ground_skill, effects=effects
                    ),
                ),
            )
            if success
            else EmpiricalFailureEffects.outcomes(
                counts=failure_effect_counts,
                ground_skill=ground_skill,
                true_atoms=true_atoms,
                was_random_exploration=False,
            )
        )
        next_state = skill_belief_model(ground_skill=ground_skill).observe_outcome(
            state=transition_belief_state(state=state, added_cost=cost),
            success=success,
            was_random_exploration=False,
            resample=False,
        )
        outcomes.extend(
            (branch_probability * effect_probability, next_state, next_true_atoms)
            for effect_probability, next_true_atoms in effect_outcomes
        )
    return tuple(outcomes)


def toss_outcomes(
    *,
    state: Tossing3DBeliefState,
    true_atoms: frozenset[GroundAtom],
    ground_skill: GroundSkill,
    toss_cost: float,
    exploration_epsilon: float,
    random_toss_competence: float,
    effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ],
    failure_effect_counts: tuple[FailureEffectCount, ...] = (),
    competence_evidence: CompetenceEvidence = CompetenceEvidence.NON_EPSILON,
) -> tuple[TransitionBranch, ...]:
    """Branch on the greedy/epsilon draw, then on S/F.

    Each branch conditions competence exactly when `competence_evidence` would
    admit the real attempt it imagines, so search forecasts under the evidence
    rule the robot will actually apply. The greedy draw's consultation is taken
    as `INFORMED` from a mixed-class fit and `UNINFORMATIVE` from a one-class or
    unfitted one; the tie-fraction fallback of a mixed fit is not modelled.
    """
    branches: list[TransitionBranch] = []
    training = state.sampler_training.get(TOSS_SKILL)
    if training is not None and not training.fitted_mixed_classes:
        exploration_epsilon = 0.0
    greedy_consultation = (
        SamplerConsultation.UNINFORMATIVE
        if training is not None and not training.fitted_mixed_classes
        else SamplerConsultation.INFORMED
    )
    for is_random, choice_probability, success_probability in (
        (
            False,
            1.0 - exploration_epsilon,
            mean_competence(belief=state.skill_beliefs[TOSS_SKILL]),
        ),
        (True, exploration_epsilon, random_toss_competence),
    ):
        for success, observation_probability in (
            (True, success_probability),
            (False, 1.0 - success_probability),
        ):
            probability = choice_probability * observation_probability
            if probability <= 0.0:
                continue
            belief = state.skill_beliefs[TOSS_SKILL]
            if competence_evidence.admits(
                consultation=SamplerConsultation.EPSILON_RANDOM
                if is_random
                else greedy_consultation
            ):
                belief = condition_skill_belief(belief=belief, success=success)
            effect_outcomes = (
                (
                    (
                        1.0,
                        apply_success_effects(
                            true_atoms=true_atoms, ground_skill=ground_skill, effects=effects
                        ),
                    ),
                )
                if success
                else EmpiricalFailureEffects.outcomes(
                    counts=failure_effect_counts,
                    ground_skill=ground_skill,
                    true_atoms=true_atoms,
                    was_random_exploration=is_random,
                )
            )
            next_state = transition_belief_state(
                state=state,
                added_cost=toss_cost,
                toss_belief=belief,
            )
            next_state = skill_belief_model(ground_skill=ground_skill).observe_training_example(
                state=next_state, success=success
            )
            branches.extend(
                (probability * effect_probability, next_state, next_true_atoms)
                for effect_probability, next_true_atoms in effect_outcomes
            )
    return tuple(branches)
