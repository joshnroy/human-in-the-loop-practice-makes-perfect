"""Symbolic practice transitions for Tossing3D."""

from functools import cache
from typing import ClassVar

from hitl_pmp.core.method.types import GroundSkill, SamplerConsultation
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.failure_effect_model import EmpiricalFailureEffects
from hitl_pmp.methods.belief_space.practice_domain import TOSSING3D_DOMAIN, PracticeDomain
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
    key = GroundSkillText.of(ground_skill=action)
    if key in state.expected_execution_costs:
        return state.expected_execution_costs[key]
    belief = state.skill_beliefs.get(action.skill.name)
    if isinstance(belief, (ParticleFilterBelief, BayesianSkillBelief)):
        return belief.mean_cost()
    return action.evaluate_practice_cost()


class GroundSkillText:
    """`str(ground_skill)`, the key `expected_execution_costs` is written under,
    rendered once per object: search prices an action at every transition, and the
    pydantic repr was a fifth of search time.

    Keyed by identity, not equality. A frozen ground skill's text never changes, but
    frozenset reprs follow hash order, so two equal ground skills built apart can
    render differently; each keeps exactly the text `str` gives it. Entries hold
    their ground skill, so an id is never reused while cached.
    """

    MAX_ENTRIES: ClassVar[int] = 4096
    _texts: ClassVar[dict[int, tuple[GroundSkill, str]]] = {}

    @staticmethod
    def of(*, ground_skill: GroundSkill) -> str:
        entry = GroundSkillText._texts.get(id(ground_skill))
        if entry is not None and entry[0] is ground_skill:
            return entry[1]
        text = str(ground_skill)
        if len(GroundSkillText._texts) >= GroundSkillText.MAX_ENTRIES:
            GroundSkillText._texts.clear()
        GroundSkillText._texts[id(ground_skill)] = (ground_skill, text)
        return text


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
    sampler_skill: str = TOSSING3D_DOMAIN.sampler_skill,
) -> Tossing3DBeliefState:
    skill_beliefs = dict(state.skill_beliefs)
    pending_examples = dict(state.pending_examples)
    if toss_belief is not None:
        skill_beliefs[sampler_skill] = toss_belief
    if added_training_examples:
        pending_examples[sampler_skill] = (
            pending_examples.get(sampler_skill, 0) + added_training_examples
        )
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
    # An ignored predicate the operator re-asserts is a functional update: its old
    # atoms are replaced by the added ones (the reset's sides, the toss's CubeAtSide).
    # One it does not re-assert is merely unknown -- the toss's GraspClear, which held
    # after 233/430 real successes -- and a deterministic forecast leaves it as it was
    # rather than claiming it false, which would make every imagined success end in a
    # reset.
    conditional_additions = {
        atom
        for effect in ground_skill.conditional_add_effects
        if effect.conditions <= true_atoms
        for atom in effect.add_effects
    }
    expected_additions = set(add_effects) | conditional_additions
    replaced = set(ignore_effects) & {atom.predicate for atom in expected_additions}
    if ground_skill.skill.name == "OpenGripper":
        replaced.update(ignore_effects)
    replaced.update(
        predicate
        for effect in ground_skill.conditional_ignore_effects
        if effect.conditions <= true_atoms
        for predicate in effect.ignore_effects
    )
    kept = {
        atom for atom in true_atoms if atom.predicate not in replaced and atom not in delete_effects
    }
    return frozenset(kept | expected_additions)


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
    domain: PracticeDomain = TOSSING3D_DOMAIN,
) -> tuple[TransitionBranch, ...]:
    assert action in ground_skills
    assert action.preconditions <= environment_state.true_atoms
    cost = estimated_action_cost(state=state, action=action)
    name = action.skill.name
    if name in domain.deterministic_skills:
        # The reset API either completes successfully or raises and aborts the
        # run. Its empirical performance telemetry is not uncertain dynamics.
        # Real completions still update S/F, costs and refits, but a forecast must
        # not invent evidence about a known mechanism or move its cost posterior.
        # It does advance the reset's training clock, exactly as the real
        # completion will, so search forecasts the same m it will observe.
        pending = dict(state.pending_examples)
        pending[action.skill.name] = pending.get(action.skill.name, 0) + 1
        return (
            (
                1.0,
                transition_belief_state(state=state, added_cost=cost).model_copy(
                    update={"pending_examples": pending}
                ),
                apply_success_effects(
                    true_atoms=environment_state.true_atoms,
                    ground_skill=action,
                    effects=effects,
                ),
            ),
        )
    if name not in domain.sampler_skills:
        belief_name = domain.belief_name(skill_name=action.skill.name)
        assert belief_name is not None, f"{name} has no competence belief"
        return binary_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            probability=mean_competence(belief=state.skill_beliefs[belief_name]),
            cost=cost,
            effects=effects,
            failure_effect_counts=failure_effect_counts,
            domain=domain,
        )
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
        domain=domain,
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
    domain: PracticeDomain = TOSSING3D_DOMAIN,
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
        next_state = skill_belief_model(ground_skill=ground_skill, domain=domain).observe_outcome(
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
    domain: PracticeDomain = TOSSING3D_DOMAIN,
) -> tuple[TransitionBranch, ...]:
    """Branch on the greedy/epsilon draw, then on S/F.

    Each branch conditions competence exactly when `competence_evidence` would
    admit the real attempt it imagines, so search forecasts under the evidence
    rule the robot will actually apply. The greedy draw's consultation is taken
    as `INFORMED` from a mixed-class fit and `UNINFORMATIVE` from a one-class or
    unfitted one; the tie-fraction fallback of a mixed fit is not modelled.
    """
    belief_name = domain.belief_name(skill_name=ground_skill.skill.name)
    assert belief_name is not None
    branches: list[TransitionBranch] = []
    training = state.sampler_training.get(ground_skill.skill.name)
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
            mean_competence(belief=state.skill_beliefs[belief_name]),
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
            belief = state.skill_beliefs[belief_name]
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
                sampler_skill=belief_name,
            )
            next_state = skill_belief_model(
                ground_skill=ground_skill, domain=domain
            ).observe_training_example(state=next_state, success=success)
            branches.extend(
                (probability * effect_probability, next_state, next_true_atoms)
                for effect_probability, next_true_atoms in effect_outcomes
            )
    return tuple(branches)
