"""Symbolic practice transitions for Tossing3D."""

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.methods.belief_space.tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    RESET_SKILL,
    TOSS_SKILL,
)
from hitl_pmp.methods.belief_space.tossing3d_observation_model import (
    SKILL_BELIEF_MODELS,
    condition_skill_belief,
    mean_competence,
)
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.search_state import Tossing3DSearchState
from hitl_pmp.methods.belief_space.types.skill_belief import SkillBelief

TransitionBranch = tuple[float, Tossing3DBeliefState, frozenset[GroundAtom]]


def make_tossing3d_search_state(
    *, state: Tossing3DBeliefState, true_atoms: frozenset[GroundAtom]
) -> Tossing3DSearchState:
    return Tossing3DSearchState(
        state=state,
        true_atoms=true_atoms,
        atoms=tuple(sorted(str(atom) for atom in true_atoms)),
    )


def transition_belief_state(
    *,
    state: Tossing3DBeliefState,
    added_cost: float,
    toss_belief: SkillBelief | None = None,
    added_training_examples: int = 0,
) -> Tossing3DBeliefState:
    skill_beliefs = dict(state.skill_beliefs)
    pending_examples = dict(state.pending_examples)
    if toss_belief is not None:
        skill_beliefs[TOSS_SKILL] = toss_belief
    if added_training_examples:
        pending_examples[TOSS_SKILL] = (
            pending_examples.get(TOSS_SKILL, 0) + added_training_examples
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
    kept = {
        atom
        for atom in true_atoms
        if atom.predicate not in ignore_effects and atom not in delete_effects
    }
    return frozenset(kept | set(add_effects))


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
) -> tuple[TransitionBranch, ...]:
    assert action in ground_skills
    assert action.preconditions <= environment_state.true_atoms
    if action.skill.name == PICK_SKILL:
        return binary_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            probability=mean_competence(belief=state.skill_beliefs[PICK_SKILL]),
            cost=action.evaluate_practice_cost(),
            effects=effects,
        )
    if action.skill.name == OPEN_GRIPPER_SKILL:
        return binary_outcomes(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            probability=mean_competence(belief=state.skill_beliefs[OPEN_GRIPPER_SKILL]),
            cost=action.evaluate_practice_cost(),
            effects=effects,
        )
    if action.skill.name == RESET_SKILL:
        return deterministic_outcome(
            state=state,
            true_atoms=environment_state.true_atoms,
            ground_skill=action,
            cost=action.evaluate_practice_cost(),
            effects=effects,
        )
    assert action.skill.name == TOSS_SKILL
    return toss_outcomes(
        state=state,
        true_atoms=environment_state.true_atoms,
        ground_skill=action,
        toss_cost=action.evaluate_practice_cost(),
        exploration_epsilon=exploration_epsilon,
        random_toss_competence=random_toss_competence,
        effects=effects,
    )


def deterministic_outcome(
    *,
    state: Tossing3DBeliefState,
    true_atoms: frozenset[GroundAtom],
    ground_skill: GroundSkill,
    cost: float,
    effects: dict[
        GroundSkill,
        tuple[frozenset[GroundAtom], frozenset[GroundAtom], frozenset[object]],
    ],
) -> tuple[TransitionBranch, ...]:
    return (
        (
            1.0,
            transition_belief_state(state=state, added_cost=cost),
            apply_success_effects(
                true_atoms=true_atoms, ground_skill=ground_skill, effects=effects
            ),
        ),
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
) -> tuple[TransitionBranch, ...]:
    outcomes = []
    for success, branch_probability in ((True, probability), (False, 1.0 - probability)):
        if branch_probability <= 0.0:
            continue
        next_true_atoms = (
            apply_success_effects(true_atoms=true_atoms, ground_skill=ground_skill, effects=effects)
            if success
            else true_atoms
        )
        outcomes.append((
            branch_probability,
            SKILL_BELIEF_MODELS[ground_skill.skill].observe_outcome(
                state=transition_belief_state(state=state, added_cost=cost),
                success=success,
                was_random_exploration=False,
            ),
            next_true_atoms,
        ))
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
) -> tuple[TransitionBranch, ...]:
    branches: list[TransitionBranch] = []
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
            belief = (
                state.skill_beliefs[TOSS_SKILL]
                if is_random
                else condition_skill_belief(
                    belief=state.skill_beliefs[TOSS_SKILL], success=success
                )
            )
            next_true_atoms = (
                apply_success_effects(
                    true_atoms=true_atoms, ground_skill=ground_skill, effects=effects
                )
                if success
                else true_atoms
            )
            branches.append((
                probability,
                transition_belief_state(
                    state=state,
                    added_cost=toss_cost,
                    toss_belief=belief,
                    added_training_examples=1,
                ),
                next_true_atoms,
            ))
    return tuple(branches)
