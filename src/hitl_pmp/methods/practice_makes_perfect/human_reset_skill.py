from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.grounding import SkillGrounder

# The one name `_EesEpisode.step` checks for to intercept this ground skill before it
# would otherwise be dispatched through the normal controller/skill-execution path --
# see that method. Public so a test (and the planner's parsed plan) can compare against
# it without importing the builder that produced it.
ASK_FOR_RESET_TASK_INITIAL_NAME = "ask_for_reset_task_initial"

# Same interception contract as the name above, for the sibling ground skill built by
# `build_ask_for_reset_random_task` -- see that method and `_EesEpisode.step`.
ASK_FOR_RESET_RANDOM_TASK_NAME = "ask_for_reset_random_task"


class HumanResetSkillBuilder:
    """Builds the two per-episode ground skills that let EES's planner ask a human
    for a reset, priced via `--ask-for-reset-task-initial-cost`/
    `--ask-for-reset-random-task-cost` and injected into `ground_skill_costs` by
    `EesMethod.plan_to` (which also does the injecting; this only builds operators).

    `ask_for_reset_task_initial` restores THIS period's task-initial state.
    `ask_for_reset_random_task` is dispatched onto a freshly sampled task instead,
    but its operator is built from the SAME `task_initial_atoms` -- sound because
    Tossing3D's task family is shape-invariant (every sampled task abstracts to the
    same atoms; verified empirically). Not a general guarantee for other domains.

    Built fresh inside `plan_to` rather than declared once in `skills()`, since
    "reset to this task's own init_atoms" is per-episode data, not fixed PDDL text.
    Made STRIPS-sound by naming every atom NOT in init_atoms as a delete effect and
    every atom that IS as an add effect (disjoint by construction, so applying both
    always leaves the state exactly init_atoms). One grounding per skill, binding
    all `objects`.

    A static-method container, never instantiated, same as this project's other
    business-logic classes."""

    @staticmethod
    def build_ask_for_reset_task_initial(
        *,
        objects: tuple[Object, ...],
        predicates: tuple[Predicate, ...],
        init_atoms: frozenset[GroundAtom],
    ) -> GroundSkill:
        return HumanResetSkillBuilder._build(
            name=ASK_FOR_RESET_TASK_INITIAL_NAME,
            objects=objects,
            predicates=predicates,
            init_atoms=init_atoms,
        )

    @staticmethod
    def build_ask_for_reset_random_task(
        *,
        objects: tuple[Object, ...],
        predicates: tuple[Predicate, ...],
        init_atoms: frozenset[GroundAtom],
    ) -> GroundSkill:
        """Same construction as `build_ask_for_reset_task_initial` -- see this class's
        own docstring for why `init_atoms` here is `task_initial_atoms` rather than
        whatever a freshly sampled task's own initial atoms happen to be, and what
        that assumes about the domain."""
        return HumanResetSkillBuilder._build(
            name=ASK_FOR_RESET_RANDOM_TASK_NAME,
            objects=objects,
            predicates=predicates,
            init_atoms=init_atoms,
        )

    @staticmethod
    def _build(
        *,
        name: str,
        objects: tuple[Object, ...],
        predicates: tuple[Predicate, ...],
        init_atoms: frozenset[GroundAtom],
    ) -> GroundSkill:
        variable_by_object = {obj: Variable(name=obj.name, type=obj.type) for obj in objects}
        universe = SkillGrounder.all_possible_ground_atoms(objects=objects, predicates=predicates)
        skill = Skill(
            name=name,
            parameters=tuple(variable_by_object[obj] for obj in objects),
            # No real precondition: always applicable during practice -- see
            # EesMethod.plan_to, which is the only caller and only ever offers this
            # ground skill while the episode is practicing.
            preconditions=frozenset(),
            add_effects=HumanResetSkillBuilder._lifted(
                atoms=init_atoms, variable_by_object=variable_by_object
            ),
            delete_effects=HumanResetSkillBuilder._lifted(
                atoms=universe - init_atoms, variable_by_object=variable_by_object
            ),
            param_dim=0,
        )
        return GroundSkill(skill=skill, objects=objects)

    @staticmethod
    def _lifted(
        *, atoms: frozenset[GroundAtom], variable_by_object: dict[Object, Variable]
    ) -> frozenset[LiftedAtom]:
        return frozenset(
            LiftedAtom(
                predicate=atom.predicate,
                variables=tuple(variable_by_object[obj] for obj in atom.objects),
            )
            for atom in atoms
        )
