from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.planning.grounding import SkillGrounder

# The one name `_EesEpisode.step` checks for to intercept this ground skill before it
# would otherwise be dispatched through the normal controller/skill-execution path --
# see that method. Public so a test (and the planner's parsed plan) can compare against
# it without importing the builder that produced it.
ASK_FOR_RESET_TASK_INITIAL_NAME = "ask_for_reset_task_initial"


class HumanResetSkillBuilder:
    """Builds `ask_for_reset_task_initial`: the one ground skill that lets EES's own
    planner choose to ask a human to put the world back to a task's own initial
    symbolic state, priced like any other ground skill (`--ask-for-reset-task-initial-
    cost`, injected directly into `FastDownwardPlanner`'s `ground_skill_costs` --
    EesMethod.plan_to does the injecting, this only builds the operator).

    **Why this cannot be a domain-general `Skill`, declared once beside `pick_cube`/
    `move_to_toss_location_and_toss`.** A lifted `Skill`'s add/delete effects are fixed
    PDDL text, identical for every problem the domain ever plans over. "Reset to THIS
    task's own init_atoms" is per-episode data -- two tasks of the same domain can have
    different init_atoms -- so it can only be built once `init_atoms` is known, which is
    exactly when `EesMethod.plan_to` is called (it already receives `init_atoms` per
    episode). So this builder is called fresh inside `plan_to`, not registered once in
    a domain's `skills()`.

    **How "reset to exactly init_atoms" becomes a sound STRIPS operator.** A STRIPS
    delete effect only removes the ground atoms it names; whatever it doesn't name
    stays exactly as it was. So achieving "every atom equals init_atoms, regardless of
    what was true before" needs the operator to name every possible atom that is NOT in
    init_atoms as a delete effect (`SkillGrounder.all_possible_ground_atoms(...) -
    init_atoms`) and every atom that IS in init_atoms as an add effect. Since add/delete
    are disjoint by construction (an atom is in init_atoms or it isn't), applying both
    together leaves the state exactly init_atoms no matter what was true beforehand.

    **One parameter per object, all of them bound in the single grounding this
    returns.** A `LiftedAtom` can only reference this skill's own declared parameters
    (`Skill._check_variables_are_declared_parameters`), so every `Object` init_atoms or
    the universe could ever mention needs its own `Variable` -- there is exactly one
    grounding of this skill, binding all of `objects` in order, which is also the only
    grounding `EesMethod.plan_to` ever offers the planner.

    A static-method container, never instantiated, same as every other business-logic
    class in this project."""

    @staticmethod
    def build_ask_for_reset_task_initial(
        *,
        objects: tuple[Object, ...],
        predicates: tuple[Predicate, ...],
        init_atoms: frozenset[GroundAtom],
    ) -> GroundSkill:
        variable_by_object = {obj: Variable(name=obj.name, type=obj.type) for obj in objects}
        universe = SkillGrounder.all_possible_ground_atoms(objects=objects, predicates=predicates)
        skill = Skill(
            name=ASK_FOR_RESET_TASK_INITIAL_NAME,
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
