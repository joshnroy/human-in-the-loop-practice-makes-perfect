"""Covers `HumanResetSkillBuilder`: the per-episode ground skill that lets EES's own
planner choose to ask a human to reset the world to a task's own init_atoms.

The load-bearing property is that applying this operator's effects to ANY state
produces exactly `init_atoms` -- not "whatever init_atoms adds on top of the current
state", which is what a naive add-effects-only operator would give. That is what the
delete-effects-cover-the-universe construction below pins."""

from hitl_pmp.core.method.types import Skill
from hitl_pmp.core.problem.environment.types import Object, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate
from hitl_pmp.methods.practice_makes_perfect.human_reset_skill import (
    ASK_FOR_RESET_RANDOM_TASK_NAME,
    ASK_FOR_RESET_TASK_INITIAL_NAME,
    HumanResetSkillBuilder,
)
from hitl_pmp.planning.fast_downward import FastDownwardPlanner

_ROBOT_TYPE = Type(name="robot", feature_names=())
_CELL_TYPE = Type(name="cell", feature_names=())
_ROBOT = Object(name="robot", type=_ROBOT_TYPE)
_CELL0 = Object(name="cell0", type=_CELL_TYPE)
_CELL1 = Object(name="cell1", type=_CELL_TYPE)

_IN_CELL = Predicate(
    name="InCell", types=(_ROBOT_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)
_ADJACENT = Predicate(
    name="Adjacent", types=(_CELL_TYPE, _CELL_TYPE), holds=lambda state, objects: True
)

_OBJECTS = (_ROBOT, _CELL0, _CELL1)
_PREDICATES = (_IN_CELL, _ADJACENT)
_INIT_ATOMS = frozenset({
    GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL0)),
    GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL1)),
})


def _build() -> Skill:
    ground = HumanResetSkillBuilder.build_ask_for_reset_task_initial(
        objects=_OBJECTS, predicates=_PREDICATES, init_atoms=_INIT_ATOMS
    )
    return ground


def test_the_ground_skill_is_named_and_bound_to_every_object() -> None:
    ground = _build()
    assert ground.skill.name == ASK_FOR_RESET_TASK_INITIAL_NAME
    assert ground.objects == _OBJECTS


def test_it_has_no_precondition_so_it_is_always_applicable() -> None:
    assert _build().preconditions == frozenset()


def test_add_effects_are_exactly_init_atoms() -> None:
    ground = _build()
    assert ground.add_effects == _INIT_ATOMS


def test_delete_effects_are_every_possible_atom_outside_init_atoms() -> None:
    """The universe over these objects/predicates minus init_atoms -- every possible
    InCell/Adjacent combination that is NOT one of the two atoms declared initial,
    including ones that are false today (nothing here is currently true at all, since
    this builder never reads a State) and reflexive Adjacent(cell, cell) atoms."""
    ground = _build()
    assert ground.delete_effects == frozenset({
        GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL1, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL1, _CELL1)),
    })


def test_add_and_delete_effects_never_overlap() -> None:
    """Structural soundness: an atom cannot be both asserted and retracted by the same
    operator, or STRIPS application would be ambiguous about the result."""
    ground = _build()
    assert ground.add_effects.isdisjoint(ground.delete_effects)


def _build_random() -> Skill:
    return HumanResetSkillBuilder.build_ask_for_reset_random_task(
        objects=_OBJECTS, predicates=_PREDICATES, init_atoms=_INIT_ATOMS
    )


def test_the_random_task_reset_skill_is_named_and_bound_to_every_object() -> None:
    """Same operator shape as ask_for_reset_task_initial (see this module's own
    docstring for why: init_atoms here is task_initial_atoms, not a freshly sampled
    task's own atoms), distinguished only by name -- the one thing _EesEpisode.step
    needs to dispatch it as a HumanRandomTaskResetRequested rather than a
    HumanHelpRequested."""
    ground = _build_random()
    assert ground.skill.name == ASK_FOR_RESET_RANDOM_TASK_NAME
    assert ground.objects == _OBJECTS


def test_the_random_task_reset_skill_has_no_precondition_so_it_is_always_applicable() -> None:
    assert _build_random().preconditions == frozenset()


def test_the_random_task_reset_skills_add_effects_are_exactly_init_atoms() -> None:
    ground = _build_random()
    assert ground.add_effects == _INIT_ATOMS


def test_the_random_task_reset_skills_delete_effects_cover_everything_outside_init_atoms() -> None:
    ground = _build_random()
    assert ground.delete_effects == frozenset({
        GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL0, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL1, _CELL0)),
        GroundAtom(predicate=_ADJACENT, objects=(_CELL1, _CELL1)),
    })


def test_the_two_reset_skills_are_distinct_ground_skills_despite_identical_effects() -> None:
    """The load-bearing distinction between the two is the name alone -- everything
    else about the built operator is identical when both are built from the same
    init_atoms, which is exactly the case EesMethod.plan_to constructs (both from this
    period's own task_initial_atoms). GroundSkill equality goes through Skill equality,
    which includes the name, so the two must compare unequal."""
    task_initial = _build()
    random_task = _build_random()
    assert task_initial != random_task
    assert task_initial.add_effects == random_task.add_effects
    assert task_initial.delete_effects == random_task.delete_effects


def test_a_real_fast_downward_accepts_the_generated_operator() -> None:
    """The empty precondition and the (possibly large) delete-effect list have to
    survive real PDDL translation and search, not just construction -- this is the one
    place that gets exercised end to end rather than only asserted on the Python
    objects."""
    ground = _build()
    plan = FastDownwardPlanner.plan(
        skills=(ground.skill,),
        predicates=_PREDICATES,
        types=(_ROBOT_TYPE, _CELL_TYPE),
        objects=_OBJECTS,
        init_atoms=frozenset({
            GroundAtom(predicate=_IN_CELL, objects=(_ROBOT, _CELL1)),
        }),
        goal=_INIT_ATOMS,
        ground_skill_costs={ground: 0.5},
    )
    assert plan == [ground]
