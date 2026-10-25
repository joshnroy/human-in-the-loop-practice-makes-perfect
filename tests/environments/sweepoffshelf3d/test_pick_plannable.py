"""`PickPlannable` on SweepOffShelf3D, off-simulator: when the dry run is made, and what
the operators that require it do on a state where it is absent."""

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.pick_plannable import (
    KB_PICK_PLANNABLE,
    PickPlannableAtoms,
)
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    HAND_EMPTY,
    IN_BOX,
    ON_FLOOR,
    ON_SHELF,
    PICK_PLANNABLE,
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import BenchmarkCase
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.environments.sweepoffshelf3d.types import SweepOffShelf3DState
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder

WEDGED = O.cubes[0]


def _named(*, on_shelf=(), on_floor=(), in_box=(), hand="empty"):
    atoms = {("WiperHome", (O.wiper.name,))}
    if hand == "empty":
        atoms.add(("HandEmpty", (O.robot.name,)))
    elif hand == "wiper":
        atoms = {("HoldingWiper", (O.robot.name, O.wiper.name))}
    atoms |= {("OnShelf", (c.name,)) for c in on_shelf}
    atoms |= {("OnFloor", (c.name,)) for c in on_floor}
    atoms |= {("InBox", (c.name, O.box.name)) for c in in_box}
    return SweepOffShelf3DAtoms.with_summaries(atoms=frozenset(atoms))


class _DryRuns:
    def __init__(self, *, refused=()):
        self.refused = {c.name for c in refused}
        self.asked: list[str] = []

    def __call__(self, cube):  # noqa: PLR0917 -- `with_verdicts` calls it positionally
        self.asked.append(cube)
        return cube not in self.refused


def _observed(*, refused=(), **layout):
    named = PickPlannableAtoms.with_verdicts(
        atoms=_named(**layout), plannable=_DryRuns(refused=refused)
    )
    state = SweepOffShelf3DState(data={}, abstract_atoms=named)
    return (
        SkillGrounder.abstract_state(
            state=state,
            objects=O.all_objects(),
            predicates=SweepOffShelf3DSkillProvider.model_construct().predicates(),
        )
        | SweepOffShelf3DPredicates.static_atoms()
    )


def _plannable(*, cube):
    return GroundAtom(predicate=PICK_PLANNABLE, objects=(O.robot, cube))


def _ground(*, skill, atoms):
    return [
        g.objects[1]
        for g in SkillGrounder.applicable_ground_skills(
            skills=(skill,), objects=O.all_objects(), true_atoms=atoms
        )
    ]


# --- when the dry run is made -------------------------------------------------


def test_the_dry_run_is_made_for_exactly_the_off_shelf_cubes_with_the_hand_empty():
    dry = _DryRuns()
    atoms = PickPlannableAtoms.with_verdicts(
        atoms=_named(on_shelf=O.cubes[:2], on_floor=O.cubes[2:4], in_box=O.cubes[4:]),
        plannable=dry,
    )
    assert dry.asked == [c.name for c in O.cubes[2:]]
    assert {(KB_PICK_PLANNABLE, (O.robot.name, c.name)) for c in O.cubes} <= atoms


def test_a_refused_off_shelf_cube_has_no_atom_and_the_others_keep_theirs():
    atoms = PickPlannableAtoms.with_verdicts(
        atoms=_named(on_shelf=O.cubes[1:], on_floor=(WEDGED,)),
        plannable=_DryRuns(refused=(WEDGED,)),
    )
    assert (KB_PICK_PLANNABLE, (O.robot.name, WEDGED.name)) not in atoms
    for cube in O.cubes[1:]:
        assert (KB_PICK_PLANNABLE, (O.robot.name, cube.name)) in atoms


def test_no_dry_run_while_the_hand_is_full_and_every_cube_carries_the_atom():
    dry = _DryRuns(refused=O.cubes)
    atoms = PickPlannableAtoms.with_verdicts(
        atoms=_named(on_floor=O.cubes[:2], in_box=O.cubes[2:], hand="wiper"), plannable=dry
    )
    assert dry.asked == []
    assert {(KB_PICK_PLANNABLE, (O.robot.name, c.name)) for c in O.cubes} <= atoms


# --- what it gates ------------------------------------------------------------

GATED = (S.PICK_CUBE_FROM_FLOOR, S.RETURN_CUBE_TO_SHELF_FROM_FLOOR)


@pytest.mark.parametrize("skill", GATED)
def test_a_wedged_unpickable_floor_cube_cannot_be_picked_or_returned(*, skill):
    atoms = _observed(on_shelf=O.cubes[1:], on_floor=(WEDGED,), refused=(WEDGED,))
    assert GroundAtom(predicate=ON_FLOOR, objects=(WEDGED,)) in atoms
    assert _plannable(cube=WEDGED) not in atoms
    assert _ground(skill=skill, atoms=atoms) == []
    plannable = _observed(on_shelf=O.cubes[1:], on_floor=(WEDGED,))
    assert _ground(skill=skill, atoms=plannable) == [WEDGED]


def test_an_unpickable_cube_in_the_box_cannot_be_returned():
    atoms = _observed(on_shelf=O.cubes[1:], in_box=(WEDGED,), refused=(WEDGED,))
    assert GroundAtom(predicate=IN_BOX, objects=(WEDGED, O.box)) in atoms
    assert _ground(skill=S.RETURN_CUBE_TO_SHELF_FROM_BOX, atoms=atoms) == []
    plannable = _observed(on_shelf=O.cubes[1:], in_box=(WEDGED,))
    assert _ground(skill=S.RETURN_CUBE_TO_SHELF_FROM_BOX, atoms=plannable) == [WEDGED]


def test_the_shelf_pick_is_not_gated():
    assert PICK_PLANNABLE not in {a.predicate for a in S.PICK_CUBE_FROM_SHELF.preconditions}
    gated = {
        skill.name
        for skill in (*S.robot_skills(), *S.human_skills())
        if PICK_PLANNABLE in {a.predicate for a in skill.preconditions}
    }
    assert gated == {
        "PickCubeFromFloor",
        "ReturnCubeToShelfFromFloor",
        "ReturnCubeToShelfFromBox",
    }


def test_reset_cubes_is_unchanged_and_puts_the_wedged_cube_back_on_the_shelf():
    for effects in (S.RESET_CUBES.add_effects, S.RESET_CUBES.delete_effects):
        assert PICK_PLANNABLE not in {a.predicate for a in effects}
    atoms = _observed(on_shelf=O.cubes[1:], on_floor=(WEDGED,), refused=(WEDGED,))
    (reset,) = SkillGrounder.applicable_ground_skills(
        skills=(S.RESET_CUBES,), objects=O.all_objects(), true_atoms=atoms
    )
    after = (atoms - reset.delete_effects) | reset.add_effects
    assert GroundAtom(predicate=ON_SHELF, objects=(WEDGED,)) in after
    assert GroundAtom(predicate=ON_FLOOR, objects=(WEDGED,)) not in after


def test_with_the_wedged_cube_unpickable_the_way_to_the_goal_is_the_human_reset():
    atoms = _observed(on_shelf=O.cubes[1:], on_floor=(WEDGED,), refused=(WEDGED,))
    goal = SweepOffShelf3DPredicates.goal_atoms()
    plan_args = {
        "objects": O.all_objects(),
        "predicates": SweepOffShelf3DPredicates.all(),
        "types": O.all_types(),
        "init_atoms": atoms,
        "goal": goal,
    }
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(skills=S.robot_skills(), **plan_args)
    plan = FastDownwardPlanner.plan(skills=(*S.robot_skills(), *S.human_skills()), **plan_args)
    assert "reset_cubes" in [g.skill.name for g in plan]


def test_both_planners_abstract_through_the_provider_that_declares_it():
    """EES (`EesMethod.abstract_state`) and the POMDP method (`Tossing3DPomdpMethod`)
    both abstract with `skill_provider.predicates()`; it must name the atom, or every
    gated operator is inapplicable everywhere."""
    provider = SweepOffShelf3DSkillProvider.model_construct()
    assert PICK_PLANNABLE in provider.predicates()
    atoms = _observed(on_floor=O.cubes, refused=())
    assert HAND_EMPTY in {a.predicate for a in atoms}
    assert {_plannable(cube=c) for c in O.cubes} <= atoms


def test_the_skill_benchmark_runs_a_gated_skill_whatever_the_dry_run_says():
    """The benchmark measures the controller, so its denominators hold the states
    `PickPlannable` would rule out; its applicability check ignores the atom."""
    atoms = sorted(f"{n}({','.join(o)})" for n, o in _named(on_floor=O.cubes))
    for skill in ("PickCubeFromFloor", "ReturnCubeToShelfFromFloor"):
        case = BenchmarkCase(skill=skill, cube=WEDGED.name, params=[])
        assert StateCapture.applicable(case=case, atoms=atoms)
