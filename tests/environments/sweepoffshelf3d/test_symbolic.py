"""The SweepOffShelf3D operator model, planned off-simulator."""

import itertools

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    ANY_CUBE_OFF_SHELF,
    ANY_CUBE_ON_SHELF,
    CUBE_ORDER,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    IN_BOX,
    ON_FLOOR,
    ON_SHELF,
    PICK_PLANNABLE,
    WIPER_AWAY,
    WIPER_HOME,
    WIPER_ON_FLOOR,
    SweepOffShelf3DAtoms,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder


def _atom(*, predicate, objects):
    return GroundAtom(predicate=predicate, objects=objects)


def _summarized(*, atoms):
    """What the backend's classifier would add: the summary atoms derived from the
    per-object ones, through the one function both use."""
    by_name = {predicate.name: predicate for predicate in SweepOffShelf3DPredicates.all()}
    objects = {obj.name: obj for obj in O.all_objects()}
    named = frozenset(
        (atom.predicate.name, tuple(obj.name for obj in atom.objects))
        for atom in atoms
        if atom.predicate != CUBE_ORDER
    )
    derived = SweepOffShelf3DAtoms.with_summaries(atoms=named) - named
    return atoms | frozenset(
        _atom(predicate=by_name[name], objects=tuple(objects[n] for n in names))
        for name, names in derived
    )


def _start(*, on_shelf=O.cubes, on_floor=(), in_box=(), wiper_home=True, unplannable=()):
    """An observed state; every cube's pick plans unless it is in `unplannable`."""
    atoms = {_atom(predicate=HAND_EMPTY, objects=(O.robot,))}
    atoms |= {
        _atom(predicate=PICK_PLANNABLE, objects=(O.robot, cube))
        for cube in O.cubes
        if cube not in unplannable
    }
    if wiper_home:
        atoms.add(_atom(predicate=WIPER_HOME, objects=(O.wiper,)))
    else:
        atoms.add(_atom(predicate=WIPER_ON_FLOOR, objects=(O.wiper,)))
    atoms |= {_atom(predicate=ON_SHELF, objects=(cube,)) for cube in on_shelf}
    atoms |= {_atom(predicate=ON_FLOOR, objects=(cube,)) for cube in on_floor}
    atoms |= {_atom(predicate=IN_BOX, objects=(cube, O.box)) for cube in in_box}
    return _summarized(atoms=frozenset(atoms | SweepOffShelf3DPredicates.static_atoms()))


def _holding_wiper(*, atoms):
    return _apply(ground=_groundings(skill=S.PICK_WIPER, atoms=atoms)[0], atoms=atoms)


def _groundings(*, skill, atoms):
    return SkillGrounder.applicable_ground_skills(
        skills=(skill,), objects=O.all_objects(), true_atoms=atoms
    )


def _apply(*, ground, atoms):
    assert ground.preconditions <= atoms
    added = set(ground.add_effects)
    for effect in ground.conditional_add_effects:
        if effect.conditions <= atoms:
            added |= set(effect.add_effects)
    return frozenset((atoms - ground.delete_effects) | added)


def test_the_goal_is_every_cube_in_the_box():
    assert SweepOffShelf3DPredicates.goal_atoms() == frozenset(
        _atom(predicate=IN_BOX, objects=(cube, O.box)) for cube in O.cubes
    )


def test_cube_order_pins_one_sweep_grounding():
    holding = _start() - {_atom(predicate=HAND_EMPTY, objects=(O.robot,))} | {
        _atom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper))
    }
    groundings = _groundings(skill=S.SWEEP_SHELF, atoms=holding)
    assert len(groundings) == 1
    assert groundings[0].objects[-5:] == O.cubes


def test_cube_order_is_a_chain_over_cube_names():
    assert SweepOffShelf3DPredicates.static_atoms() == frozenset(
        _atom(predicate=CUBE_ORDER, objects=(a, b)) for a, b in itertools.pairwise(O.cubes)
    )


@pytest.mark.parametrize("on_shelf", [O.cubes, O.cubes[:2], O.cubes[3:]])
def test_a_sweep_moves_every_shelf_cube_and_only_those(*, on_shelf):
    elsewhere = tuple(cube for cube in O.cubes if cube not in on_shelf)
    atoms = _start(on_shelf=on_shelf, on_floor=elsewhere)
    atoms = _apply(ground=_groundings(skill=S.PICK_WIPER, atoms=atoms)[0], atoms=atoms)
    after = _apply(ground=_groundings(skill=S.SWEEP_SHELF, atoms=atoms)[0], atoms=atoms)
    for cube in on_shelf:
        assert _atom(predicate=IN_BOX, objects=(cube, O.box)) in after
        assert _atom(predicate=ON_SHELF, objects=(cube,)) not in after
    for cube in elsewhere:
        assert _atom(predicate=IN_BOX, objects=(cube, O.box)) not in after
        assert _atom(predicate=ON_FLOOR, objects=(cube,)) in after
    assert _atom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper)) in after


def test_pick_and_place_moves_one_cube():
    atoms = _start()
    pick = next(
        g
        for g in _groundings(skill=S.PICK_CUBE_FROM_SHELF, atoms=atoms)
        if g.objects[1] == O.cubes[2]
    )
    atoms = _apply(ground=pick, atoms=atoms)
    assert _atom(predicate=HOLDING_CUBE, objects=(O.robot, O.cubes[2])) in atoms
    assert _atom(predicate=HAND_EMPTY, objects=(O.robot,)) not in atoms
    place = _groundings(skill=S.PLACE_CUBE_IN_BOX, atoms=atoms)
    assert len(place) == 1
    atoms = _apply(ground=place[0], atoms=atoms)
    assert _atom(predicate=IN_BOX, objects=(O.cubes[2], O.box)) in atoms
    assert _atom(predicate=HAND_EMPTY, objects=(O.robot,)) in atoms


def test_a_floor_cube_is_picked_and_placed():
    atoms = _start(on_shelf=O.cubes[1:], on_floor=O.cubes[:1])
    pick = _groundings(skill=S.PICK_CUBE_FROM_FLOOR, atoms=atoms)
    assert [g.objects[1] for g in pick] == [O.cubes[0]]
    atoms = _apply(ground=pick[0], atoms=atoms)
    assert _atom(predicate=ON_FLOOR, objects=(O.cubes[0],)) not in atoms


def test_return_wiper_frees_the_hand():
    atoms = _apply(ground=_groundings(skill=S.PICK_WIPER, atoms=_start())[0], atoms=_start())
    atoms = _apply(ground=_groundings(skill=S.RETURN_WIPER, atoms=atoms)[0], atoms=atoms)
    assert {
        _atom(predicate=HAND_EMPTY, objects=(O.robot,)),
        _atom(predicate=WIPER_HOME, objects=(O.wiper,)),
    } <= atoms


def test_floor_atoms_are_never_added_by_any_skill():
    """OnFloor and WiperOnFloor are observed outcomes, never promised by an operator."""
    for skill in (*S.robot_skills(), *S.human_skills()):
        promised = {atom.predicate for atom in skill.add_effects} | {
            atom.predicate
            for effect in skill.conditional_add_effects
            for atom in effect.add_effects
        }
        assert ON_FLOOR not in promised
        assert WIPER_ON_FLOOR not in promised


def test_reset_cubes_returns_every_cube_to_the_shelf():
    atoms = _start(on_shelf=(), on_floor=O.cubes[:2], in_box=O.cubes[2:])
    atoms = _apply(ground=_groundings(skill=S.RESET_CUBES, atoms=atoms)[0], atoms=atoms)
    for cube in O.cubes:
        assert _atom(predicate=ON_SHELF, objects=(cube,)) in atoms
        assert _atom(predicate=IN_BOX, objects=(cube, O.box)) not in atoms
        assert _atom(predicate=ON_FLOOR, objects=(cube,)) not in atoms


def test_reset_cubes_takes_a_held_cube_and_empties_the_hand():
    atoms = _start()
    pick = _groundings(skill=S.PICK_CUBE_FROM_SHELF, atoms=atoms)[0]
    atoms = _apply(ground=pick, atoms=atoms)
    atoms = _apply(ground=_groundings(skill=S.RESET_CUBES, atoms=atoms)[0], atoms=atoms)
    assert _atom(predicate=HAND_EMPTY, objects=(O.robot,)) in atoms
    assert not any(atom.predicate == HOLDING_CUBE for atom in atoms)


def test_reset_wiper_returns_a_held_or_fallen_wiper():
    atoms = _apply(ground=_groundings(skill=S.PICK_WIPER, atoms=_start())[0], atoms=_start())
    atoms = _apply(ground=_groundings(skill=S.RESET_WIPER, atoms=atoms)[0], atoms=atoms)
    assert {
        _atom(predicate=HAND_EMPTY, objects=(O.robot,)),
        _atom(predicate=WIPER_HOME, objects=(O.wiper,)),
    } <= atoms
    fallen = _start(wiper_home=False)
    atoms = _apply(ground=_groundings(skill=S.RESET_WIPER, atoms=fallen)[0], atoms=fallen)
    assert _atom(predicate=WIPER_HOME, objects=(O.wiper,)) in atoms
    assert _atom(predicate=WIPER_ON_FLOOR, objects=(O.wiper,)) not in atoms


def test_only_the_sweep_has_continuous_parameters():
    assert S.SWEEP_SHELF.param_dim == 2
    for skill in (*S.robot_skills(), *S.human_skills()):
        if skill is not S.SWEEP_SHELF:
            assert skill.param_dim == 0


def test_sweep_bounds_cover_heights_above_a_cube():
    """The sampled head height must reach past a 2 cm cube so the range holds
    parameters that cannot work as well as ones that can."""
    (h_low, h_high), _ = S.SWEEP_BOUNDS
    assert h_low == 0.0
    assert h_high > 2 * O.cube_half_size


def test_fast_downward_plans_to_the_goal_from_the_start():
    plan = FastDownwardPlanner.plan(
        skills=S.robot_skills(),
        objects=O.all_objects(),
        predicates=SweepOffShelf3DPredicates.all(),
        types=O.all_types(),
        init_atoms=_start(),
        goal=SweepOffShelf3DPredicates.goal_atoms(),
    )
    assert [ground.skill.name for ground in plan] == ["PickWiper", "SweepShelf"]


_LAYOUTS = {
    "all on the shelf": {"on_shelf": O.cubes},
    "all on the floor": {"on_shelf": (), "on_floor": O.cubes},
    "mixed": {"on_shelf": O.cubes[:2], "on_floor": O.cubes[2:4], "in_box": O.cubes[4:]},
    "one left on the floor": {"on_shelf": (), "on_floor": O.cubes[:1], "in_box": O.cubes[1:]},
}


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
@pytest.mark.parametrize("wiper_home", [True, False])
@pytest.mark.parametrize("holding", ["nothing", "wiper", "cube"])
def test_the_robot_alone_can_reach_the_goal_from_every_kind_of_scene(
    *, layout, wiper_home, holding
):
    """The always-ask wrapper's stranded trigger fires only when no robot-only plan
    reaches the goal. Every cube can be picked from the shelf or the floor and placed
    without the wiper, so even a fallen wiper -- which only the human can recover --
    strands nothing, and p (the human's competence at reset_cubes, which robot-only
    planning never uses) cannot make the trigger fire."""
    if holding == "wiper" and not wiper_home:
        pytest.skip("a fallen wiper cannot be in the hand")
    atoms = _start(wiper_home=wiper_home, **_LAYOUTS[layout])
    if holding == "wiper":
        atoms = _holding_wiper(atoms=atoms)
    elif holding == "cube":
        pick = next(
            g
            for skill in (S.PICK_CUBE_FROM_SHELF, S.PICK_CUBE_FROM_FLOOR)
            for g in _groundings(skill=skill, atoms=atoms)
        )
        atoms = _apply(ground=pick, atoms=atoms)
    plan = FastDownwardPlanner.plan(
        skills=S.robot_skills(),
        objects=O.all_objects(),
        predicates=SweepOffShelf3DPredicates.all(),
        types=O.all_types(),
        init_atoms=atoms,
        goal=SweepOffShelf3DPredicates.goal_atoms(),
    )
    assert plan


# --- competence semantics: no skill can succeed vacuously ---------------------


def test_summary_atoms_are_derived_from_the_per_object_ones():
    full = _start()
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) in full
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) not in full
    assert _atom(predicate=WIPER_AWAY, objects=(O.wiper,)) not in full
    empty = _start(on_shelf=(), in_box=O.cubes, wiper_home=False)
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) not in empty
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) in empty
    assert _atom(predicate=WIPER_AWAY, objects=(O.wiper,)) in empty
    partial = _start(on_shelf=O.cubes[:1], on_floor=O.cubes[1:])
    assert {
        _atom(predicate=ANY_CUBE_ON_SHELF, objects=()),
        _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()),
    } <= partial


def test_a_sweep_of_an_empty_shelf_is_not_applicable():
    atoms = _holding_wiper(atoms=_start(on_shelf=(), in_box=O.cubes))
    assert _groundings(skill=S.SWEEP_SHELF, atoms=atoms) == []


@pytest.mark.parametrize("count", [1, 3, 5])
def test_a_sweep_of_n_cubes_promises_exactly_those_n_in_the_box(*, count):
    on_shelf = O.cubes[:count]
    atoms = _holding_wiper(atoms=_start(on_shelf=on_shelf, in_box=O.cubes[count:]))
    (sweep,) = _groundings(skill=S.SWEEP_SHELF, atoms=atoms)
    promised = sweep.expected_add_effects(before_atoms=atoms)
    assert {atom for atom in promised if atom.predicate == IN_BOX} == {
        _atom(predicate=IN_BOX, objects=(cube, O.box)) for cube in on_shelf
    }
    after = _apply(ground=sweep, atoms=atoms)
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) not in after
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) in after


def test_reset_cubes_is_not_applicable_when_every_cube_is_on_the_shelf():
    """The cancelled batch: with all cubes still on the shelf, reset_cubes
    "succeeded" every time, because its promised effects already held."""
    assert _groundings(skill=S.RESET_CUBES, atoms=_start()) == []


def test_reset_wiper_is_not_applicable_when_the_wiper_is_home():
    assert _groundings(skill=S.RESET_WIPER, atoms=_start()) == []


def test_picking_a_cube_off_the_shelf_makes_reset_cubes_applicable():
    atoms = _start()
    pick = _groundings(skill=S.PICK_CUBE_FROM_SHELF, atoms=atoms)[0]
    atoms = _apply(ground=pick, atoms=atoms)
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) in atoms
    (reset,) = _groundings(skill=S.RESET_CUBES, atoms=atoms)
    after = _apply(ground=reset, atoms=atoms)
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) in after
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) not in after


def test_the_wiper_is_away_exactly_while_it_is_not_home():
    atoms = _holding_wiper(atoms=_start())
    assert _atom(predicate=WIPER_AWAY, objects=(O.wiper,)) in atoms
    returned = _apply(ground=_groundings(skill=S.RETURN_WIPER, atoms=atoms)[0], atoms=atoms)
    assert _atom(predicate=WIPER_AWAY, objects=(O.wiper,)) not in returned
    reset = _apply(ground=_groundings(skill=S.RESET_WIPER, atoms=atoms)[0], atoms=atoms)
    assert _atom(predicate=WIPER_AWAY, objects=(O.wiper,)) not in reset


@pytest.mark.parametrize(
    "skill",
    [
        S.RESET_CUBES,
        S.RESET_WIPER,
        S.SWEEP_SHELF,
        S.RETURN_CUBE_TO_SHELF_FROM_BOX,
        S.RETURN_CUBE_TO_SHELF_FROM_FLOOR,
    ],
)
def test_no_skill_promises_nothing_from_any_state_it_applies_in(*, skill):
    """A skill whose promised effects already hold would be scored a success with
    nothing done; every state it can be grounded in must leave it something to do."""
    states = [
        _start(on_shelf=O.cubes[:k], **{elsewhere: O.cubes[k:]}, wiper_home=home)
        for k in range(len(O.cubes) + 1)
        for home in (True, False)
        for elsewhere in ("in_box", "on_floor")
    ]
    states += [
        _holding_wiper(atoms=state)
        for state in states
        if _groundings(skill=S.PICK_WIPER, atoms=state)
    ]
    for atoms in states:
        for ground in _groundings(skill=skill, atoms=atoms):
            assert not ground.expected_add_effects(before_atoms=atoms) <= atoms


# --- the robot returns a cube to the shelf itself -----------------------------

RETURNS = [S.RETURN_CUBE_TO_SHELF_FROM_BOX, S.RETURN_CUBE_TO_SHELF_FROM_FLOOR]


@pytest.mark.parametrize(("skill", "where"), zip(RETURNS, ("in_box", "on_floor"), strict=True))
def test_returning_a_cube_puts_that_cube_on_the_shelf_with_the_hand_empty(*, skill, where):
    atoms = _start(on_shelf=O.cubes[1:], **{where: O.cubes[:1]})
    (ground,) = _groundings(skill=skill, atoms=atoms)
    assert ground.objects[1] == O.cubes[0]
    after = _apply(ground=ground, atoms=atoms)
    assert _atom(predicate=ON_SHELF, objects=(O.cubes[0],)) in after
    assert _atom(predicate=IN_BOX, objects=(O.cubes[0], O.box)) not in after
    assert _atom(predicate=ON_FLOOR, objects=(O.cubes[0],)) not in after
    assert _atom(predicate=HAND_EMPTY, objects=(O.robot,)) in after
    assert not any(atom.predicate == HOLDING_CUBE for atom in after)


def test_a_return_promises_a_cube_on_the_shelf_and_leaves_any_off_shelf_alone():
    """As `PickCubeFromShelf` cannot know it took the last shelf cube, a return cannot
    know it brought back the last stray one; every observed state is reclassified."""
    atoms = _start(on_shelf=(), in_box=O.cubes)
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) not in atoms
    ground = _groundings(skill=S.RETURN_CUBE_TO_SHELF_FROM_BOX, atoms=atoms)[0]
    after = _apply(ground=ground, atoms=atoms)
    assert _atom(predicate=ANY_CUBE_ON_SHELF, objects=()) in after
    assert _atom(predicate=ANY_CUBE_OFF_SHELF, objects=()) in after


@pytest.mark.parametrize("skill", RETURNS)
def test_a_cube_is_returned_only_with_an_empty_hand_and_only_from_off_the_shelf(*, skill):
    assert _groundings(skill=skill, atoms=_start()) == []
    stray = _start(on_shelf=O.cubes[2:], on_floor=O.cubes[:1], in_box=O.cubes[1:2])
    assert len(_groundings(skill=skill, atoms=stray)) == 1
    assert _groundings(skill=skill, atoms=_holding_wiper(atoms=stray)) == []


def test_the_return_skills_are_robot_skills():
    assert set(S.RETURN_CUBE_SKILLS) == set(RETURNS)
    assert set(RETURNS) <= set(S.robot_skills())
