"""The SweepOffShelf3D operator model, planned off-simulator."""

import itertools

import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import (
    CUBE_ORDER,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    IN_BOX,
    ON_FLOOR,
    ON_SHELF,
    WIPER_HOME,
    WIPER_ON_FLOOR,
    SweepOffShelf3DPredicates,
)
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.planning.fast_downward import FastDownwardPlanner
from hitl_pmp.planning.grounding import SkillGrounder


def _atom(*, predicate, objects):
    return GroundAtom(predicate=predicate, objects=objects)


def _start(*, on_shelf=O.cubes, on_floor=(), in_box=()):
    atoms = {
        _atom(predicate=HAND_EMPTY, objects=(O.robot,)),
        _atom(predicate=WIPER_HOME, objects=(O.wiper,)),
    }
    atoms |= {_atom(predicate=ON_SHELF, objects=(cube,)) for cube in on_shelf}
    atoms |= {_atom(predicate=ON_FLOOR, objects=(cube,)) for cube in on_floor}
    atoms |= {_atom(predicate=IN_BOX, objects=(cube, O.box)) for cube in in_box}
    return frozenset(atoms | SweepOffShelf3DPredicates.static_atoms())


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
    fallen = _start() - {_atom(predicate=WIPER_HOME, objects=(O.wiper,))} | {
        _atom(predicate=WIPER_ON_FLOOR, objects=(O.wiper,))
    }
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
