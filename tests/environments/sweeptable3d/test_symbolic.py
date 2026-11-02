"""Whole-table promises, terminal applicability and human resets without MuJoCo."""

import itertools

import numpy as np
import pytest

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects
from hitl_pmp.environments.sweeptable3d.objects import SweepTable3DObjects as O
from hitl_pmp.environments.sweeptable3d.predicates import (
    ANY_CUBE_OFF_TABLE,
    ANY_CUBE_ON_TABLE,
    BALLS_RESTORED,
    CUBE_ORDER,
    DYNAMIC_PREDICATES,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    IN_BOX,
    ON_FLOOR,
    ON_TABLE,
    RESET_BALLS_NEEDED,
    TASK_ACTIVE,
    WIPER_AWAY,
    WIPER_HOME,
    WIPER_ON_FLOOR,
    SweepTable3DAtoms,
    SweepTable3DPredicates,
)
from hitl_pmp.environments.sweeptable3d.skills import SweepTable3DSkills as S
from hitl_pmp.planning.fast_downward import FastDownwardPlanner, PlanningFailure
from hitl_pmp.planning.grounding import SkillGrounder


class AbstractState(State):
    data: dict[Object, np.ndarray]
    abstract_atoms: frozenset[tuple[str, tuple[str, ...]]]
    table_center_y: float = 0.0
    table_edge_x: float = 1.373
    sweep_start_x: float = 1.60


class SymbolicScene:
    @staticmethod
    def atom(*, predicate, objects=()):
        return GroundAtom(predicate=predicate, objects=objects)

    @staticmethod
    def state(
        *,
        on_table=O.cubes,
        on_floor=(),
        in_box=(),
        holding=None,
        wiper_home=True,
        active=True,
        bin_y=0.0,
        cube_y=0.0,
        table_center_y=0.0,
        table_edge_x=1.373,
        sweep_start_x=1.60,
    ):
        atoms = {("OnTable", (cube.name,)) for cube in on_table if cube != holding}
        atoms |= {("OnFloor", (cube.name,)) for cube in on_floor}
        atoms |= {("InBox", (cube.name, O.box.name)) for cube in in_box}
        if active:
            atoms.add(("TaskActive", ()))
        if holding == O.wiper:
            atoms.add(("HoldingWiper", (O.robot.name, O.wiper.name)))
        else:
            atoms.add(("WiperHome" if wiper_home else "WiperOnFloor", (O.wiper.name,)))
            if holding is None:
                atoms.add(("HandEmpty", (O.robot.name,)))
            else:
                atoms.add(("HoldingCube", (O.robot.name, holding.name)))
        data = {obj: np.zeros(obj.type.dim) for obj in (*O.all_objects(), O.scene)}
        for cube in O.cubes:
            data[cube][:] = (1.50, cube_y, 0.50)
        data[O.box][:] = (1.10, bin_y - 0.15, 1.35, bin_y + 0.15, 0.30)
        return AbstractState(
            data=data,
            abstract_atoms=SweepTable3DAtoms.with_summaries(atoms=frozenset(atoms)),
            table_center_y=table_center_y,
            table_edge_x=table_edge_x,
            sweep_start_x=sweep_start_x,
        )

    @staticmethod
    def atoms(*, state):
        return SkillGrounder.abstract_state(
            state=state, objects=O.all_objects(), predicates=SweepTable3DPredicates.all()
        )

    @staticmethod
    def groundings(*, skill, atoms):
        return SkillGrounder.applicable_ground_skills(
            skills=(skill,), objects=O.all_objects(), true_atoms=atoms
        )

    @staticmethod
    def apply(*, ground, atoms):
        assert ground.preconditions <= atoms
        return (atoms - ground.delete_effects) | ground.expected_add_effects(before_atoms=atoms)


def test_feature_schemas_match_shelf_but_have_unique_type_names():
    for object_type, shelf_type in zip(
        (*O.all_types(), O.scene_type),
        (*SweepOffShelf3DObjects.all_types(), SweepOffShelf3DObjects.scene_type),
        strict=True,
    ):
        assert object_type.feature_names == shelf_type.feature_names
        assert object_type.name != shelf_type.name
    assert len(O.cubes) == 5
    assert O.scene not in O.all_objects()
    assert O.scene_type.feature_names == ("seed", "steps_taken")


@pytest.mark.parametrize("predicate", DYNAMIC_PREDICATES)
def test_classifiers_require_an_abstraction_and_lookup_only_named_atoms(*, predicate):
    objects = tuple(
        next(obj for obj in O.all_objects() if obj.type == typ) for typ in predicate.types
    )
    state = SymbolicScene.state()
    named = (predicate.name, tuple(obj.name for obj in objects))
    state.abstract_atoms = frozenset()
    assert not predicate.holds(state, objects)
    state.abstract_atoms = frozenset({named})
    assert predicate.holds(state, objects)
    with pytest.raises(ValueError, match="no abstraction attached"):
        predicate.holds(State(data={}), objects)


def test_cube_order_is_exactly_the_static_chain():
    state = State(data={})
    expected = frozenset(
        SymbolicScene.atom(predicate=CUBE_ORDER, objects=pair)
        for pair in itertools.pairwise(O.cubes)
    )
    assert SweepTable3DPredicates.static_atoms() == expected
    for first, second in itertools.product(O.cubes, repeat=2):
        assert CUBE_ORDER.holds(state, (first, second)) == (
            SymbolicScene.atom(predicate=CUBE_ORDER, objects=(first, second)) in expected
        )


def test_a_sweep_has_one_grounding_containing_every_cube_and_the_live_bin():
    atoms = SymbolicScene.atoms(state=SymbolicScene.state(holding=O.wiper))
    sweeps = SymbolicScene.groundings(skill=S.SWEEP_TABLE, atoms=atoms)
    assert len(sweeps) == 1
    sweep = sweeps[0]
    assert sweep.objects == (O.robot, O.wiper, O.box, *O.cubes)
    assert sweep.add_effects == SweepTable3DPredicates.goal_atoms() | {
        SymbolicScene.atom(predicate=ANY_CUBE_OFF_TABLE),
        SymbolicScene.atom(predicate=RESET_BALLS_NEEDED),
    }
    assert sweep.delete_effects == {
        SymbolicScene.atom(predicate=ANY_CUBE_ON_TABLE),
        SymbolicScene.atom(predicate=TASK_ACTIVE),
        SymbolicScene.atom(predicate=BALLS_RESTORED),
        *(SymbolicScene.atom(predicate=ON_TABLE, objects=(cube,)) for cube in O.cubes),
    }
    after = SymbolicScene.apply(ground=sweep, atoms=atoms)
    assert SweepTable3DPredicates.goal_atoms() <= after
    assert SymbolicScene.atom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper)) in after
    assert sweep.conditional_add_effects == frozenset()


@pytest.mark.parametrize("missing", range(5))
def test_sweep_is_unavailable_if_any_cube_is_off_table(*, missing):
    state = SymbolicScene.state(
        on_table=tuple(cube for index, cube in enumerate(O.cubes) if index != missing),
        in_box=(O.cubes[missing],),
        holding=O.wiper,
    )
    assert (
        SymbolicScene.groundings(skill=S.SWEEP_TABLE, atoms=SymbolicScene.atoms(state=state)) == []
    )


def test_sweep_requires_holding_wiper_and_task_active():
    for state in (SymbolicScene.state(), SymbolicScene.state(holding=O.wiper, active=False)):
        assert (
            SymbolicScene.groundings(skill=S.SWEEP_TABLE, atoms=SymbolicScene.atoms(state=state))
            == []
        )


@pytest.mark.parametrize(
    "skill",
    [
        S.PICK_WIPER,
        S.SWEEP_TABLE,
        S.PICK_CUBE_FROM_TABLE,
        S.PICK_CUBE_FROM_FLOOR,
        S.PLACE_CUBE_IN_BOX,
    ],
)
def test_forward_skills_require_task_active(*, skill):
    assert TASK_ACTIVE in {atom.predicate for atom in skill.preconditions}


@pytest.mark.parametrize("skill", [S.RETURN_WIPER, *S.RETURN_CUBE_SKILLS, *S.human_skills()])
def test_reset_skills_do_not_require_task_active(*, skill):
    assert TASK_ACTIVE not in {atom.predicate for atom in skill.preconditions}


def test_table_pick_and_place_cannot_satisfy_the_sweep_goal():
    atoms = SymbolicScene.atoms(state=SymbolicScene.state(on_table=O.cubes[:1], in_box=O.cubes[1:]))
    pick = SymbolicScene.groundings(skill=S.PICK_CUBE_FROM_TABLE, atoms=atoms)
    assert len(pick) == 1
    atoms = SymbolicScene.apply(ground=pick[0], atoms=atoms)
    place = SymbolicScene.groundings(skill=S.PLACE_CUBE_IN_BOX, atoms=atoms)
    assert len(place) == 1
    after = SymbolicScene.apply(ground=place[0], atoms=atoms)
    assert not SweepTable3DPredicates.goal_atoms() <= after
    assert SymbolicScene.atom(predicate=HAND_EMPTY, objects=(O.robot,)) in after


def test_floor_failure_blocks_delivery_but_allows_return_to_table():
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(on_table=O.cubes[1:], on_floor=O.cubes[:1], active=False)
    )
    for skill in (
        S.PICK_WIPER,
        S.PICK_CUBE_FROM_TABLE,
        S.PICK_CUBE_FROM_FLOOR,
        S.PLACE_CUBE_IN_BOX,
    ):
        assert SymbolicScene.groundings(skill=skill, atoms=atoms) == []
    returns = SymbolicScene.groundings(skill=S.RETURN_CUBE_TO_TABLE_FROM_FLOOR, atoms=atoms)
    assert len(returns) == 1
    after = SymbolicScene.apply(ground=returns[0], atoms=atoms)
    assert SymbolicScene.atom(predicate=ON_TABLE, objects=(O.cubes[0],)) in after
    assert SymbolicScene.atom(predicate=ON_FLOOR, objects=(O.cubes[0],)) not in after
    assert SymbolicScene.atom(predicate=TASK_ACTIVE) in after


def test_human_cube_reset_restores_all_cubes_and_task_active_after_failure():
    state = SymbolicScene.state(
        on_table=(), on_floor=O.cubes[:2], in_box=O.cubes[2:4], holding=O.cubes[4], active=False
    )
    atoms = SymbolicScene.atoms(state=state)
    resets = SymbolicScene.groundings(skill=S.RESET_CUBES, atoms=atoms)
    assert len(resets) == 1
    assert resets[0].objects == (O.robot, O.box, O.wiper, *O.cubes)
    after = SymbolicScene.apply(ground=resets[0], atoms=atoms)
    assert {
        SymbolicScene.atom(predicate=TASK_ACTIVE),
        SymbolicScene.atom(predicate=HAND_EMPTY, objects=(O.robot,)),
        SymbolicScene.atom(predicate=ANY_CUBE_ON_TABLE),
        *(SymbolicScene.atom(predicate=ON_TABLE, objects=(cube,)) for cube in O.cubes),
    } <= after
    assert not any(
        atom.predicate in {IN_BOX, ON_FLOOR, HOLDING_CUBE, ANY_CUBE_OFF_TABLE} for atom in after
    )


def test_human_cube_reset_preserves_a_held_wiper():
    atoms = SymbolicScene.atoms(
        state=SymbolicScene.state(on_table=(), in_box=O.cubes, holding=O.wiper, active=False)
    )
    reset = SymbolicScene.groundings(skill=S.RESET_CUBES, atoms=atoms)[0]
    after = SymbolicScene.apply(ground=reset, atoms=atoms)
    assert SymbolicScene.atom(predicate=HOLDING_WIPER, objects=(O.robot, O.wiper)) in after
    assert SymbolicScene.atom(predicate=HAND_EMPTY, objects=(O.robot,)) not in after


@pytest.mark.parametrize("holding_wiper", [False, True])
def test_human_wiper_reset_handles_fallen_and_held_wipers_after_failure(*, holding_wiper):
    state = SymbolicScene.state(
        holding=O.wiper if holding_wiper else None, wiper_home=False, active=False
    )
    atoms = SymbolicScene.atoms(state=state)
    reset = SymbolicScene.groundings(skill=S.RESET_WIPER, atoms=atoms)[0]
    after = SymbolicScene.apply(ground=reset, atoms=atoms)
    assert SymbolicScene.atom(predicate=WIPER_HOME, objects=(O.wiper,)) in after
    assert SymbolicScene.atom(predicate=HAND_EMPTY, objects=(O.robot,)) in after
    assert not any(atom.predicate in {WIPER_ON_FLOOR, WIPER_AWAY, HOLDING_WIPER} for atom in after)
    assert SymbolicScene.atom(predicate=TASK_ACTIVE) not in after


def test_summaries_are_recomputed_without_inventing_task_active():
    atoms = frozenset({
        ("AnyCubeOnTable", ()),
        ("WiperAway", (O.wiper.name,)),
        ("WiperHome", (O.wiper.name,)),
        *(("InBox", (cube.name, O.box.name)) for cube in O.cubes),
    })
    summarized = SweepTable3DAtoms.with_summaries(atoms=atoms)
    assert ("AnyCubeOffTable", ()) in summarized
    assert ("AnyCubeOnTable", ()) not in summarized
    assert ("WiperAway", (O.wiper.name,)) not in summarized
    assert ("TaskActive", ()) not in summarized
    active = SweepTable3DAtoms.with_summaries(atoms=atoms | {("TaskActive", ())})
    assert ("TaskActive", ()) in active


def test_only_sweep_has_continuous_parameters_and_yaw_stays_below_ninety():
    assert S.SWEEP_BOUNDS == ((-45.0, 45.0), (0.05, 0.30))
    for skill in (*S.robot_skills(), *S.human_skills()):
        assert skill.param_dim == (2 if skill == S.SWEEP_TABLE else 0)
        assert not any(atom.predicate in {ON_FLOOR, WIPER_ON_FLOOR} for atom in skill.add_effects)


def test_human_resets_are_unavailable_when_their_promises_already_hold():
    atoms = SymbolicScene.atoms(state=SymbolicScene.state())
    for skill in S.human_skills():
        assert SymbolicScene.groundings(skill=skill, atoms=atoms) == []


def test_fast_downward_plans_one_sweep_for_the_entire_table():
    plan = FastDownwardPlanner.plan(
        skills=S.robot_skills(),
        objects=O.all_objects(),
        predicates=SweepTable3DPredicates.all(),
        types=O.all_types(),
        init_atoms=SymbolicScene.atoms(state=SymbolicScene.state()),
        goal=SweepTable3DPredicates.goal_atoms(),
    )
    assert [ground.skill.name for ground in plan] == ["PickWiper", "SweepTable"]


def test_fast_downward_cannot_deliver_after_ground_failure():
    with pytest.raises(PlanningFailure):
        FastDownwardPlanner.plan(
            skills=tuple(skill for skill in S.robot_skills() if skill not in S.RETURN_CUBE_SKILLS),
            objects=O.all_objects(),
            predicates=SweepTable3DPredicates.all(),
            types=O.all_types(),
            init_atoms=SymbolicScene.atoms(
                state=SymbolicScene.state(on_table=O.cubes[1:], on_floor=O.cubes[:1], active=False)
            ),
            goal=SweepTable3DPredicates.goal_atoms(),
        )


def test_fast_downward_can_resume_delivery_after_human_cube_reset():
    plan = FastDownwardPlanner.plan(
        skills=(
            *(skill for skill in S.robot_skills() if skill not in S.RETURN_CUBE_SKILLS),
            *S.human_skills(),
        ),
        objects=O.all_objects(),
        predicates=SweepTable3DPredicates.all(),
        types=O.all_types(),
        init_atoms=SymbolicScene.atoms(
            state=SymbolicScene.state(on_table=O.cubes[1:], on_floor=O.cubes[:1], active=False)
        ),
        goal=SweepTable3DPredicates.goal_atoms(),
    )
    assert [ground.skill.name for ground in plan] == ["reset_cubes", "PickWiper", "SweepTable"]


@pytest.mark.parametrize("tool_first", [False, True])
def test_failed_on_table_sweep_requires_explicit_ball_reset_and_tool_home(*, tool_first):
    atoms = SymbolicScene.atoms(state=SymbolicScene.state(holding=O.wiper, active=False))
    sequence = (S.RETURN_WIPER, S.RESET_CUBES) if tool_first else (S.RESET_CUBES, S.RETURN_WIPER)
    for index, skill in enumerate(sequence):
        grounds = SymbolicScene.groundings(skill=skill, atoms=atoms)
        assert len(grounds) == 1
        atoms = SymbolicScene.apply(ground=grounds[0], atoms=atoms)
        assert (SymbolicScene.atom(predicate=TASK_ACTIVE) in atoms) is (index == 1)
