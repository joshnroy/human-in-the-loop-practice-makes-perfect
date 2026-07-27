import numpy as np
import pytest

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    ADJACENT,
    HAND_EMPTY,
    HOLDING,
    ITEM_IN_BIN,
    ROBOT_IN_ROOM,
)
from hitl_pmp.environments.tossingroom.skills import TossingRoomSkills

_ENV = TossingRoomEnvironment()
_ROBOT = TossingRoomEnvironment.robot
_RECYCLING = TossingRoomEnvironment.recycling
_RECYCLING_BIN = TossingRoomEnvironment.recycling_bin


def _state():
    return _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)


def test_pickup_declares_its_parameters_and_effects() -> None:
    skill = TossingRoomSkills.PICKUP
    assert skill.name == "Pickup"
    assert skill.param_dim == 0
    robot, item, room = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
        LiftedAtom(predicate=HAND_EMPTY, variables=(robot,)),
    })
    assert skill.add_effects == frozenset({LiftedAtom(predicate=HOLDING, variables=(robot, item))})
    assert skill.delete_effects == frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(robot,))})


def test_move_room_requires_adjacency() -> None:
    skill = TossingRoomSkills.MOVE_ROOM
    assert skill.name == "MoveRoom"
    assert skill.param_dim == 0
    robot, from_room, to_room = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, from_room)),
        LiftedAtom(predicate=ADJACENT, variables=(from_room, to_room)),
    })
    assert skill.add_effects == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, to_room))
    })


def test_throw_declares_one_continuous_force_param() -> None:
    skill = TossingRoomSkills.THROW
    assert skill.name == "Throw"
    assert skill.param_dim == 1
    robot, item, _bin, _room = skill.parameters
    assert LiftedAtom(predicate=HOLDING, variables=(robot, item)) in skill.preconditions
    assert LiftedAtom(predicate=ITEM_IN_BIN, variables=(item, _bin)) in skill.add_effects


def test_press_adds_bin_empty_for_both_bins() -> None:
    skill = TossingRoomSkills.PRESS
    assert skill.name == "Press"
    assert skill.param_dim == 0
    assert len(skill.add_effects) == 2  # BinEmpty(recycling_bin) and BinEmpty(trash_bin)


def test_sample_params_is_empty_for_zero_dim_skills() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM,
        objects=(_ROBOT, _ENV.get_rooms()[3], _ENV.get_rooms()[2]),
    )
    params = TossingRoomSkills.sample_params(
        ground_skill=ground_skill, rng=np.random.default_rng(0)
    )
    assert params.shape == (0,)


def test_sample_params_for_throw_is_a_single_value_in_unit_interval() -> None:
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.THROW,
        objects=(_ROBOT, _RECYCLING, _RECYCLING_BIN, _ENV.get_rooms()[_ENV.recycling_bin_room]),
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        params = TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)
        assert params.shape == (1,)
        assert 0.0 <= params[0] <= 1.0


def test_compute_action_for_pickup_encodes_the_item_kind() -> None:
    state = _state()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.PICKUP, objects=(_ROBOT, _RECYCLING, _ENV.get_rooms()[3])
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_PICKUP),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.0,
    ]


def test_compute_action_for_move_room_encodes_the_destination_index() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [float(TossingRoomEnvironment.SKILL_MOVE_ROOM), 2.0, 0.0]


def test_compute_action_for_throw_uses_the_sampled_force() -> None:
    state = _state()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.THROW,
        objects=(_ROBOT, _RECYCLING, _RECYCLING_BIN, _ENV.get_rooms()[_ENV.recycling_bin_room]),
    )
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.array([0.42]), state=state
    )
    assert action.tolist() == [
        float(TossingRoomEnvironment.SKILL_THROW),
        float(TossingRoomEnvironment.RECYCLING_KIND),
        0.42,
    ]


def test_compute_action_dispatches_by_value_not_identity() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    move = TossingRoomSkills.MOVE_ROOM
    reconstructed = Skill(
        name=move.name,
        parameters=move.parameters,
        preconditions=move.preconditions,
        add_effects=move.add_effects,
        delete_effects=move.delete_effects,
        param_dim=move.param_dim,
    )
    assert reconstructed is not TossingRoomSkills.MOVE_ROOM
    assert reconstructed == TossingRoomSkills.MOVE_ROOM
    ground_skill = GroundSkill(skill=reconstructed, objects=(_ROBOT, rooms[3], rooms[2]))
    action = TossingRoomSkills.compute_action(
        ground_skill=ground_skill, params=np.zeros(0), state=state
    )
    assert action.tolist() == [float(TossingRoomEnvironment.SKILL_MOVE_ROOM), 2.0, 0.0]


def test_compute_action_rejects_an_unknown_skill() -> None:
    unknown = Skill(
        name="Unknown",
        parameters=(),
        preconditions=frozenset(),
        add_effects=frozenset(),
        delete_effects=frozenset(),
        param_dim=0,
    )
    ground_skill = GroundSkill(skill=unknown, objects=())
    with pytest.raises(ValueError, match="Unknown skill"):
        TossingRoomSkills.compute_action(
            ground_skill=ground_skill, params=np.zeros(0), state=_state()
        )


def test_move_room_ground_skill_grounds_preconditions() -> None:
    rooms = _ENV.get_rooms()
    ground_skill = GroundSkill(
        skill=TossingRoomSkills.MOVE_ROOM, objects=(_ROBOT, rooms[3], rooms[2])
    )
    assert ground_skill.add_effects == frozenset({
        GroundAtom(predicate=ROBOT_IN_ROOM, objects=(_ROBOT, rooms[2]))
    })
