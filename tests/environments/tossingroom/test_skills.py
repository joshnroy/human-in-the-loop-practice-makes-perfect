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
    PILE_IN_ROOM,
    ROBOT_IN_ROOM,
)
from hitl_pmp.environments.tossingroom.skill_provider import TossingRoomSkillProvider
from hitl_pmp.environments.tossingroom.skills import TossingRoomSkills
from hitl_pmp.environments.tossingroom.tasks import TossingRoomTasks
from hitl_pmp.planning.grounding import SkillGrounder

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
    robot, item, room, pile = skill.parameters
    assert skill.preconditions == frozenset({
        LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(robot, room)),
        # The pile precondition is what stops the planner scheduling a pickup in
        # a room the dynamics refuse to pick up in -- see
        # TestPickupIsRestrictedToThePileRoom.
        LiftedAtom(predicate=PILE_IN_ROOM, variables=(pile, room)),
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
        skill=TossingRoomSkills.PICKUP,
        objects=(_ROBOT, _RECYCLING, _ENV.get_rooms()[3], _ENV.pile),
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


class TestPickupIsRestrictedToThePileRoom:
    """Pickup's lifted model must not permit what the dynamics deny.

    `TossingRoomEnvironment._apply_pickup` only acts when `robot_room == start_room`
    (there is a single limitless pile there), but Pickup's preconditions used to say
    only "robot is in some room + hand empty". Because pickup-early and pickup-late use
    the identical multiset of skills, the two orderings are *exactly cost-tied*, so Fast
    Downward broke the tie arbitrarily and routinely emitted plans that walk away from
    the pile and then pick up in the bin room -- a silent no-op. Measured on the default
    task distribution before the fix: 9 of 10 plans contained an unexecutable Pickup and
    only 1 of 10 tasks was solved (the one Press-only goal).

    This is the same defect class as the Ball-Ring `ignore_effects` bug: an
    over-permissive symbolic model producing plans that look valid and cannot execute.
    """

    @staticmethod
    def test_pickup_is_not_applicable_outside_the_pile_room() -> None:
        env = TossingRoomEnvironment()
        tasks = TossingRoomTasks(env=env, seed=0)
        provider = TossingRoomSkillProvider(env=env)
        state = tasks.sample_test_task().initial_state
        other_room = (env.start_room + 1) % env.num_rooms
        state.set(obj=env.robot, feature_name="room", feature_val=float(other_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)

        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        applicable = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        pickups = [g for g in applicable if g.skill.name == "Pickup"]
        assert not pickups, (
            f"Pickup is applicable in room {other_room}, but the pile is in "
            f"room {env.start_room}; executing it is a silent no-op."
        )

    @staticmethod
    def test_pickup_is_applicable_in_the_pile_room() -> None:
        """The complement, so the fix cannot be 'make Pickup never applicable'."""
        env = TossingRoomEnvironment()
        tasks = TossingRoomTasks(env=env, seed=0)
        provider = TossingRoomSkillProvider(env=env)
        state = tasks.sample_test_task().initial_state
        state.set(obj=env.robot, feature_name="room", feature_val=float(env.start_room))
        state.set(obj=env.robot, feature_name="holding", feature_val=0.0)

        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        applicable = SkillGrounder.applicable_ground_skills(
            skills=provider.skills(), objects=provider.objects(), true_atoms=atoms
        )
        assert [g for g in applicable if g.skill.name == "Pickup"]
