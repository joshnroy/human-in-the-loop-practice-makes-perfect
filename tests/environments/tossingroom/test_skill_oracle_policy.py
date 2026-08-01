from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import BIN_EMPTY, ITEM_IN_BIN
from hitl_pmp.environments.tossingroom.skill_oracle_policy import SkillOraclePolicy

_ENV = TossingRoomEnvironment()
_ROBOT = TossingRoomEnvironment.robot


def _recycling_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            ITEM_IN_BIN(
                state=state,
                objects=(TossingRoomEnvironment.recycling, TossingRoomEnvironment.recycling_bin),
            )
        })
    )


def _empty_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            BIN_EMPTY(state=state, objects=(TossingRoomEnvironment.recycling_bin,)),
            BIN_EMPTY(state=state, objects=(TossingRoomEnvironment.trash_bin,)),
        })
    )


def test_recycling_oracle_picks_up_first_when_hand_is_empty() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomEnvironment.RECYCLING_KIND
    assert labeled.label.startswith("Pickup(")


def test_recycling_oracle_steps_left_toward_the_bin_room_while_holding() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.RECYCLING_KIND)
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomEnvironment.SKILL_MOVE_ROOM
    # start_room 3 -> steps LEFT toward recycling room 1, i.e. to room 2.
    assert labeled.action[1] == _ENV.start_room - 1
    assert labeled.label.startswith("MoveRoom(")


def test_recycling_oracle_throws_with_the_exact_target_force_once_in_the_bin_room() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.73)
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.RECYCLING_KIND)
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomEnvironment.SKILL_THROW
    assert labeled.action[2] == 0.73  # exactly the known target -> always within tolerance
    assert labeled.label.startswith("Throw(")
    assert "params=[0.73]" in labeled.label


def test_empty_oracle_walks_to_the_button_room_then_presses() -> None:
    state = _ENV.build_initial_state(
        trash_target_force=0.5, recycling_target_force=0.5, recycling_count=2, trash_count=2
    )
    # Not yet in the button room -> steps right toward it.
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_empty_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomEnvironment.SKILL_MOVE_ROOM
    assert labeled.action[1] == _ENV.start_room + 1

    # Already in the button room -> presses.
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.button_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_empty_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomEnvironment.SKILL_PRESS
    assert labeled.label.startswith("Press(")
