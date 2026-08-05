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


def _both_bins_full():
    return _ENV.build_initial_state(
        trash_target_force=0.5, recycling_target_force=0.5, recycling_count=1, trash_count=1
    )


class TestEmptyIsAnOrderingTask:
    """Each bin now has its own button beside it, so EMPTY needs both pressed -- and the
    one-way ledge makes the order load-bearing. The trash button (room 6) is right of
    start; the recycling one (room 1) is behind the ledge, reachable only by dropping
    LEFT across it, after which nothing to the right can be reached again. So the only
    solution is trash first, then recycling; the reverse order strands the robot.

    That is deliberate: it makes EMPTY exercise the irreversibility this domain is
    about, instead of being a deterministic walk-and-press smoke test."""

    @staticmethod
    def test_the_oracle_heads_for_the_trash_button_first() -> None:
        state = _both_bins_full()
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomEnvironment.SKILL_MOVE_ROOM
        # start_room 3 -> steps RIGHT toward the trash button in room 6.
        assert labeled.action[1] == _ENV.start_room + 1

    @staticmethod
    def test_it_presses_the_trash_button_on_arrival() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomEnvironment.TRASH_KIND
        assert labeled.label.startswith("Press(")

    @staticmethod
    def test_only_then_does_it_cross_the_ledge_for_the_recycling_button() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        state.set(obj=TossingRoomEnvironment.trash_bin, feature_name="count", feature_val=0.0)
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomEnvironment.SKILL_MOVE_ROOM
        assert labeled.action[1] == _ENV.trash_bin_room - 1

    @staticmethod
    def test_it_presses_the_recycling_button_beside_the_recycling_bin() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
        state.set(obj=TossingRoomEnvironment.trash_bin, feature_name="count", feature_val=0.0)
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomEnvironment.RECYCLING_KIND
