"""Tossing Room's oracle tests, plus the one thing the split adds: the goal must now
select which *lifted skill* the oracle uses, not just which objects it binds. The label
assertions below are what pin that -- `PickupTrash(`/`ThrowRecycling(` rather than
`Pickup(`/`Throw(`."""

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplit.skill_oracle_policy import SkillOraclePolicy

_ENV = TossingRoomSplitEnvironment()
_ROBOT = TossingRoomSplitEnvironment.robot


def _recycling_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_IN_BIN(
                state=state,
                objects=(
                    TossingRoomSplitEnvironment.recycling,
                    TossingRoomSplitEnvironment.recycling_bin,
                ),
            )
        })
    )


def _trash_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            TRASH_IN_BIN(
                state=state,
                objects=(
                    TossingRoomSplitEnvironment.trash,
                    TossingRoomSplitEnvironment.trash_bin,
                ),
            )
        })
    )


def _empty_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_BIN_EMPTY(state=state, objects=(TossingRoomSplitEnvironment.recycling_bin,)),
            TRASH_BIN_EMPTY(state=state, objects=(TossingRoomSplitEnvironment.trash_bin,)),
        })
    )


def test_recycling_oracle_picks_up_with_the_recycling_specific_skill() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitEnvironment.RECYCLING_KIND
    assert labeled.label.startswith("PickupRecycling(")


def test_trash_oracle_picks_up_with_the_trash_specific_skill() -> None:
    """The complement, so "always chooses the recycling branch" cannot pass."""
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitEnvironment.TRASH_KIND
    assert labeled.label.startswith("PickupTrash(")


def test_recycling_oracle_steps_left_toward_the_bin_room_while_holding() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitEnvironment.RECYCLING_KIND),
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_MOVE_ROOM
    # start_room 3 -> steps LEFT toward recycling room 1, i.e. to room 2.
    assert labeled.action[1] == _ENV.start_room - 1
    assert labeled.label.startswith("MoveRoom(")


def test_recycling_oracle_throws_with_the_exact_target_force_once_in_the_bin_room() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.73)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitEnvironment.RECYCLING_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_THROW
    assert labeled.action[2] == 0.73  # exactly the known target -> always within tolerance
    assert labeled.label.startswith("ThrowRecycling(")
    assert "params=[0.73]" in labeled.label


def test_trash_oracle_throws_with_the_trash_specific_skill_and_its_own_target() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.61, recycling_target_force=0.73)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitEnvironment.TRASH_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_THROW
    assert labeled.action[2] == 0.61
    assert labeled.label.startswith("ThrowTrash(")


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

    The labels are what pin the split's own contribution here: `PressTrash(`/
    `PressRecycling(` rather than one shared `Press(`, because the bin and button types
    are split per kind."""

    @staticmethod
    def test_the_oracle_heads_for_the_trash_button_first() -> None:
        state = _both_bins_full()
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_MOVE_ROOM
        # start_room 3 -> steps RIGHT toward the trash button in room 6.
        assert labeled.action[1] == _ENV.start_room + 1

    @staticmethod
    def test_it_presses_the_trash_button_on_arrival() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitEnvironment.TRASH_KIND
        assert labeled.label.startswith("PressTrash(")

    @staticmethod
    def test_only_then_does_it_cross_the_ledge_for_the_recycling_button() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        state.set(obj=TossingRoomSplitEnvironment.trash_bin, feature_name="count", feature_val=0.0)
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_MOVE_ROOM
        assert labeled.action[1] == _ENV.trash_bin_room - 1

    @staticmethod
    def test_it_presses_the_recycling_button_beside_the_recycling_bin() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
        state.set(obj=TossingRoomSplitEnvironment.trash_bin, feature_name="count", feature_val=0.0)
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitEnvironment.RECYCLING_KIND
        assert labeled.label.startswith("PressRecycling(")
