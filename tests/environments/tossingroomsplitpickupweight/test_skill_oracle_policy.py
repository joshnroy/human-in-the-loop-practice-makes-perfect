"""Tossing Room's oracle tests, plus the one thing the split adds: the goal must now
select which *lifted skill* the oracle uses, not just which objects it binds. The label
assertions below are what pin that -- `PickupTrash(`/`ThrowRecycling(` rather than
`Pickup(`/`Throw(`."""

import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossingroomsplitpickupweight.environment import (
    TossingRoomSplitPickupWeightEnvironment,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplitpickupweight.skill_oracle_policy import SkillOraclePolicy

_ENV = TossingRoomSplitPickupWeightEnvironment()
_ROBOT = TossingRoomSplitPickupWeightEnvironment.robot


def _recycling_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_IN_BIN(
                state=state,
                objects=(
                    TossingRoomSplitPickupWeightEnvironment.recycling,
                    TossingRoomSplitPickupWeightEnvironment.recycling_bin,
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
                    TossingRoomSplitPickupWeightEnvironment.trash,
                    TossingRoomSplitPickupWeightEnvironment.trash_bin,
                ),
            )
        })
    )


def _empty_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_BIN_EMPTY(
                state=state, objects=(TossingRoomSplitPickupWeightEnvironment.recycling_bin,)
            ),
            TRASH_BIN_EMPTY(
                state=state, objects=(TossingRoomSplitPickupWeightEnvironment.trash_bin,)
            ),
        })
    )


def test_recycling_oracle_picks_up_with_the_recycling_specific_skill() -> None:
    state = _ENV.build_initial_state(weight_seed=0)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitPickupWeightEnvironment.RECYCLING_KIND
    assert labeled.label.startswith("PickupRecycling(")


def test_trash_oracle_picks_up_with_the_trash_specific_skill() -> None:
    """The complement, so "always chooses the recycling branch" cannot pass."""
    state = _ENV.build_initial_state(weight_seed=0)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitPickupWeightEnvironment.TRASH_KIND
    assert labeled.label.startswith("PickupTrash(")


def test_recycling_oracle_steps_left_toward_the_bin_room_while_holding() -> None:
    state = _ENV.build_initial_state(weight_seed=0)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitPickupWeightEnvironment.RECYCLING_KIND),
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_MOVE_ROOM
    # start_room 3 -> steps LEFT toward recycling room 1, i.e. to room 2.
    assert labeled.action[1] == _ENV.start_room - 1
    assert labeled.label.startswith("MoveRoom(")


def test_recycling_oracle_throws_with_the_exact_required_force_once_in_the_bin_room() -> None:
    """The oracle no longer copies a `target_force` feature out of the state -- there is
    none. It reads the two CAUSES (the bin's throw_distance, the item's weight) and
    applies `TossingRoomSplitPickupWeightEnvironment.required_force`, whose coefficients are
    privileged knowledge neither throw sampler has. The distance is fixed at the
    reference, so weight 1.25 makes the required force 0.5 + 0.8 * 0.25 = 0.70 -- a value
    equal to neither the weight nor anything else in the row, so a passthrough bug could
    not produce it. The weight is written onto the item the way a real pickup writes it,
    rather than passed to build_initial_state, which no longer takes one."""
    state = _ENV.build_initial_state(weight_seed=0)
    state.set(
        obj=TossingRoomSplitPickupWeightEnvironment.recycling,
        feature_name="weight",
        feature_val=1.25,
    )
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitPickupWeightEnvironment.RECYCLING_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_THROW
    # Exactly the required force -> always within tolerance, on any task.
    assert labeled.action[2] == pytest.approx(0.70)
    assert labeled.label.startswith("ThrowRecycling(")
    assert "params=[0.7]" in labeled.label


def test_trash_oracle_throws_with_the_trash_specific_skill_and_its_own_required_force() -> None:
    """Each kind's weight is drawn by its own pickup, so the two throws need genuinely
    different forces in the same state -- which is what the split samplers have to learn
    separately. Trash: weight 0.75 -> 0.5 + 0.8 * (-0.25) = 0.30."""
    state = _ENV.build_initial_state(weight_seed=0)
    state.set(
        obj=TossingRoomSplitPickupWeightEnvironment.trash, feature_name="weight", feature_val=0.75
    )
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitPickupWeightEnvironment.TRASH_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_THROW
    assert labeled.action[2] == pytest.approx(0.30)
    assert labeled.label.startswith("ThrowTrash(")


def _both_bins_full():
    return _ENV.build_initial_state(
        weight_seed=0,
        recycling_count=1,
        trash_count=1,
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
        assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_MOVE_ROOM
        # start_room 3 -> steps RIGHT toward the trash button in room 6.
        assert labeled.action[1] == _ENV.start_room + 1

    @staticmethod
    def test_it_presses_the_trash_button_on_arrival() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitPickupWeightEnvironment.TRASH_KIND
        assert labeled.label.startswith("PressTrash(")

    @staticmethod
    def test_only_then_does_it_cross_the_ledge_for_the_recycling_button() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        state.set(
            obj=TossingRoomSplitPickupWeightEnvironment.trash_bin,
            feature_name="count",
            feature_val=0.0,
        )
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_MOVE_ROOM
        assert labeled.action[1] == _ENV.trash_bin_room - 1

    @staticmethod
    def test_it_presses_the_recycling_button_beside_the_recycling_bin() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
        state.set(
            obj=TossingRoomSplitPickupWeightEnvironment.trash_bin,
            feature_name="count",
            feature_val=0.0,
        )
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitPickupWeightEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitPickupWeightEnvironment.RECYCLING_KIND
        assert labeled.label.startswith("PressRecycling(")
