"""Tossing Room's oracle tests, plus the one thing the split adds: the goal must now
select which *lifted skill* the oracle uses, not just which objects it binds. The label
assertions below are what pin that -- `PickupTrash(`/`ThrowRecycling(` rather than
`Pickup(`/`Throw(`."""

import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossingroomsplit.environment import TossingRoomSplitEnvironment
from hitl_pmp.environments.tossingroomsplit.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplit.problem import TossingRoomSplitProblem
from hitl_pmp.environments.tossingroomsplit.skill_oracle_policy import SkillOraclePolicy
from hitl_pmp.environments.tossingroomsplit.tasks import TossingRoomSplitTasks

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
    state = _ENV.build_initial_state(
        trash_weight=1.0, recycling_weight=1.0, trash_bin_distance=2.0, recycling_bin_distance=2.0
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitEnvironment.RECYCLING_KIND
    assert labeled.label.startswith("PickupRecycling(")


def test_trash_oracle_picks_up_with_the_trash_specific_skill() -> None:
    """The complement, so "always chooses the recycling branch" cannot pass."""
    state = _ENV.build_initial_state(
        trash_weight=1.0, recycling_weight=1.0, trash_bin_distance=2.0, recycling_bin_distance=2.0
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitEnvironment.TRASH_KIND
    assert labeled.label.startswith("PickupTrash(")


def test_recycling_oracle_steps_left_toward_the_bin_room_while_holding() -> None:
    state = _ENV.build_initial_state(
        trash_weight=1.0, recycling_weight=1.0, trash_bin_distance=2.0, recycling_bin_distance=2.0
    )
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


def test_recycling_oracle_throws_with_the_exact_required_force_once_in_the_bin_room() -> None:
    """The oracle no longer copies a `target_force` feature out of the state -- there is
    none. It reads the two CAUSES (the bin's throw_distance, the item's weight) and
    applies `TossingRoomSplitEnvironment.required_force`, whose coefficients are
    privileged knowledge neither throw sampler has. Distance 2.5 and weight 1.25 make the
    required force 0.5 + 0.2 * 0.5 + 0.4 * 0.25 = 0.70, a value equal to neither cause,
    so a passthrough bug could not produce it."""
    state = _ENV.build_initial_state(
        trash_weight=0.5,
        recycling_weight=1.25,
        trash_bin_distance=2.0,
        recycling_bin_distance=2.5,
    )
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
    # Exactly the required force -> always within tolerance, on any task.
    assert labeled.action[2] == pytest.approx(0.70)
    assert labeled.label.startswith("ThrowRecycling(")
    assert "params=[0.7]" in labeled.label


def test_trash_oracle_throws_with_the_trash_specific_skill_and_its_own_required_force() -> None:
    """The two kinds' causes are drawn independently, so the two throws need genuinely
    different forces in the same state -- which is what the split samplers have to learn
    separately. Trash: distance 1.5, weight 0.75 -> 0.5 - 0.1 - 0.1 = 0.30."""
    state = _ENV.build_initial_state(
        trash_weight=0.75,
        recycling_weight=1.25,
        trash_bin_distance=1.5,
        recycling_bin_distance=2.5,
    )
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
    assert labeled.action[2] == pytest.approx(0.30)
    assert labeled.label.startswith("ThrowTrash(")


def _both_bins_full():
    return _ENV.build_initial_state(
        trash_weight=1.0,
        recycling_weight=1.0,
        trash_bin_distance=2.0,
        recycling_bin_distance=2.0,
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


class TestTheOracleUnderATwoWayLedge:
    """`--two-way-ledge` must not silently break the privileged solver. `_empty_step`
    visits still-full bins in DESCENDING room index because the one-way ledge makes that
    the only feasible order; two-way, both orders work and descending is no longer the
    cheapest. It is kept anyway -- it is still *correct* in both worlds, and changing it
    would make the oracle's behaviour depend on the flag for no measured benefit.

    What that costs is bounded and asserted below: descending exceeds the cheapest order
    by |start - far_bin| - |start - near_bin|, which is 1 on the default layout against
    the 2 spare steps `max_episode_steps` allows."""

    @staticmethod
    def _rollout(*, env: TossingRoomSplitEnvironment, goal_builder) -> int:
        state = env.build_initial_state(
            trash_weight=1.0,
            recycling_weight=1.0,
            trash_bin_distance=2.0,
            recycling_bin_distance=2.0,
            trash_count=1,
            recycling_count=1,
        )
        env.set_state(state=state)
        goal = goal_builder(state=state)
        for step in range(64):
            if goal.is_satisfied(state=env.get_current_state()):
                return step
            labeled = SkillOraclePolicy.get_labeled_action(
                state=env.get_current_state(), env=env, goal=goal
            )
            env.take_action(action=labeled.action)
        raise AssertionError("oracle did not reach the goal")

    def test_it_still_empties_both_bins_two_way(self) -> None:
        env = TossingRoomSplitEnvironment(two_way_ledge=True)
        assert TestTheOracleUnderATwoWayLedge._rollout(env=env, goal_builder=_empty_goal) == 10

    def test_it_empties_both_bins_one_way_in_the_same_number_of_steps(self) -> None:
        """Unchanged behaviour with the flag off: the flag adds a world, it does not
        move the existing one."""
        env = TossingRoomSplitEnvironment()
        assert TestTheOracleUnderATwoWayLedge._rollout(env=env, goal_builder=_empty_goal) == 10

    def test_its_empty_solve_still_fits_the_two_way_horizon(self) -> None:
        """10 descending steps against a horizon of 11 -- one spare, not zero."""
        env = TossingRoomSplitEnvironment(two_way_ledge=True)
        problem = TossingRoomSplitProblem(
            env=env, tasks=TossingRoomSplitTasks(env=env, forced_goal_type=None)
        )
        steps = TestTheOracleUnderATwoWayLedge._rollout(env=env, goal_builder=_empty_goal)
        assert steps <= problem.max_episode_steps()

    def test_it_still_solves_both_throw_families_two_way(self) -> None:
        for goal_builder, expected in ((_recycling_goal, 4), (_trash_goal, 5)):
            env = TossingRoomSplitEnvironment(two_way_ledge=True)
            state = env.build_initial_state(
                trash_weight=1.0,
                recycling_weight=1.0,
                trash_bin_distance=2.0,
                recycling_bin_distance=2.0,
            )
            env.set_state(state=state)
            goal = goal_builder(state=state)
            for step in range(64):
                if goal.is_satisfied(state=env.get_current_state()):
                    assert step == expected
                    break
                labeled = SkillOraclePolicy.get_labeled_action(
                    state=env.get_current_state(), env=env, goal=goal
                )
                env.take_action(action=labeled.action)
            else:
                raise AssertionError("oracle did not reach the goal")
