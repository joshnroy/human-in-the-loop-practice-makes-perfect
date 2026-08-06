"""Tossing Room's oracle tests, plus the one thing the split adds: the goal must now
select which *lifted skill* the oracle uses, not just which objects it binds. The label
assertions below are what pin that -- `PickupTrash(`/`ThrowRecycling(` rather than
`Pickup(`/`Throw(`.

The throw-force assertions differ from the causal arm's, and the difference is the whole
point of this domain: there the oracle is genuinely privileged (it applies
`required_force`'s five relation coefficients, which no sampler can see); here
`required_force` is the identity, so "the oracle's answer" and "input index 4 of the
sampler's own row" are the same number. What is left to test is therefore not that the
oracle knows something secret, but that it reads the RIGHT item's feature -- which is
what the two throw families having independent target forces below is for.
"""

import pytest

from hitl_pmp.core.problem.tasks.types import Goal
from hitl_pmp.environments.tossingroomsplitidentity.environment import (
    TossingRoomSplitIdentityEnvironment,
)
from hitl_pmp.environments.tossingroomsplitidentity.predicates import (
    RECYCLING_BIN_EMPTY,
    RECYCLING_IN_BIN,
    TRASH_BIN_EMPTY,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroomsplitidentity.skill_oracle_policy import SkillOraclePolicy

_ENV = TossingRoomSplitIdentityEnvironment()
_ROBOT = TossingRoomSplitIdentityEnvironment.robot


def _recycling_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_IN_BIN(
                state=state,
                objects=(
                    TossingRoomSplitIdentityEnvironment.recycling,
                    TossingRoomSplitIdentityEnvironment.recycling_bin,
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
                    TossingRoomSplitIdentityEnvironment.trash,
                    TossingRoomSplitIdentityEnvironment.trash_bin,
                ),
            )
        })
    )


def _empty_goal(*, state) -> Goal:
    return Goal(
        atoms=frozenset({
            RECYCLING_BIN_EMPTY(
                state=state, objects=(TossingRoomSplitIdentityEnvironment.recycling_bin,)
            ),
            TRASH_BIN_EMPTY(state=state, objects=(TossingRoomSplitIdentityEnvironment.trash_bin,)),
        })
    )


def test_recycling_oracle_picks_up_with_the_recycling_specific_skill() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitIdentityEnvironment.RECYCLING_KIND
    assert labeled.label.startswith("PickupRecycling(")


def test_trash_oracle_picks_up_with_the_trash_specific_skill() -> None:
    """The complement, so "always chooses the recycling branch" cannot pass."""
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_PICKUP
    assert labeled.action[1] == TossingRoomSplitIdentityEnvironment.TRASH_KIND
    assert labeled.label.startswith("PickupTrash(")


def test_recycling_oracle_steps_left_toward_the_bin_room_while_holding() -> None:
    state = _ENV.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitIdentityEnvironment.RECYCLING_KIND),
    )
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_MOVE_ROOM
    # start_room 3 -> steps LEFT toward recycling room 1, i.e. to room 2.
    assert labeled.action[1] == _ENV.start_room - 1
    assert labeled.label.startswith("MoveRoom(")


def test_recycling_oracle_throws_with_the_exact_required_force_once_in_the_bin_room() -> None:
    """The oracle throws at exactly `recycling.target_force`, because under the identity
    representation `required_force` IS that feature.

    The causal arm's version of this test asserts a value equal to NEITHER observed cause,
    so that a passthrough bug could not produce it. Here a passthrough is the correct
    behaviour, so that guard is impossible and would be dishonest to keep. What remains
    testable -- and is what this test actually pins -- is that the oracle reads the
    RECYCLING item's feature: the two items are given deliberately different targets, so
    reading the wrong one yields 0.30 rather than 0.70."""
    state = _ENV.build_initial_state(trash_target_force=0.30, recycling_target_force=0.70)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitIdentityEnvironment.RECYCLING_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_recycling_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_THROW
    # Exactly the required force -> always within tolerance, on any task.
    assert labeled.action[2] == pytest.approx(0.70)
    assert labeled.label.startswith("ThrowRecycling(")
    assert "params=[0.7]" in labeled.label


def test_trash_oracle_throws_with_the_trash_specific_skill_and_its_own_required_force() -> None:
    """The complement in the same state: the two kinds' targets are drawn independently,
    so the two throws need genuinely different forces even here -- which is what the two
    split samplers each have to read out of their own row."""
    state = _ENV.build_initial_state(trash_target_force=0.30, recycling_target_force=0.70)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitIdentityEnvironment.TRASH_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_THROW
    assert labeled.action[2] == pytest.approx(0.30)
    assert labeled.label.startswith("ThrowTrash(")


def test_the_oracle_holds_no_privileged_information_a_sampler_lacks() -> None:
    """Stated positively, because it is this arm's defining property and the sharpest
    single difference from `environments/tossingroomsplit`.

    `required_force` there needs five relation constants that never enter a State; here it
    needs only `item.target_force`, which every throw sampler already observes at index 4
    of its own classifier row. So the force the oracle emits is reproducible from the
    state alone -- no environment coefficients required."""
    state = _ENV.build_initial_state(trash_target_force=0.62, recycling_target_force=0.24)
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomSplitIdentityEnvironment.TRASH_KIND),
    )
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
    labeled = SkillOraclePolicy.get_labeled_action(
        state=state, env=_ENV, goal=_trash_goal(state=state)
    )
    readable_from_state = float(
        state.get(obj=TossingRoomSplitIdentityEnvironment.trash, feature_name="target_force")
    )
    assert labeled.action[2] == pytest.approx(readable_from_state)


def _both_bins_full():
    return _ENV.build_initial_state(
        trash_target_force=0.5,
        recycling_target_force=0.5,
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
        assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_MOVE_ROOM
        # start_room 3 -> steps RIGHT toward the trash button in room 6.
        assert labeled.action[1] == _ENV.start_room + 1

    @staticmethod
    def test_it_presses_the_trash_button_on_arrival() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitIdentityEnvironment.TRASH_KIND
        assert labeled.label.startswith("PressTrash(")

    @staticmethod
    def test_only_then_does_it_cross_the_ledge_for_the_recycling_button() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.trash_bin_room))
        state.set(
            obj=TossingRoomSplitIdentityEnvironment.trash_bin,
            feature_name="count",
            feature_val=0.0,
        )
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_MOVE_ROOM
        assert labeled.action[1] == _ENV.trash_bin_room - 1

    @staticmethod
    def test_it_presses_the_recycling_button_beside_the_recycling_bin() -> None:
        state = _both_bins_full()
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(_ENV.recycling_bin_room))
        state.set(
            obj=TossingRoomSplitIdentityEnvironment.trash_bin,
            feature_name="count",
            feature_val=0.0,
        )
        labeled = SkillOraclePolicy.get_labeled_action(
            state=state, env=_ENV, goal=_empty_goal(state=state)
        )
        assert labeled.action[0] == TossingRoomSplitIdentityEnvironment.SKILL_PRESS
        assert labeled.action[1] == TossingRoomSplitIdentityEnvironment.RECYCLING_KIND
        assert labeled.label.startswith("PressRecycling(")
