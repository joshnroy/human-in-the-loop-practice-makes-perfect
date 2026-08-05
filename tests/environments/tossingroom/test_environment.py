import numpy as np
import pytest
from gymnasium.spaces import Box

from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment

_ROBOT = TossingRoomEnvironment.robot
_RECYCLING_BIN = TossingRoomEnvironment.recycling_bin
_TRASH_BIN = TossingRoomEnvironment.trash_bin


def _env() -> TossingRoomEnvironment:
    return TossingRoomEnvironment()


def _fresh_state(*, env: TossingRoomEnvironment):
    state = env.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    env.set_state(state=state)
    return state


def _pickup(*, kind: int) -> np.ndarray:
    return np.array([float(TossingRoomEnvironment.SKILL_PICKUP), float(kind), 0.0])


def _move(*, to_room: int) -> np.ndarray:
    return np.array([float(TossingRoomEnvironment.SKILL_MOVE_ROOM), float(to_room), 0.0])


def _throw(*, kind: int, force: float) -> np.ndarray:
    return np.array([float(TossingRoomEnvironment.SKILL_THROW), float(kind), force])


def _press(*, kind: int) -> np.ndarray:
    """`kind` names WHICH button is pressed -- each bin has its own, beside it. Press's
    `arg` used to be unused, back when one button emptied both bins."""
    return np.array([float(TossingRoomEnvironment.SKILL_PRESS), float(kind), 0.0])


def test_hard_reset_sets_canonical_starting_state() -> None:
    env = _env()
    env.hard_reset()
    state = env.get_current_state()
    assert state.get(obj=_ROBOT, feature_name="room") == env.start_room
    assert state.get(obj=_ROBOT, feature_name="holding") == 0.0
    assert state.get(obj=_RECYCLING_BIN, feature_name="count") == 0.0
    assert state.get(obj=_TRASH_BIN, feature_name="count") == 0.0


def test_build_initial_state_places_bins_and_buttons_in_their_rooms() -> None:
    env = _env()
    state = env.build_initial_state(trash_target_force=0.7, recycling_target_force=0.3)
    assert state.get(obj=_RECYCLING_BIN, feature_name="room") == env.recycling_bin_room
    assert state.get(obj=_TRASH_BIN, feature_name="room") == env.trash_bin_room
    assert state.get(obj=TossingRoomEnvironment.trash, feature_name="target_force") == 0.7
    assert state.get(obj=TossingRoomEnvironment.recycling, feature_name="target_force") == 0.3


def test_build_initial_state_lays_out_room_indices() -> None:
    env = TossingRoomEnvironment(num_rooms=5)
    state = env.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    rooms = env.get_rooms()
    assert len(rooms) == 5
    for i, room in enumerate(rooms):
        assert state.get(obj=room, feature_name="index") == i


def test_pickup_at_start_room_fills_the_hand() -> None:
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(action=_pickup(kind=TossingRoomEnvironment.RECYCLING_KIND))
    assert (
        next_state.get(obj=_ROBOT, feature_name="holding") == TossingRoomEnvironment.RECYCLING_KIND
    )


def test_pickup_is_a_no_op_when_not_in_the_start_room() -> None:
    env = _env()
    state = _fresh_state(env=env)
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.start_room - 1))
    env.set_state(state=state)
    next_state = env.take_action(action=_pickup(kind=TossingRoomEnvironment.TRASH_KIND))
    assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_pickup_is_a_no_op_when_the_hand_is_already_full() -> None:
    env = _env()
    state = _fresh_state(env=env)
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.TRASH_KIND)
    )
    env.set_state(state=state)
    next_state = env.take_action(action=_pickup(kind=TossingRoomEnvironment.RECYCLING_KIND))
    # Still holding the original item, not overwritten.
    assert next_state.get(obj=_ROBOT, feature_name="holding") == TossingRoomEnvironment.TRASH_KIND


def test_move_to_an_adjacent_room_succeeds() -> None:
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(action=_move(to_room=env.start_room + 1))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.start_room + 1


def test_move_to_a_non_adjacent_room_is_a_no_op() -> None:
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(action=_move(to_room=env.start_room + 2))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.start_room


def test_ledge_blocks_stepping_right_across_it() -> None:
    """Moving RIGHT from blocked_right_from to the next room is the one irreversible-
    blocked step -- a no-op."""
    env = _env()
    state = _fresh_state(env=env)
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.blocked_right_from))
    env.set_state(state=state)
    next_state = env.take_action(action=_move(to_room=env.blocked_right_from + 1))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.blocked_right_from


def test_ledge_allows_stepping_left_across_it() -> None:
    env = _env()
    state = _fresh_state(env=env)
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.blocked_right_from + 1))
    env.set_state(state=state)
    next_state = env.take_action(action=_move(to_room=env.blocked_right_from))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.blocked_right_from


def _carry_to_recycling_room(*, env: TossingRoomEnvironment):
    state = env.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.recycling_bin_room))
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.RECYCLING_KIND)
    )
    env.set_state(state=state)
    return state


def test_throw_within_tolerance_lands_in_the_bin_and_empties_the_hand() -> None:
    env = _env()
    _carry_to_recycling_room(env=env)
    next_state = env.take_action(
        action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5)
    )
    assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0
    assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_throw_outside_tolerance_consumes_the_item_without_binning_it() -> None:
    """A missed throw releases the item: nothing lands in the bin, but the hand
    empties. Previously a miss changed nothing at all, which made it a *free retry* --
    the robot still held the item and still stood in the bin room, so the very next
    step re-threw at zero cost. That turned the evaluation horizon into a silent
    "number of attempts" dial: an unpracticed EES scored 94.7% purely by re-rolling
    (a pre-release, pre-fixed-composition figure, quoted as the history that motivated
    this rather than as a number today's code reproduces).

    The thrown item is gone rather than recoverable. Items are singleton
    discriminators with features (kind, target_force) and no position, so "it is lying
    on the floor near the bin" is not representable -- and making it so would
    reintroduce the cheap retry this fixes. The only way to try again is a fresh item
    from the limitless pile, which costs a round trip to the start room. That is what
    makes an evaluation miss terminal while practice, with a 100-step budget, can
    still afford to retry."""
    env = _env()
    _carry_to_recycling_room(env=env)
    # target is 0.5, tolerance 0.1 -> a force of 0.9 misses.
    next_state = env.take_action(
        action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.9)
    )
    assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 0.0
    assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_a_missed_throw_can_be_retried_only_by_fetching_a_fresh_item() -> None:
    """The complement to the above: after a miss the robot cannot re-throw in place
    (its hand is empty) and cannot pick up where it stands (the pile is in the start
    room), so a retry costs the walk back."""
    env = _env()
    _carry_to_recycling_room(env=env)
    env.take_action(action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.9))
    # Picking up in the bin room does nothing: the pile is in the start room.
    after_pickup = env.take_action(
        action=np.array([
            float(TossingRoomEnvironment.SKILL_PICKUP),
            float(TossingRoomEnvironment.RECYCLING_KIND),
            0.0,
        ])
    )
    assert after_pickup.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_throw_in_the_wrong_room_still_releases_the_item() -> None:
    """Throwing is a release wherever you do it -- the item does not land in a bin you
    are not standing next to, but it does leave your hand. Only an empty-handed throw
    is a true no-op, since there is nothing to release."""
    env = _env()
    state = env.build_initial_state(trash_target_force=0.5, recycling_target_force=0.5)
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.RECYCLING_KIND)
    )
    # Robot is at start_room, not the recycling bin room.
    env.set_state(state=state)
    next_state = env.take_action(
        action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5)
    )
    assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 0.0
    assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_throw_with_an_empty_hand_is_a_no_op() -> None:
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(
        action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5)
    )
    assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 0.0


def test_press_outside_a_button_room_is_a_no_op() -> None:
    env = _env()
    state = env.build_initial_state(
        trash_target_force=0.5, recycling_target_force=0.5, recycling_count=1, trash_count=1
    )
    # Robot at start_room, which holds neither bin's button.
    env.set_state(state=state)
    next_state = env.take_action(action=_press(kind=TossingRoomEnvironment.TRASH_KIND))
    assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0
    assert next_state.get(obj=_TRASH_BIN, feature_name="count") == 1.0


class TestBinsHoldAtMostOneItem:
    """Capacity 1 is the whole point of the bin redesign: with the count provably 0 at
    throw time, `ItemInBin` goes false -> true exactly once per throw, so EES's
    `add_effects <= atoms(next_state)` verdict cannot be satisfied by an item somebody
    else's throw put there. The dynamics REFUSE a throw at a full bin (a silent no-op,
    like every other out-of-context action here) rather than swallowing the item, and
    `Throw` carries the matching `BinEmpty` precondition so the symbolic model stays
    exactly as strong as this guard."""

    @staticmethod
    def _at_the_full_recycling_bin(*, env: TossingRoomEnvironment):
        state = env.build_initial_state(
            trash_target_force=0.5, recycling_target_force=0.5, recycling_count=1
        )
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.recycling_bin_room))
        state.set(
            obj=_ROBOT,
            feature_name="holding",
            feature_val=float(TossingRoomEnvironment.RECYCLING_KIND),
        )
        env.set_state(state=state)
        return state

    @staticmethod
    def test_a_throw_at_a_full_bin_is_refused_and_the_item_stays_in_hand() -> None:
        env = _env()
        TestBinsHoldAtMostOneItem._at_the_full_recycling_bin(env=env)
        # Force is exactly the target, so only the capacity guard can refuse this.
        next_state = env.take_action(
            action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5)
        )
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0
        assert (
            next_state.get(obj=_ROBOT, feature_name="holding")
            == TossingRoomEnvironment.RECYCLING_KIND
        )

    @staticmethod
    def test_the_same_throw_lands_once_the_bin_has_been_emptied() -> None:
        """The complement, so the guard cannot be 'a throw never lands'."""
        env = _env()
        TestBinsHoldAtMostOneItem._at_the_full_recycling_bin(env=env)
        env.take_action(action=_press(kind=TossingRoomEnvironment.RECYCLING_KIND))
        next_state = env.take_action(
            action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5)
        )
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0
        assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0

    @staticmethod
    def test_a_landed_throw_never_pushes_a_bin_past_one() -> None:
        env = _env()
        state = _carry_to_recycling_room(env=env)
        del state
        env.take_action(action=_throw(kind=TossingRoomEnvironment.RECYCLING_KIND, force=0.5))
        assert env.get_current_state().get(obj=_RECYCLING_BIN, feature_name="count") == 1.0

    @staticmethod
    def test_build_initial_state_rejects_a_count_beyond_capacity() -> None:
        with pytest.raises(ValueError, match="at most one item"):
            _env().build_initial_state(
                trash_target_force=0.5, recycling_target_force=0.5, trash_count=2
            )


class TestEachBinHasItsOwnButtonBesideIt:
    """One button per bin, in that bin's own room, emptying only that bin. The single
    shared button that emptied both is what forced `Press`'s `ignore_effects` (a
    universal delete no per-item effect could express); with per-bin effects the delete
    is expressible as an ordinary `delete_effect`."""

    @staticmethod
    def _both_bins_full(*, env: TossingRoomEnvironment):
        state = env.build_initial_state(
            trash_target_force=0.5,
            recycling_target_force=0.5,
            recycling_count=1,
            trash_count=1,
        )
        env.set_state(state=state)
        return state

    @staticmethod
    def test_each_button_sits_in_its_own_bins_room() -> None:
        env = _env()
        state = TestEachBinHasItsOwnButtonBesideIt._both_bins_full(env=env)
        assert (
            state.get(obj=TossingRoomEnvironment.trash_button, feature_name="room")
            == env.trash_bin_room
        )
        assert (
            state.get(obj=TossingRoomEnvironment.recycling_button, feature_name="room")
            == env.recycling_bin_room
        )

    @staticmethod
    def test_pressing_the_trash_button_empties_only_the_trash_bin() -> None:
        env = _env()
        state = TestEachBinHasItsOwnButtonBesideIt._both_bins_full(env=env)
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.trash_bin_room))
        env.set_state(state=state)
        next_state = env.take_action(action=_press(kind=TossingRoomEnvironment.TRASH_KIND))
        assert next_state.get(obj=_TRASH_BIN, feature_name="count") == 0.0
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0

    @staticmethod
    def test_pressing_the_recycling_button_empties_only_the_recycling_bin() -> None:
        env = _env()
        state = TestEachBinHasItsOwnButtonBesideIt._both_bins_full(env=env)
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.recycling_bin_room))
        env.set_state(state=state)
        next_state = env.take_action(action=_press(kind=TossingRoomEnvironment.RECYCLING_KIND))
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 0.0
        assert next_state.get(obj=_TRASH_BIN, feature_name="count") == 1.0

    @staticmethod
    def test_pressing_a_button_from_the_other_buttons_room_is_a_no_op() -> None:
        """The buttons are not remote controls: standing beside the trash button and
        asking for the recycling one does nothing."""
        env = _env()
        state = TestEachBinHasItsOwnButtonBesideIt._both_bins_full(env=env)
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.trash_bin_room))
        env.set_state(state=state)
        next_state = env.take_action(action=_press(kind=TossingRoomEnvironment.RECYCLING_KIND))
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0
        assert next_state.get(obj=_TRASH_BIN, feature_name="count") == 1.0

    @staticmethod
    def test_pressing_an_unknown_button_kind_is_a_no_op() -> None:
        env = _env()
        state = TestEachBinHasItsOwnButtonBesideIt._both_bins_full(env=env)
        state.set(obj=_ROBOT, feature_name="room", feature_val=float(env.trash_bin_room))
        env.set_state(state=state)
        next_state = env.take_action(action=_press(kind=99))
        assert next_state.get(obj=_TRASH_BIN, feature_name="count") == 1.0
        assert next_state.get(obj=_RECYCLING_BIN, feature_name="count") == 1.0


def test_take_action_updates_current_state() -> None:
    env = _env()
    _fresh_state(env=env)
    env.take_action(action=_move(to_room=env.start_room + 1))
    assert env.get_current_state().get(obj=_ROBOT, feature_name="room") == env.start_room + 1


def test_take_action_is_total_for_a_non_finite_action() -> None:
    """The Box action space contains +-inf, and round(inf) raises OverflowError -- the
    env must treat it as a silent no-op, never crash."""
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(action=np.array([np.inf, 0.0, 0.0]))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.start_room


def test_take_action_is_total_for_an_unknown_skill_id() -> None:
    env = _env()
    _fresh_state(env=env)
    next_state = env.take_action(action=np.array([99.0, 0.0, 0.0]))
    assert next_state.get(obj=_ROBOT, feature_name="room") == env.start_room
    assert next_state.get(obj=_ROBOT, feature_name="holding") == 0.0


def test_get_valid_actions_is_empty_for_the_continuous_space() -> None:
    assert _env().get_valid_actions() == []


def test_action_space_is_three_dimensional_unbounded_box() -> None:
    assert isinstance(TossingRoomEnvironment.action_space, Box)
    assert TossingRoomEnvironment.action_space.shape == (3,)
    assert np.all(np.isinf(TossingRoomEnvironment.action_space.low))
    assert np.all(np.isinf(TossingRoomEnvironment.action_space.high))
