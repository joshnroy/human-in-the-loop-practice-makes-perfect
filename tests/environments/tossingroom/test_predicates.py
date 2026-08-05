from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    ADJACENT,
    BIN_EMPTY,
    BIN_IN_ROOM,
    BUTTON_FOR_BIN,
    BUTTON_IN_ROOM,
    HAND_EMPTY,
    HOLDING,
    ITEM_IN_BIN,
    ROBOT_IN_ROOM,
)

_ENV = TossingRoomEnvironment()
_ROBOT = TossingRoomEnvironment.robot
_RECYCLING = TossingRoomEnvironment.recycling
_TRASH = TossingRoomEnvironment.trash
_RECYCLING_BIN = TossingRoomEnvironment.recycling_bin
_TRASH_BIN = TossingRoomEnvironment.trash_bin
_TRASH_BUTTON = TossingRoomEnvironment.trash_button
_RECYCLING_BUTTON = TossingRoomEnvironment.recycling_button


def _state(*, recycling_count: int = 0, trash_count: int = 0):
    return _ENV.build_initial_state(
        trash_weight=1.0,
        recycling_weight=1.0,
        trash_bin_distance=2.0,
        recycling_bin_distance=2.0,
        recycling_count=recycling_count,
        trash_count=trash_count,
    )


def test_robot_in_room_holds_for_the_start_room() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    atom = ROBOT_IN_ROOM(state=state, objects=(_ROBOT, rooms[_ENV.start_room]))
    assert atom.predicate.holds(state, atom.objects) is True


def test_robot_in_room_does_not_hold_for_another_room() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    atom = ROBOT_IN_ROOM(state=state, objects=(_ROBOT, rooms[0]))
    assert atom.predicate.holds(state, atom.objects) is False


def test_hand_empty_holds_at_start_and_not_while_holding() -> None:
    state = _state()
    assert HAND_EMPTY(state=state, objects=(_ROBOT,)).predicate.holds(state, (_ROBOT,)) is True
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.TRASH_KIND)
    )
    assert HAND_EMPTY.holds(state, (_ROBOT,)) is False


def test_holding_matches_the_item_kind_being_held() -> None:
    state = _state()
    state.set(
        obj=_ROBOT, feature_name="holding", feature_val=float(TossingRoomEnvironment.RECYCLING_KIND)
    )
    assert HOLDING.holds(state, (_ROBOT, _RECYCLING)) is True
    assert HOLDING.holds(state, (_ROBOT, _TRASH)) is False


def test_adjacent_holds_only_for_consecutive_rooms() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    assert ADJACENT.holds(state, (rooms[2], rooms[3])) is True
    assert ADJACENT.holds(state, (rooms[2], rooms[4])) is False


def test_item_in_bin_requires_a_matching_non_empty_bin() -> None:
    state = _state(recycling_count=1)
    assert ITEM_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is True
    # Trash is not in the (recycling) bin even though that bin is non-empty: the kind
    # must match.
    assert ITEM_IN_BIN.holds(state, (_TRASH, _RECYCLING_BIN)) is False


def test_item_in_bin_is_false_for_an_empty_bin() -> None:
    state = _state(recycling_count=0)
    assert ITEM_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is False


def test_bin_empty_holds_only_when_count_is_zero() -> None:
    empty = _state(recycling_count=0)
    non_empty = _state(recycling_count=1)
    assert BIN_EMPTY.holds(empty, (_RECYCLING_BIN,)) is True
    assert BIN_EMPTY.holds(non_empty, (_RECYCLING_BIN,)) is False


def test_bin_in_room_and_button_in_room_are_static_placements() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    assert BIN_IN_ROOM.holds(state, (_RECYCLING_BIN, rooms[_ENV.recycling_bin_room])) is True
    assert BIN_IN_ROOM.holds(state, (_TRASH_BIN, rooms[_ENV.trash_bin_room])) is True
    # Each button sits beside its own bin, so its room IS that bin's room.
    assert BUTTON_IN_ROOM.holds(state, (_TRASH_BUTTON, rooms[_ENV.trash_bin_room])) is True
    assert BUTTON_IN_ROOM.holds(state, (_RECYCLING_BUTTON, rooms[_ENV.recycling_bin_room])) is True
    assert BUTTON_IN_ROOM.holds(state, (_TRASH_BUTTON, rooms[_ENV.recycling_bin_room])) is False


def test_button_for_bin_ties_each_button_to_exactly_one_bin() -> None:
    """`Press` empties only its own button's bin, so the operator needs a symbolic tie
    between the two -- without it `Press`'s delete effect on `ItemInBin` is not
    expressible per bin and has to fall back to a blanket `ignore_effects`."""
    state = _state()
    assert BUTTON_FOR_BIN.holds(state, (_TRASH_BUTTON, _TRASH_BIN)) is True
    assert BUTTON_FOR_BIN.holds(state, (_RECYCLING_BUTTON, _RECYCLING_BIN)) is True
    assert BUTTON_FOR_BIN.holds(state, (_TRASH_BUTTON, _RECYCLING_BIN)) is False
    assert BUTTON_FOR_BIN.holds(state, (_RECYCLING_BUTTON, _TRASH_BIN)) is False


def test_predicates_declare_their_types_and_names() -> None:
    assert ROBOT_IN_ROOM.name == "RobotInRoom"
    assert ITEM_IN_BIN.types == (TossingRoomEnvironment.item_type, TossingRoomEnvironment.bin_type)
    assert BIN_EMPTY.types == (TossingRoomEnvironment.bin_type,)
