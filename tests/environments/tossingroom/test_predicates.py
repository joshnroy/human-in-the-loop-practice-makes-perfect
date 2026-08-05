from hitl_pmp.environments.tossingroom.environment import TossingRoomEnvironment
from hitl_pmp.environments.tossingroom.predicates import (
    ADJACENT,
    BIN_EMPTY,
    BIN_IN_ROOM,
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
_BUTTON = TossingRoomEnvironment.button


def _state(*, recycling_count: int = 0, trash_count: int = 0):
    return _ENV.build_initial_state(
        trash_target_force=0.5,
        recycling_target_force=0.5,
        recycling_count=recycling_count,
        trash_count=trash_count,
    )


def _landed(*, state, item):
    """The state as it would be after a throw of `item` LANDED. Only the item's own
    in_bin flag distinguishes that from a bin that merely holds something."""
    state.set(obj=item, feature_name="in_bin", feature_val=1.0)
    return state


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


def test_item_in_bin_requires_a_matching_bin() -> None:
    state = _landed(state=_state(recycling_count=1), item=_RECYCLING)
    assert ITEM_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is True
    # Trash is not in the (recycling) bin even though that bin holds something: the
    # kind must match.
    assert ITEM_IN_BIN.holds(state, (_TRASH, _RECYCLING_BIN)) is False


def test_item_in_bin_is_false_for_an_empty_bin() -> None:
    state = _state(recycling_count=0)
    assert ITEM_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is False


def test_item_in_bin_reads_the_item_flag_not_the_bin_count() -> None:
    """The scoring defect this predicate was fixed for: ItemInBin is Throw's add
    effect, and EES scores an attempt by `add_effects <= atoms(next_state)`. A count
    reading made every throw into an already-non-empty bin a success at any force.
    Both states below have a genuinely NON-EMPTY bin -- only the flag differs."""
    landed = _landed(state=_state(recycling_count=3), item=_RECYCLING)
    prefilled_only = _state(recycling_count=3)
    # Non-vacuity: the two cases straddle the flag, not the count.
    assert prefilled_only.get(obj=_RECYCLING_BIN, feature_name="count") >= 1.0
    assert landed.get(obj=_RECYCLING_BIN, feature_name="count") >= 1.0
    assert ITEM_IN_BIN.holds(landed, (_RECYCLING, _RECYCLING_BIN)) is True
    assert ITEM_IN_BIN.holds(prefilled_only, (_RECYCLING, _RECYCLING_BIN)) is False


def test_bin_empty_holds_only_when_count_is_zero() -> None:
    """BinEmpty deliberately stays count-based -- it is about the bin, not about the
    item in play -- so the EMPTY goal is untouched by ItemInBin's item-flag reading."""
    empty = _state(recycling_count=0)
    non_empty = _state(recycling_count=2)
    assert BIN_EMPTY.holds(empty, (_RECYCLING_BIN,)) is True
    assert BIN_EMPTY.holds(non_empty, (_RECYCLING_BIN,)) is False
    # The item flag does not enter into it, in either direction.
    assert BIN_EMPTY.holds(_landed(state=empty, item=_RECYCLING), (_RECYCLING_BIN,)) is True
    assert BIN_EMPTY.holds(_landed(state=non_empty, item=_RECYCLING), (_RECYCLING_BIN,)) is False


def test_bin_in_room_and_button_in_room_are_static_placements() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    assert BIN_IN_ROOM.holds(state, (_RECYCLING_BIN, rooms[_ENV.recycling_bin_room])) is True
    assert BIN_IN_ROOM.holds(state, (_TRASH_BIN, rooms[_ENV.trash_bin_room])) is True
    assert BUTTON_IN_ROOM.holds(state, (_BUTTON, rooms[_ENV.button_room])) is True
    assert BUTTON_IN_ROOM.holds(state, (_BUTTON, rooms[0])) is False


def test_predicates_declare_their_types_and_names() -> None:
    assert ROBOT_IN_ROOM.name == "RobotInRoom"
    assert ITEM_IN_BIN.types == (TossingRoomEnvironment.item_type, TossingRoomEnvironment.bin_type)
    assert BIN_EMPTY.types == (TossingRoomEnvironment.bin_type,)
