"""Tossing Room's predicate tests, retyped for the per-kind split.

The classifier *logic* is unchanged, so most of this is Tossing Room's file verbatim.
What is new is the last class: the split predicates must not merely behave like their
shared ancestors, they must be *unable* to be applied across kinds -- which is the
property that makes the two throw skills bindable only to their own objects.
"""

import pytest

from hitl_pmp.environments.tossingroom.environment import (
    TossingRoomEnvironment,
)
from hitl_pmp.environments.tossingroom.predicates import (
    ADJACENT,
    HAND_EMPTY,
    HOLDING_RECYCLING,
    HOLDING_TRASH,
    RECYCLING_BIN_EMPTY,
    RECYCLING_BIN_IN_ROOM,
    RECYCLING_BUTTON_IN_ROOM,
    RECYCLING_IN_BIN,
    ROBOT_IN_ROOM,
    TRASH_BIN_EMPTY,
    TRASH_BIN_IN_ROOM,
    TRASH_BUTTON_IN_ROOM,
    TRASH_IN_BIN,
)
from hitl_pmp.environments.tossingroom.skill_provider import (
    TossingRoomSkillProvider,
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
        weight_seed=0,
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
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomEnvironment.TRASH_KIND),
    )
    assert HAND_EMPTY.holds(state, (_ROBOT,)) is False


def test_the_two_holding_predicates_track_their_own_kind() -> None:
    state = _state()
    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomEnvironment.RECYCLING_KIND),
    )
    assert HOLDING_RECYCLING.holds(state, (_ROBOT, _RECYCLING)) is True
    assert HOLDING_TRASH.holds(state, (_ROBOT, _TRASH)) is False

    state.set(
        obj=_ROBOT,
        feature_name="holding",
        feature_val=float(TossingRoomEnvironment.TRASH_KIND),
    )
    assert HOLDING_TRASH.holds(state, (_ROBOT, _TRASH)) is True
    assert HOLDING_RECYCLING.holds(state, (_ROBOT, _RECYCLING)) is False


def test_adjacent_holds_only_for_consecutive_rooms() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    assert ADJACENT.holds(state, (rooms[2], rooms[3])) is True
    assert ADJACENT.holds(state, (rooms[2], rooms[4])) is False


def test_in_bin_requires_a_non_empty_bin_of_that_kind() -> None:
    state = _state(recycling_count=1)
    assert RECYCLING_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is True
    # The trash bin is still empty, so the trash goal is not incidentally satisfied by
    # the recycling one -- the two families never share a bin count.
    assert TRASH_IN_BIN.holds(state, (_TRASH, _TRASH_BIN)) is False


def test_in_bin_is_false_for_an_empty_bin() -> None:
    state = _state(recycling_count=0)
    assert RECYCLING_IN_BIN.holds(state, (_RECYCLING, _RECYCLING_BIN)) is False


def test_bin_empty_holds_only_when_count_is_zero() -> None:
    empty = _state(recycling_count=0)
    non_empty = _state(recycling_count=1)
    assert RECYCLING_BIN_EMPTY.holds(empty, (_RECYCLING_BIN,)) is True
    assert RECYCLING_BIN_EMPTY.holds(non_empty, (_RECYCLING_BIN,)) is False


def test_bin_in_room_and_button_in_room_are_static_placements() -> None:
    state = _state()
    rooms = _ENV.get_rooms()
    assert (
        RECYCLING_BIN_IN_ROOM.holds(state, (_RECYCLING_BIN, rooms[_ENV.recycling_bin_room])) is True
    )
    assert TRASH_BIN_IN_ROOM.holds(state, (_TRASH_BIN, rooms[_ENV.trash_bin_room])) is True
    # Each button sits beside its own bin, so its room IS that bin's room.
    assert TRASH_BUTTON_IN_ROOM.holds(state, (_TRASH_BUTTON, rooms[_ENV.trash_bin_room])) is True
    assert (
        RECYCLING_BUTTON_IN_ROOM.holds(state, (_RECYCLING_BUTTON, rooms[_ENV.recycling_bin_room]))
        is True
    )
    assert TRASH_BUTTON_IN_ROOM.holds(state, (_TRASH_BUTTON, rooms[0])) is False


def test_each_buttons_declared_type_ties_it_to_exactly_one_bins_press() -> None:
    """The `ButtonForBin` analogue. Tossing Room needs a predicate to tie each button to
    the one bin it empties; here the button types are split, so `PressTrash` can only
    ever bind `trash_button` and `PressRecycling` only `recycling_button`. That is the
    stronger form -- a mismatched pairing is unbindable rather than merely disallowed."""
    assert TRASH_BUTTON_IN_ROOM.types[0] == TossingRoomEnvironment.trash_button_type
    assert RECYCLING_BUTTON_IN_ROOM.types[0] == TossingRoomEnvironment.recycling_button_type
    assert _TRASH_BUTTON.type != _RECYCLING_BUTTON.type


def test_predicates_declare_their_types_and_names() -> None:
    assert ROBOT_IN_ROOM.name == "RobotInRoom"
    assert TRASH_IN_BIN.types == (
        TossingRoomEnvironment.trash_type,
        TossingRoomEnvironment.trash_bin_type,
    )
    assert RECYCLING_IN_BIN.types == (
        TossingRoomEnvironment.recycling_type,
        TossingRoomEnvironment.recycling_bin_type,
    )
    assert TRASH_BIN_EMPTY.types == (TossingRoomEnvironment.trash_bin_type,)
    assert RECYCLING_BIN_EMPTY.types == (TossingRoomEnvironment.recycling_bin_type,)


class TestTheSplitPredicatesCannotBeAppliedAcrossKinds:
    """Where the separation is actually enforced.

    **Not** in `GroundAtom`: it is deliberately un-validated (`core/problem/tasks/
    types.py` declares `predicate` and `objects` with no cross-check), so
    `TrashInBin(recycling, recycling_bin)` can be *constructed* by hand. That is not a
    hole -- nothing in a real run constructs an atom that way. Atoms reach a Method
    through `SkillGrounder.abstract_state`, which builds each slot's candidate list as
    `[obj for obj in objects if obj.type == object_type]`, and reach a plan through
    `GroundSkill`, which *does* validate. So the tests below pin those two, and the
    declared types that drive them.
    """

    @staticmethod
    @pytest.mark.parametrize(
        ("predicate", "wrong_object"),
        [
            (TRASH_IN_BIN, _RECYCLING_BIN),
            (RECYCLING_IN_BIN, _TRASH_BIN),
            (HOLDING_TRASH, _RECYCLING),
            (HOLDING_RECYCLING, _TRASH),
        ],
    )
    def test_the_other_kinds_object_does_not_match_the_declared_slot_type(
        *, predicate, wrong_object
    ) -> None:
        """`abstract_state`'s filter is exactly this equality, so a mismatch here is what
        keeps the cross-kind atom out of every abstraction."""
        assert wrong_object.type != predicate.types[-1]

    @staticmethod
    def test_the_abstraction_contains_no_cross_kind_atom() -> None:
        """Non-vacuity: the state below has BOTH bins non-empty and the robot holding
        something, so every in-bin/holding predicate has a chance to fire. Only the
        same-kind atoms may appear."""
        provider = TossingRoomSkillProvider(env=_ENV)
        state = _state(recycling_count=1, trash_count=1)
        state.set(
            obj=_ROBOT,
            feature_name="holding",
            feature_val=float(TossingRoomEnvironment.TRASH_KIND),
        )
        from hitl_pmp.planning.grounding import SkillGrounder

        atoms = SkillGrounder.abstract_state(
            state=state, objects=provider.objects(), predicates=provider.predicates()
        )
        in_bin = {
            (atom.predicate.name, tuple(obj.name for obj in atom.objects))
            for atom in atoms
            if atom.predicate in (TRASH_IN_BIN, RECYCLING_IN_BIN)
        }
        assert in_bin == {
            ("TrashInBin", ("trash", "trash_bin")),
            ("RecyclingInBin", ("recycling", "recycling_bin")),
        }
        holding = {
            (atom.predicate.name, tuple(obj.name for obj in atom.objects))
            for atom in atoms
            if atom.predicate in (HOLDING_TRASH, HOLDING_RECYCLING)
        }
        assert holding == {("HoldingTrash", ("robot", "trash"))}
