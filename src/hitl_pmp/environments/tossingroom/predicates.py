"""Symbolic classifiers for Tossing Room (split throws).

Every *classifier* below is the same logic as `environments/tossingroom/predicates.py`
-- they read the same feature names off the same feature schemas, so the two files diff
cleanly. What changes is the `Predicate` declarations at the bottom: because the item,
bin and button types are split (see `environment.py`), each kind gets its own predicate,
and two predicates are dropped outright.

Split, one per kind:
  * `Holding` -> `HoldingTrash` / `HoldingRecycling`
  * `ItemInBin` -> `TrashInBin` / `RecyclingInBin`
  * `BinInRoom` -> `TrashBinInRoom` / `RecyclingBinInRoom`
  * `BinEmpty` -> `TrashBinEmpty` / `RecyclingBinEmpty`
  * `ButtonInRoom` -> `TrashButtonInRoom` / `RecyclingButtonInRoom`

Dropped, both for the same reason -- a *type* already enforces what the predicate
asserts, so it would be a tautology, and keeping a tautology as a precondition is worse
than dropping it: it reads as a live constraint while constraining nothing.

  * `BinAcceptsItem`. Tossing Room needed it because its single `Throw` could bind any
    bin in the room to the held item, while `_apply_throw` routes purely by the held
    item's kind -- so `Throw(trash -> recycling_bin)` was applicable and could never
    succeed at any force. Here `ThrowTrash`'s bin parameter is typed `trash_bin` and
    only one object has that type, so the pairing is enforced by the grounder itself.
    **Re-checked against the capacity-1 redesign**, which gave Tossing Room's `Press` an
    `?item` parameter pinned by exactly this predicate (so that `Press` can name the
    in-bin atom it deletes): the argument survives unchanged, because `PressTrash`'s item
    parameter is typed `trash` and its bin parameter `trash_bin`, one object each.
  * `ButtonForBin`, which Tossing Room added with that same redesign to tie each button
    to the one bin it empties. `PressTrash` binds `trash_button` and `trash_bin` by type,
    so the tie is structural here rather than asserted. This is the strictly stronger
    version of what that predicate buys: a mismatched pairing is not merely disallowed,
    it is unbindable, under any layout -- including a degenerate one that puts both bins
    in the same room, where `ButtonInRoom` alone would not separate them.

`tests/environments/tossingroom/test_skills.py` asserts a cross-kind grounding is
rejected, which is the property both dropped predicates used to buy.

**Under `--unsplit-skills` (`TossingRoomEnvironment.unsplit_skills`) the split above is
reversed**, and the second block of `Predicate` declarations at the bottom of this file is
what that arm grounds over: `Holding`, `ItemInBin`, `BinEmpty`, `BinInRoom` and
`ButtonInRoom` over the shared item/bin/button types, plus `BinAcceptsItem` and
`ButtonForBin` -- which are not tautologies there, because with one item type and one bin
type a `Throw(trash -> recycling_bin)` grounding is well-typed and has to be ruled out by
an asserted constraint instead. Every *classifier* is shared between the two blocks; only
the `Predicate`s' declared `types` differ, which is the whole of what the flag changes
here. `ROBOT_IN_ROOM`, `HAND_EMPTY`, `ADJACENT`, `CAN_MOVE_ROOM` and `PILE_IN_ROOM` are
shared outright: their types carry no kind to split on.
"""

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import TossingRoomEnvironment


class RobotInRoomClassifier:
    """Whether the robot's room index matches a room object's index. A static-method
    container, never instantiated, same as every other business-logic class in this
    project."""

    @staticmethod
    def holds(*, state: State, robot: Object, room: Object) -> bool:
        robot_room = int(round(state.get(obj=robot, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return robot_room == room_index


class HandEmptyClassifier:
    @staticmethod
    def holds(*, state: State, robot: Object) -> bool:
        return int(round(state.get(obj=robot, feature_name="holding"))) == 0


class HoldingClassifier:
    """Whether the robot is holding the given item, i.e. the robot's holding code
    equals that item's kind (and the hand is not empty). Shared by both `HoldingTrash`
    and `HoldingRecycling` -- the two differ only in which item type they accept, which
    the `Predicate`'s own `types` declares."""

    @staticmethod
    def holds(*, state: State, robot: Object, item: Object) -> bool:
        holding = int(round(state.get(obj=robot, feature_name="holding")))
        kind = int(round(state.get(obj=item, feature_name="kind")))
        return holding != 0 and holding == kind


class AdjacentClassifier:
    """Whether two rooms are next to each other (indices exactly one apart)."""

    @staticmethod
    def holds(*, state: State, room1: Object, room2: Object) -> bool:
        index1 = int(round(state.get(obj=room1, feature_name="index")))
        index2 = int(round(state.get(obj=room2, feature_name="index")))
        return abs(index1 - index2) == 1


class ItemInBinClassifier:
    """Whether an item of this kind is in the (matching) bin: the bin is non-empty AND
    it is the correct bin for the item (item.kind == bin.kind).

    The kind check is redundant at the symbolic layer -- the types already stop a trash
    item being paired with the recycling bin -- but it is kept so this classifier is
    identical to Tossing Room's and stays correct on its own terms rather than depending
    on its callers to have declared the right types.

    Deliberately still count-based, and deliberately NOT paired with an `in_bin` flag on
    the item. A bin holds at most one item (`TossingRoomEnvironment.BIN_CAPACITY`)
    and each throw requires its bin to be empty, so the count is provably 0 at throw
    time -- which makes this predicate flip false -> true exactly once per throw and
    therefore an honest add-effect for EES to score against. `count in {0, 1}` is the
    single representation of a bin's contents, read here and by `BinEmptyClassifier`
    below; storing the same fact twice is what this redesign exists to avoid."""

    @staticmethod
    def holds(*, state: State, item: Object, bin_obj: Object) -> bool:
        count = int(round(state.get(obj=bin_obj, feature_name="count")))
        item_kind = int(round(state.get(obj=item, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return count >= 1 and item_kind == bin_kind


class BinEmptyClassifier:
    """The complement of `ItemInBinClassifier` over the same `count`, and the other half
    of what makes throw scoring honest: as `ThrowTrash`/`ThrowRecycling`'s precondition it
    pins the count to 0 before the throw."""

    @staticmethod
    def holds(*, state: State, bin_obj: Object) -> bool:
        return int(round(state.get(obj=bin_obj, feature_name="count"))) == 0


class BinInRoomClassifier:
    @staticmethod
    def holds(*, state: State, bin_obj: Object, room: Object) -> bool:
        bin_room = int(round(state.get(obj=bin_obj, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return bin_room == room_index


class CanMoveRoomClassifier:
    """Adjacency, minus the one-way ledge. `Adjacent` is symmetric, but `_apply_move`
    refuses the RIGHTWARD step across the ledge, so a symmetric precondition let the
    planner schedule a move the dynamics silently drop."""

    @staticmethod
    def holds(*, state: State, from_room: Object, to_room: Object) -> bool:
        if not AdjacentClassifier.holds(state=state, room1=from_room, room2=to_room):
            return False
        from_index = int(round(state.get(obj=from_room, feature_name="index")))
        to_index = int(round(state.get(obj=to_room, feature_name="index")))
        blocks_right = int(round(state.get(obj=from_room, feature_name="blocks_right"))) == 1
        return not (blocks_right and to_index == from_index + 1)


class PileInRoomClassifier:
    """True when the item pile sits in this room. The pile is a state object rather
    than env config precisely so this classifier can exist."""

    @staticmethod
    def holds(*, state: State, pile: Object, room: Object) -> bool:
        pile_room = int(round(state.get(obj=pile, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return pile_room == room_index


class BinAcceptsItemClassifier:
    """Whether this bin is the one that takes items of this kind. A tautology under the
    split types (one object per type, matched by the grounder), which is why no split
    predicate uses it -- but a live constraint under `--unsplit-skills`, where a single
    `Throw` can bind any bin to any item and `_apply_throw` routes purely by the HELD
    item's kind, so a mismatched pairing could never succeed at any force."""

    @staticmethod
    def holds(*, state: State, item: Object, bin_obj: Object) -> bool:
        item_kind = int(round(state.get(obj=item, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return item_kind == bin_kind


class ButtonForBinClassifier:
    """Which single bin a button empties -- each bin has its own button, beside it, and
    `_apply_press` empties only that one. The same story as `BinAcceptsItemClassifier`:
    the split button/bin types make this a tautology, the shared ones make it the thing
    that keeps `Press`'s effects expressible per bin."""

    @staticmethod
    def holds(*, state: State, button: Object, bin_obj: Object) -> bool:
        button_kind = int(round(state.get(obj=button, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return button_kind == bin_kind


class ButtonInRoomClassifier:
    """Where a button is. Shared by `TrashButtonInRoom` and `RecyclingButtonInRoom` --
    each button sits beside its own bin, so its room IS that bin's room."""

    @staticmethod
    def holds(*, state: State, button: Object, room: Object) -> bool:
        button_room = int(round(state.get(obj=button, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return button_room == room_index


# Predicate.holds is a positional (state, objects) callable per its interface contract
# (Goal.is_satisfied calls it that way) -- each lambda below just adapts that into a
# call to the relevant class's keyword-only holds.
ROBOT_IN_ROOM = Predicate(
    name="RobotInRoom",
    types=(
        TossingRoomEnvironment.robot_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: RobotInRoomClassifier.holds(
        state=state, robot=objects[0], room=objects[1]
    ),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(TossingRoomEnvironment.robot_type,),
    holds=lambda state, objects: HandEmptyClassifier.holds(state=state, robot=objects[0]),
)

HOLDING_TRASH = Predicate(
    name="HoldingTrash",
    types=(
        TossingRoomEnvironment.robot_type,
        TossingRoomEnvironment.trash_type,
    ),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], item=objects[1]
    ),
)

HOLDING_RECYCLING = Predicate(
    name="HoldingRecycling",
    types=(
        TossingRoomEnvironment.robot_type,
        TossingRoomEnvironment.recycling_type,
    ),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], item=objects[1]
    ),
)

ADJACENT = Predicate(
    name="Adjacent",
    types=(
        TossingRoomEnvironment.room_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: AdjacentClassifier.holds(
        state=state, room1=objects[0], room2=objects[1]
    ),
)

TRASH_IN_BIN = Predicate(
    name="TrashInBin",
    types=(
        TossingRoomEnvironment.trash_type,
        TossingRoomEnvironment.trash_bin_type,
    ),
    holds=lambda state, objects: ItemInBinClassifier.holds(
        state=state, item=objects[0], bin_obj=objects[1]
    ),
)

RECYCLING_IN_BIN = Predicate(
    name="RecyclingInBin",
    types=(
        TossingRoomEnvironment.recycling_type,
        TossingRoomEnvironment.recycling_bin_type,
    ),
    holds=lambda state, objects: ItemInBinClassifier.holds(
        state=state, item=objects[0], bin_obj=objects[1]
    ),
)

TRASH_BIN_EMPTY = Predicate(
    name="TrashBinEmpty",
    types=(TossingRoomEnvironment.trash_bin_type,),
    holds=lambda state, objects: BinEmptyClassifier.holds(state=state, bin_obj=objects[0]),
)

RECYCLING_BIN_EMPTY = Predicate(
    name="RecyclingBinEmpty",
    types=(TossingRoomEnvironment.recycling_bin_type,),
    holds=lambda state, objects: BinEmptyClassifier.holds(state=state, bin_obj=objects[0]),
)

TRASH_BIN_IN_ROOM = Predicate(
    name="TrashBinInRoom",
    types=(
        TossingRoomEnvironment.trash_bin_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: BinInRoomClassifier.holds(
        state=state, bin_obj=objects[0], room=objects[1]
    ),
)

RECYCLING_BIN_IN_ROOM = Predicate(
    name="RecyclingBinInRoom",
    types=(
        TossingRoomEnvironment.recycling_bin_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: BinInRoomClassifier.holds(
        state=state, bin_obj=objects[0], room=objects[1]
    ),
)

CAN_MOVE_ROOM = Predicate(
    name="CanMoveRoom",
    types=(
        TossingRoomEnvironment.room_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: CanMoveRoomClassifier.holds(
        state=state, from_room=objects[0], to_room=objects[1]
    ),
)

PILE_IN_ROOM = Predicate(
    name="PileInRoom",
    types=(
        TossingRoomEnvironment.pile_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: PileInRoomClassifier.holds(
        state=state, pile=objects[0], room=objects[1]
    ),
)

TRASH_BUTTON_IN_ROOM = Predicate(
    name="TrashButtonInRoom",
    types=(
        TossingRoomEnvironment.trash_button_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: ButtonInRoomClassifier.holds(
        state=state, button=objects[0], room=objects[1]
    ),
)

RECYCLING_BUTTON_IN_ROOM = Predicate(
    name="RecyclingButtonInRoom",
    types=(
        TossingRoomEnvironment.recycling_button_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: ButtonInRoomClassifier.holds(
        state=state, button=objects[0], room=objects[1]
    ),
)

# --------------------------------------------------------------------------------------
# The UNSPLIT symbolic layer, live only under `TossingRoomEnvironment.unsplit_skills`.
# Same classifiers as above, declared over the shared item/bin/button types -- so these
# are the predicates a single lifted `Throw` can be written in terms of. Restored from the
# retired original Tossing Room rather than reinvented; see this module's docstring.
# --------------------------------------------------------------------------------------

HOLDING = Predicate(
    name="Holding",
    types=(
        TossingRoomEnvironment.robot_type,
        TossingRoomEnvironment.item_type,
    ),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], item=objects[1]
    ),
)

ITEM_IN_BIN = Predicate(
    name="ItemInBin",
    types=(
        TossingRoomEnvironment.item_type,
        TossingRoomEnvironment.bin_type,
    ),
    holds=lambda state, objects: ItemInBinClassifier.holds(
        state=state, item=objects[0], bin_obj=objects[1]
    ),
)

BIN_EMPTY = Predicate(
    name="BinEmpty",
    types=(TossingRoomEnvironment.bin_type,),
    holds=lambda state, objects: BinEmptyClassifier.holds(state=state, bin_obj=objects[0]),
)

BIN_IN_ROOM = Predicate(
    name="BinInRoom",
    types=(
        TossingRoomEnvironment.bin_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: BinInRoomClassifier.holds(
        state=state, bin_obj=objects[0], room=objects[1]
    ),
)

BUTTON_IN_ROOM = Predicate(
    name="ButtonInRoom",
    types=(
        TossingRoomEnvironment.button_type,
        TossingRoomEnvironment.room_type,
    ),
    holds=lambda state, objects: ButtonInRoomClassifier.holds(
        state=state, button=objects[0], room=objects[1]
    ),
)

BIN_ACCEPTS_ITEM = Predicate(
    name="BinAcceptsItem",
    types=(
        TossingRoomEnvironment.item_type,
        TossingRoomEnvironment.bin_type,
    ),
    holds=lambda state, objects: BinAcceptsItemClassifier.holds(
        state=state, item=objects[0], bin_obj=objects[1]
    ),
)

BUTTON_FOR_BIN = Predicate(
    name="ButtonForBin",
    types=(
        TossingRoomEnvironment.button_type,
        TossingRoomEnvironment.bin_type,
    ),
    holds=lambda state, objects: ButtonForBinClassifier.holds(
        state=state, button=objects[0], bin_obj=objects[1]
    ),
)
