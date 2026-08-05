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
    equals that item's kind (and the hand is not empty)."""

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
    """Whether THIS item is in the (matching) bin: the item's own in_bin flag is set
    AND it is the correct bin for it (item.kind == bin.kind). The kind match is what
    makes ItemInBin(trash, recycling_bin) false even when the recycling bin holds the
    recycling item.

    It reads the item's flag rather than the bin's `count` because this predicate is
    Throw's add effect, and EES scores a skill attempt by
    `add_effects <= atoms(next_state)` -- that one boolean is both the competence
    observation and the sampler's training label. A Throw always releases the item
    whether or not it lands, so "count >= 1" scored EVERY throw into an already-
    non-empty bin a success at any force whatsoever. Two mechanisms made bins
    non-empty: EMPTY-goal tasks prefill them, and within a practice period the first
    landed throw of a kind made every later throw of that kind free. Only the trash
    bin is reachable twice in a period (the recycling bin sits behind the one-way
    ledge), so the inflation was asymmetric and corrupted competence *ranking*, not
    just level. BinEmpty below keeps the count reading -- that predicate really is
    about the bin, and the EMPTY goal it serves is unaffected."""

    @staticmethod
    def holds(*, state: State, item: Object, bin_obj: Object) -> bool:
        in_bin = int(round(state.get(obj=item, feature_name="in_bin")))
        item_kind = int(round(state.get(obj=item, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return in_bin == 1 and item_kind == bin_kind


class BinEmptyClassifier:
    @staticmethod
    def holds(*, state: State, bin_obj: Object) -> bool:
        return int(round(state.get(obj=bin_obj, feature_name="count"))) == 0


class BinInRoomClassifier:
    @staticmethod
    def holds(*, state: State, bin_obj: Object, room: Object) -> bool:
        bin_room = int(round(state.get(obj=bin_obj, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return bin_room == room_index


class BinAcceptsItemClassifier:
    """True when this bin takes this item's kind. `_apply_throw` routes purely by the
    HELD item's kind and ignores the bound bin, so without this Throw was applicable
    with a mismatched bin and could never succeed at any force."""

    @staticmethod
    def holds(*, state: State, item: Object, bin_obj: Object) -> bool:
        item_kind = int(round(state.get(obj=item, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return item_kind == bin_kind


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
    than env config precisely so this classifier can exist -- see
    TossingRoomEnvironment.pile_type."""

    @staticmethod
    def holds(*, state: State, pile: Object, room: Object) -> bool:
        pile_room = int(round(state.get(obj=pile, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return pile_room == room_index


class ButtonInRoomClassifier:
    @staticmethod
    def holds(*, state: State, button: Object, room: Object) -> bool:
        button_room = int(round(state.get(obj=button, feature_name="room")))
        room_index = int(round(state.get(obj=room, feature_name="index")))
        return button_room == room_index


# Predicate.holds is a positional (state, objects) callable per its interface contract
# (Goal.is_satisfied calls it that way) -- each lambda below just adapts that into a
# call to the relevant class's keyword-only holds, exactly like Light Switch's
# predicates.py.
ROBOT_IN_ROOM = Predicate(
    name="RobotInRoom",
    types=(TossingRoomEnvironment.robot_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: RobotInRoomClassifier.holds(
        state=state, robot=objects[0], room=objects[1]
    ),
)

HAND_EMPTY = Predicate(
    name="HandEmpty",
    types=(TossingRoomEnvironment.robot_type,),
    holds=lambda state, objects: HandEmptyClassifier.holds(state=state, robot=objects[0]),
)

HOLDING = Predicate(
    name="Holding",
    types=(TossingRoomEnvironment.robot_type, TossingRoomEnvironment.item_type),
    holds=lambda state, objects: HoldingClassifier.holds(
        state=state, robot=objects[0], item=objects[1]
    ),
)

ADJACENT = Predicate(
    name="Adjacent",
    types=(TossingRoomEnvironment.room_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: AdjacentClassifier.holds(
        state=state, room1=objects[0], room2=objects[1]
    ),
)

ITEM_IN_BIN = Predicate(
    name="ItemInBin",
    types=(TossingRoomEnvironment.item_type, TossingRoomEnvironment.bin_type),
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
    types=(TossingRoomEnvironment.bin_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: BinInRoomClassifier.holds(
        state=state, bin_obj=objects[0], room=objects[1]
    ),
)

BIN_ACCEPTS_ITEM = Predicate(
    name="BinAcceptsItem",
    types=(TossingRoomEnvironment.item_type, TossingRoomEnvironment.bin_type),
    holds=lambda state, objects: BinAcceptsItemClassifier.holds(
        state=state, item=objects[0], bin_obj=objects[1]
    ),
)

CAN_MOVE_ROOM = Predicate(
    name="CanMoveRoom",
    types=(TossingRoomEnvironment.room_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: CanMoveRoomClassifier.holds(
        state=state, from_room=objects[0], to_room=objects[1]
    ),
)

PILE_IN_ROOM = Predicate(
    name="PileInRoom",
    types=(TossingRoomEnvironment.pile_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: PileInRoomClassifier.holds(
        state=state, pile=objects[0], room=objects[1]
    ),
)

BUTTON_IN_ROOM = Predicate(
    name="ButtonInRoom",
    types=(TossingRoomEnvironment.button_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: ButtonInRoomClassifier.holds(
        state=state, button=objects[0], room=objects[1]
    ),
)
