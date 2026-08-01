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
    """Whether an item of this kind is in the (matching) bin: the bin holds at least
    one item AND it is the correct bin for the item (item.kind == bin.kind). The kind
    match is what makes ItemInBin(trash, recycling_bin) false even when the recycling
    bin is non-empty."""

    @staticmethod
    def holds(*, state: State, item: Object, bin_obj: Object) -> bool:
        count = int(round(state.get(obj=bin_obj, feature_name="count")))
        item_kind = int(round(state.get(obj=item, feature_name="kind")))
        bin_kind = int(round(state.get(obj=bin_obj, feature_name="kind")))
        return count >= 1 and item_kind == bin_kind


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

BUTTON_IN_ROOM = Predicate(
    name="ButtonInRoom",
    types=(TossingRoomEnvironment.button_type, TossingRoomEnvironment.room_type),
    holds=lambda state, objects: ButtonInRoomClassifier.holds(
        state=state, button=objects[0], room=objects[1]
    ),
)
