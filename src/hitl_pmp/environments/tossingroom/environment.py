from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type


class TossingRoomEnvironment(Environment):
    """The "Tossing Room" environment: a robot in a 1-D hallway of rooms must sort
    items into the correct bin. A limitless pile of trash and recycling sits at the
    robot's start room; the recycling bin is in one room, the trash bin and an
    "empty/incinerate" button in another. A ONE-WAY LEDGE sits between two adjacent
    rooms: the robot can step LEFT across it (down in index) but never RIGHT (up in
    index) -- that rightward step is the domain's single irreversible action. The
    recycling bin therefore sits behind the ledge (closer to start but only reachable
    by dropping down), while the trash bin sits freely reachable in the other
    direction. No HumanOracle exists yet (Problem.human stays None): the oracle solves
    every task forward-only, so it never needs to be lifted back up the ledge.

    This is a fresh first-principles environment (not a port), but it deliberately
    mirrors LightSwitchEnvironment's structure exactly -- the same real-per-instance
    vs. ClassVar split. Layout config (num_rooms/start_room/the bin & button rooms/
    the ledge/throw_tolerance) is genuine per-run configuration, so those are
    constructor arguments (e.g. TossingRoomEnvironment(num_rooms=9)). The types, the
    fixed Objects (robot/bins/button/item discriminators), the discrete skill-id
    encoding and action_space are structural constants shared by every instance that
    will ever exist, so they stay ClassVar. Unlike LightSwitchEnvironment, NO value
    here needs to stay ClassVar for predicates.py's sake: every TossingRoom predicate
    classifies purely from features already in the State (a robot's room index, a
    bin's count, an item's kind), never from instance layout config, so nothing forces
    start_room/throw_tolerance to be class-level.

    The raw action is [skill_id, arg, force] (3-D, continuous/unbounded to match the
    Box action_space, decoded by rounding the first two entries):
      * skill_id -- which skill: 0 Pickup, 1 MoveRoom, 2 Throw, 3 Press.
      * arg -- skill-dependent integer: Pickup/Throw = item kind (1 trash,
        2 recycling); MoveRoom = destination room index; Press = unused.
      * force -- only read by Throw (the continuous dial, like Light Switch's dlight).
    take_action is TOTAL over the whole Box: any action out of context (wrong room,
    hand already full, a non-finite or unknown value) is a silent no-op, never a
    crash -- an environment exposed to a raw action space must handle all of it.
    """

    # Discrete skill ids, shared by the decode below and skills.py's compute_action
    # encode -- one source of truth for both directions.
    SKILL_PICKUP: ClassVar[int] = 0
    SKILL_MOVE_ROOM: ClassVar[int] = 1
    SKILL_THROW: ClassVar[int] = 2
    SKILL_PRESS: ClassVar[int] = 3

    # Item-kind discriminators (also the robot's "holding" encoding: 0 = empty hand).
    TRASH_KIND: ClassVar[int] = 1
    RECYCLING_KIND: ClassVar[int] = 2

    num_rooms: int = 7
    start_room: int = 3
    recycling_bin_room: int = 1
    trash_bin_room: int = 6
    button_room: int = 6
    # The one-way ledge sits between blocked_right_from and blocked_right_from + 1:
    # stepping RIGHT from blocked_right_from is blocked; every other adjacent step
    # (including stepping LEFT back across it) is allowed.
    blocked_right_from: int = 2
    throw_tolerance: float = (
        0.1  # max |force - target| for a Throw to land, like Light Switch's 0.1
    )
    # hard_reset's non-random per-item target force; only used by the canonical reset
    # state, never by task sampling (Tasks samples its own per-task forces).
    canonical_target_force: float = 0.5

    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("room", "holding"))
    # blocks_right marks the one-way ledge. Like the pile's room, this lives in the
    # STATE rather than in config so a module-level Predicate -- whose signature is
    # only (state, objects) -- can read it and keep MoveRoom's model as strong as
    # _apply_move's guard.
    room_type: ClassVar[Type] = Type(name="room", feature_names=("index", "blocks_right"))
    bin_type: ClassVar[Type] = Type(name="bin", feature_names=("count", "room", "kind"))
    button_type: ClassVar[Type] = Type(name="button", feature_names=("room",))
    # The limitless item pile. Modelled as a real object with a room feature -- the
    # same shape as `button` -- so a module-level Predicate can tell which room it is
    # in. `start_room` is per-instance config, which a Predicate (whose signature is
    # only (state, objects)) cannot read; putting the pile in the STATE is what lets
    # Pickup's symbolic precondition be exactly as strong as _apply_pickup's guard.
    pile_type: ClassVar[Type] = Type(name="pile", feature_names=("room",))
    item_type: ClassVar[Type] = Type(name="item", feature_names=("kind", "target_force"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    recycling_bin: ClassVar[Object] = Object(name="recycling_bin", type=bin_type)
    trash_bin: ClassVar[Object] = Object(name="trash_bin", type=bin_type)
    button: ClassVar[Object] = Object(name="button", type=button_type)
    pile: ClassVar[Object] = Object(name="pile", type=pile_type)
    # Singleton discriminator objects, so skills/predicates/goals have a concrete
    # Object to bind their "item" argument to. Their kind feature is what maps a held
    # item back to the right bin/room.
    trash: ClassVar[Object] = Object(name="trash", type=item_type)
    recycling: ClassVar[Object] = Object(name="recycling", type=item_type)

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (3,))

    def get_rooms(self) -> tuple[Object, ...]:
        """One Object per room, index feature = i. Built fresh every call (not cached)
        -- num_rooms can differ between instances (CLI override, test overrides), so
        caching risks a stale value; Object equality/hash are value-based (frozen
        pydantic), so rebuilding is correct, just not free (negligible at this scale).
        Mirrors LightSwitchEnvironment.get_cells."""
        return tuple(Object(name=f"room_{i}", type=self.room_type) for i in range(self.num_rooms))

    def bin_for_kind(self, *, kind: int) -> Object:
        return self.recycling_bin if kind == self.RECYCLING_KIND else self.trash_bin

    def item_for_kind(self, *, kind: int) -> Object:
        return self.recycling if kind == self.RECYCLING_KIND else self.trash

    def bin_room_for_kind(self, *, kind: int) -> int:
        return self.recycling_bin_room if kind == self.RECYCLING_KIND else self.trash_bin_room

    def build_initial_state(
        self,
        *,
        trash_target_force: float,
        recycling_target_force: float,
        recycling_count: int = 0,
        trash_count: int = 0,
    ) -> State:
        """The robot always starts in start_room with an empty hand; the bins/button
        sit in their configured rooms. Only the per-item throw targets and the initial
        bin counts vary between callers -- hard_reset uses canonical values with empty
        bins, Tasks samples targets per episode (and non-empty counts for the
        empty-buckets goal). Mirrors LightSwitchEnvironment.build_initial_state."""
        data: dict[Object, np.ndarray] = {
            self.robot: np.array([float(self.start_room), 0.0]),
            self.recycling_bin: np.array([
                float(recycling_count),
                float(self.recycling_bin_room),
                float(self.RECYCLING_KIND),
            ]),
            self.trash_bin: np.array([
                float(trash_count),
                float(self.trash_bin_room),
                float(self.TRASH_KIND),
            ]),
            self.button: np.array([float(self.button_room)]),
            self.pile: np.array([float(self.start_room)]),
            self.trash: np.array([float(self.TRASH_KIND), float(trash_target_force)]),
            self.recycling: np.array([float(self.RECYCLING_KIND), float(recycling_target_force)]),
        }
        for i, room in enumerate(self.get_rooms()):
            data[room] = np.array([float(i), float(i == self.blocked_right_from)])
        return State(data=data)

    def take_action(self, *, action: Action) -> State:
        state = self.get_current_state()
        next_state = state.model_copy(deep=True)

        raw_skill, raw_arg, raw_force = float(action[0]), float(action[1]), float(action[2])
        # Totality guard: the Box contains +-inf, and round(inf) raises OverflowError.
        # A non-finite skill/arg is out of context -- a silent no-op, never a crash.
        if np.isfinite(raw_skill) and np.isfinite(raw_arg):
            skill_id, arg = int(round(raw_skill)), int(round(raw_arg))
            robot_room = int(round(state.get(obj=self.robot, feature_name="room")))
            holding = int(round(state.get(obj=self.robot, feature_name="holding")))

            if skill_id == self.SKILL_PICKUP:
                self._apply_pickup(
                    next_state=next_state, robot_room=robot_room, holding=holding, arg=arg
                )
            elif skill_id == self.SKILL_MOVE_ROOM:
                self._apply_move(next_state=next_state, robot_room=robot_room, to_room=arg)
            elif skill_id == self.SKILL_THROW:
                self._apply_throw(
                    state=state,
                    next_state=next_state,
                    robot_room=robot_room,
                    holding=holding,
                    raw_force=raw_force,
                )
            elif skill_id == self.SKILL_PRESS:
                self._apply_press(next_state=next_state, robot_room=robot_room)
            # Any other skill_id is unknown -> no-op.

        self.set_state(state=next_state)
        return next_state

    def _apply_pickup(self, *, next_state: State, robot_room: int, holding: int, arg: int) -> None:
        # Pickup only from the limitless pile at start_room, and only with an empty
        # hand; arg must name a real item kind.
        if (
            robot_room == self.start_room
            and holding == 0
            and arg in (self.TRASH_KIND, self.RECYCLING_KIND)
        ):
            next_state.set(obj=self.robot, feature_name="holding", feature_val=float(arg))

    def _apply_move(self, *, next_state: State, robot_room: int, to_room: int) -> None:
        if 0 <= to_room < self.num_rooms and abs(to_room - robot_room) == 1:
            # The one-way ledge: stepping RIGHT from blocked_right_from is the single
            # irreversible-blocked move (a no-op). Everything else, including stepping
            # LEFT back across it, is allowed.
            crosses_ledge_rightward = (
                robot_room == self.blocked_right_from and to_room == self.blocked_right_from + 1
            )
            if not crosses_ledge_rightward:
                next_state.set(obj=self.robot, feature_name="room", feature_val=float(to_room))

    def _apply_throw(
        self,
        *,
        state: State,
        next_state: State,
        robot_room: int,
        holding: int,
        raw_force: float,
    ) -> None:
        if holding not in (self.TRASH_KIND, self.RECYCLING_KIND) or not np.isfinite(raw_force):
            return
        bin_obj = self.bin_for_kind(kind=holding)
        item_obj = self.item_for_kind(kind=holding)
        bin_room = self.bin_room_for_kind(kind=holding)
        target = state.get(obj=item_obj, feature_name="target_force")
        # Throwing always releases the item, whether or not it lands. It lands only in
        # the item's own bin room and only when the dial is within tolerance of that
        # item's target (Light Switch's LightOn logic, but within a single room -- the
        # throw never crosses rooms).
        #
        # The release is what makes a miss cost something. A miss used to change
        # nothing at all, so the robot still held the item and still stood in the bin
        # room and the very next step re-threw for free -- which quietly turned the
        # evaluation horizon into a "number of attempts" dial (an unpracticed EES
        # scored 94.7%, purely by re-rolling a ~0.19 chance). The thrown item is gone
        # rather than recoverable: items are singleton discriminators carrying only
        # (kind, target_force) with no position, so "it is lying near the bin" is not
        # representable, and making it so would just restore a cheap retry. Trying
        # again therefore means a fresh item from the limitless pile, which costs a
        # round trip to the start room -- affordable inside a 100-step practice
        # period, and not inside an evaluation horizon of longest-solve + 2. That
        # round trip has since been measured at exactly 8 steps for TRASH, putting a
        # second attempt at step 13; for RECYCLING the one-way ledge makes it
        # impossible at any horizon. The 94.7% above is the pre-release measurement
        # that motivated this and is quoted as history -- it was taken on the sampled
        # test-set composition too, so it does not reproduce against current code.
        next_state.set(obj=self.robot, feature_name="holding", feature_val=0.0)
        if robot_room == bin_room and abs(raw_force - target) < self.throw_tolerance:
            count = state.get(obj=bin_obj, feature_name="count")
            next_state.set(obj=bin_obj, feature_name="count", feature_val=count + 1.0)

    def _apply_press(self, *, next_state: State, robot_room: int) -> None:
        if robot_room == self.button_room:
            next_state.set(obj=self.recycling_bin, feature_name="count", feature_val=0.0)
            next_state.set(obj=self.trash_bin, feature_name="count", feature_val=0.0)

    def get_valid_actions(self) -> list[Action]:
        # The force dimension is continuous and unbounded (matches Light Switch's
        # convention), so there is no finite/enumerable action list to return.
        return []

    def hard_reset(self) -> None:
        self.set_state(
            state=self.build_initial_state(
                trash_target_force=self.canonical_target_force,
                recycling_target_force=self.canonical_target_force,
            )
        )
