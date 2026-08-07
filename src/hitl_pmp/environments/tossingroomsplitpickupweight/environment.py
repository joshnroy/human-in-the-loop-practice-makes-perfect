from typing import ClassVar

import numpy as np
from gymnasium.spaces import Box
from pydantic import PrivateAttr

from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type


class TossingRoomSplitPickupWeightEnvironment(Environment):
    """ "Tossing Room (split throws, weight drawn at pickup)": `tossingroomsplit` with
    **one** change to the world -- an item's `weight` is a property of the item the
    robot actually picked up, drawn when it picks it up, rather than a per-task constant
    frozen into the task's initial state. `throw_distance` is fixed in exchange, so the
    required-force relation collapses to a one-dimensional function of weight.

    **This is a new domain, not a flag on the old one, and its results cannot be pooled
    with `tossingroomsplit`'s.** The task distribution genuinely differs:
    `tossingroomsplit` draws one weight and one distance per task and holds both for the
    episode; here every pickup redraws the weight and the distance never varies at all.
    `tossingroomsplitidentity` is the precedent for forking rather than adding a switch.

    **Why the weight moves to pickup time.** The reset-free A/B
    (`docs/experiment-logs/2026-08-06-reset-free-practice-ab.md`) found that a
    `--practice-reset-policy never` arm freezes every state feature no action writes --
    including the item weight and the bin distance, which ARE the throw sampler's input
    row -- at its `hard_reset` value for the whole run. So the reset-free arm practised
    every throw at a single point of the task distribution. `reset_to_task` is the only
    thing that installs a task's continuous parameters, and the reset-free arm never
    calls it. Drawing the weight inside `_apply_pickup` puts the variation back on an
    action the robot takes, so it survives the absence of any reset.

    **How the schedule works, and the three properties it has to have.**

      * **Pre-sampled, never drawn inside `take_action`.** `build_initial_state`
        materialises the whole array up front; the dynamics only ever INDEX it. Nothing
        in `take_action` consumes randomness, so a second, independently-constructed
        Environment (which is exactly what `--practice-reset-policy never` requires for
        evaluation -- see `PracticeLoop`'s "separate evaluation environment" section)
        cannot shift what practice draws.
      * **Per task, not one flat per-run stream.** Each task carries its own
        `weight_seed`, and the array is a pure function of it, so two arms that spend
        different numbers of pickups on one task still meet the next task at identical
        weights. A single run-long stream would make every later task's weights depend
        on how busy an arm had been earlier, which is a confound rather than a
        manipulation.
      * **It never wraps.** Running off the end raises (see `_next_pickup_weight`).
        Replaying the array would make the training distribution periodic, which is
        indistinguishable from a real learning signal in anything downstream.

    The seed and the cursor live in the **State**, on the pile -- the object that issues
    the items -- for the same reason the pile's room does: a `State` is the only thing
    that travels with a `Task`, and the test set is drawn once and replayed at every
    checkpoint, so a schedule parked on the Environment would be overwritten by the next
    task built. Parking them on the pile specifically, rather than on the robot or an
    item, is what keeps every throw sampler's input row byte-for-byte the shape
    `tossingroomsplit` has: the throws bind `(robot, item, bin, room)` and never the
    pile, and the only skills that do bind it are the two `param_dim=0` pickups, which
    have no sampler at all.

    Putting the cursor in the State also buys the fixed test set for free:
    `reset_to_task` rewinds it, so the same test task attempted at every checkpoint is
    attempted at the same weights.

    Everything below this point is `TossingRoomSplitEnvironment` verbatim.

    Throwing trash and throwing recycling are two separate lifted skills rather than one
    `Throw` whose `item` variable ranges over both kinds.

    **The layout is deliberately identical**, room for room: 7 rooms, the pile and the
    robot start in room 3, the recycling bin (and its own empty/incinerate button) in
    room 1, the trash bin (and its own button) in room 6, and the one-way ledge between
    rooms 2 and 3 (the RIGHTWARD step out of room 2 is the domain's single
    irreversible-blocked action). The dynamics below are the same code as Tossing Room's,
    unchanged: the raw action is still `[skill_id, arg, force]`, `take_action` is still
    total over the whole `Box`, and every guard is the same -- **including the capacity-1
    bin and its per-bin buttons**, which are ported here verbatim rather than reinvented.
    Nothing about the *world* differs, so a reviewer can diff the two files and see the
    change is entirely in the symbolic layer.

    **A BIN HOLDS AT MOST ONE ITEM**, and `_apply_throw` REFUSES a throw at a full one
    (a silent no-op, like every other out-of-context action). That capacity is what makes
    the count-based in-bin predicate -- which is also each throw's add-effect -- mean
    "this throw landed" rather than "somebody's item is in there": with each throw
    carrying its bin's matching empty precondition, the count is provably 0 at throw
    time, so the atom goes false -> true exactly once per throw. One button per bin
    follows from the same choice: a single button emptying both bins made `Press`'s
    effect on the in-bin atoms a universal delete that no per-bin effect could express.

    Consequence, deliberately accepted and shared with Tossing Room: the EMPTY goal
    family is an ORDERING task. Both buttons must be pressed, the recycling one sits
    behind the one-way ledge, so the only solution is to press the trash button first and
    then drop across. The reverse order is unsolvable.

    That layout is what makes the domain interesting, and it is why it was not
    simplified:

    * **Trash** is a round trip -- `PickupTrash` in room 3, three `MoveRoom`s right to
      room 6, `ThrowTrash`, then three `MoveRoom`s back for another item. Eight steps per
      attempt, so a practice period buys several attempts. (A throw that *landed* fills
      the bin, so the next attempt also needs a `PressTrash` -- the trash button is in
      room 6 beside its bin, so that costs one extra step and no extra walking.)
    * **Recycling** is one-way -- `PickupRecycling` in room 3, one `MoveRoom` LEFT across
      the ledge into room 2 and one more into room 1, `ThrowRecycling`. The ledge makes
      the return to room 3 impossible, and the pile is the only source of items, so once
      that throw is spent there is no second attempt at any horizon. A practice period
      therefore buys exactly one.

    **What differs from Tossing Room, and why.** `item_type` is split into `trash_type`
    and `recycling_type`, `bin_type` into `trash_bin_type` and `recycling_bin_type`, and
    `button_type` into `trash_button_type` and `recycling_button_type`. The types are the
    enforcement mechanism: `SkillGrounder` binds a skill parameter only to objects whose
    `type` matches exactly, and `GroundSkill` validates the same thing at construction,
    so `ThrowTrash` provably cannot bind the recycling item. A precondition could not do
    this as strongly -- it would leave an `item` variable that still *ranges* over both
    kinds, and a future edit dropping the precondition would silently restore the shared
    binding. The button split is the same argument applied to Tossing Room's
    `ButtonForBin`: that predicate exists there to stop one `Press` binding the wrong
    bin's button, and here the type does it outright, so the predicate would be a
    tautology (exactly like `BinAcceptsItem` -- see `predicates.py`).

    **THE FORCE A THROW NEEDS IS NEVER IN THE STATE; ITS CAUSE IS.** The relation is
    ported unchanged from `TossingRoomEnvironment` -- read that class's docstring for the
    full argument. In brief: a throw lands when `|force - required_force| <
    throw_tolerance`, and `required_force` is `reference_force + distance_coefficient *
    (bin.throw_distance - reference_distance) + weight_coefficient * (item.weight -
    reference_weight)`. The item's `weight` is observable; the five constants are
    environment configuration that never enters a State. This replaced an
    `item.target_force` feature that sat at index 4 of each throw's own classifier input
    row and *was* the answer, so both samplers were learning `|x_10 - x_4| < 0.1`.

    Here the bin's `throw_distance` is **fixed** at `throw_distance` for every task, so
    that term is a constant and the relation is one-dimensional in the weight. The
    feature is deliberately kept on both bin types anyway, at the same index: dropping it
    would change every throw sampler's input width and stop this domain being comparable
    with `tossingroomsplit` at the same architecture.

    **The port matters more here than anywhere else.** This domain exists to compare
    `ThrowTrash` and `ThrowRecycling` as two samplers of identical architecture on
    different practice budgets. If Tossing Room learned a relation and this fork still
    learned an identity, the two domains would stop measuring the same thing, and the
    comparison to `environments/tossingroom`'s baseline -- which the split-throw
    experiment quotes directly -- would be between different learning problems.

    The feature schemas are kept **identical** across the split pairs
    (`(kind, weight)` for both item types, `(count, room, kind, throw_distance)` for both
    bin types), even though `kind` is now derivable from the type. `EesMethod.state_features`
    is `concat(state[obj] for obj in ground_skill.objects)`, so identical schemas are
    what make the two throw samplers have the same input width -- the same architecture
    with different weights, which is the comparison the experiment needs. Dropping the
    now-redundant `kind` feature would have made the two samplers differently shaped and
    confounded "learned less" with "a different network".

    Splitting the item types also forces `Pickup` to split into `PickupTrash`/
    `PickupRecycling` (a `trash` object cannot bind a parameter typed `recycling`), and
    splitting the bin/button types forces `Press` to split into `PressTrash`/
    `PressRecycling`. All three are `param_dim=0`, so none has a sampler and nothing
    about learning changes; the throw split is still the only one that does. See
    `skills.py`.

    Structure otherwise mirrors `TossingRoomEnvironment` exactly, including the same
    real-per-instance vs. `ClassVar` split: layout config is genuine per-run
    configuration and so lives in constructor arguments, while the types, the fixed
    `Object`s, the skill-id encoding and the `action_space` are structural constants.
    """

    # Discrete skill ids, shared by the decode below and skills.py's compute_action
    # encode -- one source of truth for both directions. Unchanged from Tossing Room:
    # the RAW action space does not split, only the symbolic layer above it does, so a
    # `ThrowTrash` and a `ThrowRecycling` both encode to SKILL_THROW with their own
    # kind in `arg`, and `PressTrash`/`PressRecycling` both encode to SKILL_PRESS with
    # the kind of the bin they empty in `arg` (which is how the pressed button is named
    # by the action rather than inferred from the robot's position -- it was unused when
    # one button emptied both bins).
    SKILL_PICKUP: ClassVar[int] = 0
    SKILL_MOVE_ROOM: ClassVar[int] = 1
    SKILL_THROW: ClassVar[int] = 2
    SKILL_PRESS: ClassVar[int] = 3
    # Not a skill: the id a `noop_action` carries, chosen outside the real ids so
    # the decode below falls through every branch. Negative rather than 4 so that
    # adding a fifth skill can never silently turn every no-op into it.
    SKILL_NOOP: ClassVar[int] = -1

    # Item-kind discriminators (also the robot's "holding" encoding: 0 = empty hand).
    TRASH_KIND: ClassVar[int] = 1
    RECYCLING_KIND: ClassVar[int] = 2

    # A bin's capacity. Deliberately NOT configurable: each throw's bin-empty
    # precondition, the in-bin predicates' once-per-throw semantics and EMPTY's
    # one-item-per-bin prefill are all statements about capacity 1, so a knob here would
    # let a config silently reopen the vacuous-success defect this design closes.
    BIN_CAPACITY: ClassVar[int] = 1

    num_rooms: int = 7
    start_room: int = 3
    recycling_bin_room: int = 1
    trash_bin_room: int = 6
    # There is no button-room config: each bin's button sits in that bin's own room, so
    # its placement is derived (button_room_for_kind) rather than separately configured.
    # Two knobs that must agree would be a footgun -- "beside its bin" is structural.
    # The one-way ledge sits between blocked_right_from and blocked_right_from + 1:
    # stepping RIGHT from blocked_right_from is blocked; every other adjacent step
    # (including stepping LEFT back across it) is allowed.
    blocked_right_from: int = 2
    throw_tolerance: float = (
        0.1  # max |force - required| for a Throw to land, like Light Switch's 0.1
    )
    # The unobserved required-force relation, ported verbatim from
    # TossingRoomEnvironment -- see that class for why it is written around a reference
    # throw rather than as a bare intercept, why it takes TWO causes, and the three
    # constraints these numbers satisfy (required force spans exactly [0.1, 0.9] so a
    # uniformly random force lands with probability 0.2 on every task; each cause
    # contributes a 0.4-wide band against a 0.1 tolerance so neither can be ignored; the
    # state-blind ceiling stays at parity with the target_force design). The values must
    # stay in step with Tossing Room's: the split-throw experiment compares its numbers
    # against that domain's baseline, and a different relation would make them different
    # learning problems rather than the same one under two skill decompositions.
    reference_force: float = 0.5
    reference_distance: float = 2.0
    reference_weight: float = 1.0
    distance_coefficient: float = 0.2
    # 0.8, where tossingroomsplit uses 0.4 -- THE ONE CONSTANT THAT DIFFERS, and it
    # differs in order to keep the thing that matters the same. There, the two causes
    # contributed a 0.4-wide band each and the required force spanned exactly [0.1, 0.9].
    # Fixing the distance retires one of those bands, so leaving this at 0.4 would halve
    # the span to [0.3, 0.7) -- and the state-blind ceiling (the best a sampler that
    # ignores the state entirely can do) would rise from 7/16 to about 1/2, because a
    # single fixed force would cover half of a half-width range. Half the throws landing
    # without conditioning on anything would swamp what this domain is measuring.
    # Doubling the coefficient restores the [0.1, 0.9) span, and with one uniform cause
    # instead of two the required force is uniform rather than triangular, which puts the
    # state-blind ceiling at about 1/4 -- below tossingroomsplit's, not above it.
    weight_coefficient: float = 0.8
    # FIXED, for every bin and every task -- the one layout value tossingroomsplit draws
    # per task and this domain does not. Set to reference_distance, so the distance term
    # of the relation is identically zero and required_force is reference_force +
    # weight_coefficient * (weight - reference_weight). The relation itself is unchanged
    # code; only this input stops varying.
    throw_distance: float = 2.0
    # The law every pickup draws from -- Uniform[low, high), the SAME law
    # TossingRoomSplitTasks used per task, moved onto the environment because the draw is
    # now part of the dynamics rather than part of task construction. Trash and recycling
    # draw from one law and one stream: no item kind is systematically heavier, so a
    # stranded robot's weights are unidentifiable (n=1) rather than biased.
    pickup_weight_low: float = 0.5
    pickup_weight_high: float = 1.5
    # How many pickups one task's pre-sampled array covers. Sized by the CLI from the
    # run's own budget (num_cycles * max_steps_per_interaction bounds practice pickups,
    # since every pickup costs a step); the default here only has to serve direct
    # construction in tests. Running off the end RAISES -- see _next_pickup_weight.
    weight_schedule_length: int = 64
    # hard_reset's non-random per-task values; only used by the canonical reset state,
    # never by task sampling (Tasks draws its own per-task weight seeds). The canonical
    # weight is the reference throw's, so the canonical state needs exactly
    # reference_force -- and it is also the PLACEHOLDER an item carries before anything
    # has been picked up, which no throw can ever read (a throw needs a full hand).
    canonical_item_weight: float = 1.0
    # hard_reset's weight seed. Wired from --seed by the CLI, not left at a constant:
    # under --practice-reset-policy never the hard_reset state is the only one practice
    # ever sees, so a fixed seed here would give every seed in a sweep the same practice
    # weights.
    canonical_weight_seed: int = 0

    # seed -> that task's pre-sampled weights. A cache, not state: every entry is a pure
    # function of its key, so two Environment instances that never meet still agree, and
    # nothing here is part of a run's result.
    _weight_schedules: dict[int, np.ndarray] = PrivateAttr(default_factory=dict)

    robot_type: ClassVar[Type] = Type(name="robot", feature_names=("room", "holding"))
    # blocks_right marks the one-way ledge. Like the pile's room, this lives in the
    # STATE rather than in config so a module-level Predicate -- whose signature is
    # only (state, objects) -- can read it and keep MoveRoom's model as strong as
    # _apply_move's guard.
    room_type: ClassVar[Type] = Type(name="room", feature_names=("index", "blocks_right"))
    # THE SPLIT. Tossing Room has one `bin`, one `item` and one `button` type; here each
    # kind has its own, which is what stops a grounding crossing the two. Feature schemas
    # stay identical within each pair -- see the class docstring for why that matters to
    # the sampler comparison. `count` stays the single representation of a bin's
    # contents, read by BOTH the in-bin and the bin-empty predicate: with capacity 1 it
    # is already exactly the bit they each need, and a second "is an item in here"
    # feature would be the same fact stored twice.
    # throw_distance is how far this bin sits from the doorway the robot throws from,
    # redrawn per task -- one of the two observable CAUSES of the required force (the
    # other is an item's weight). Present on BOTH bin types, identically, for the same
    # sampler-width reason the rest of the schema is shared.
    trash_bin_type: ClassVar[Type] = Type(
        name="trash_bin", feature_names=("count", "room", "kind", "throw_distance")
    )
    recycling_bin_type: ClassVar[Type] = Type(
        name="recycling_bin", feature_names=("count", "room", "kind", "throw_distance")
    )
    # A button carries `kind` for the same reason a bin does: it is what `compute_action`
    # reads to name the pressed button in the raw action, and what the raw `_apply_press`
    # routes on. At the symbolic layer the type already ties a button to its bin.
    trash_button_type: ClassVar[Type] = Type(name="trash_button", feature_names=("room", "kind"))
    recycling_button_type: ClassVar[Type] = Type(
        name="recycling_button", feature_names=("room", "kind")
    )
    # The limitless item pile. Modelled as a real object with a room feature -- the
    # same shape as `button` -- so a module-level Predicate can tell which room it is
    # in. `start_room` is per-instance config, which a Predicate (whose signature is
    # only (state, objects)) cannot read; putting the pile in the STATE is what lets
    # Pickup's symbolic precondition be exactly as strong as _apply_pickup's guard.
    # weight_seed names this task's pre-sampled weight array and num_pickups is the
    # cursor into it -- both here rather than on the robot or an item because the pile is
    # the object that issues items, and because no sampler ever reads the pile (the
    # throws do not bind it, and the two pickups that do are param_dim=0). See the class
    # docstring for why they have to live in the State at all.
    pile_type: ClassVar[Type] = Type(
        name="pile", feature_names=("room", "weight_seed", "num_pickups")
    )
    # weight is the observable cause of the required force, and it is now written by
    # `_apply_pickup` rather than by task construction. It replaces `target_force`, which
    # was the answer itself sitting in each throw sampler's own input row.
    trash_type: ClassVar[Type] = Type(name="trash", feature_names=("kind", "weight"))
    recycling_type: ClassVar[Type] = Type(name="recycling", feature_names=("kind", "weight"))

    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    recycling_bin: ClassVar[Object] = Object(name="recycling_bin", type=recycling_bin_type)
    trash_bin: ClassVar[Object] = Object(name="trash_bin", type=trash_bin_type)
    trash_button: ClassVar[Object] = Object(name="trash_button", type=trash_button_type)
    recycling_button: ClassVar[Object] = Object(name="recycling_button", type=recycling_button_type)
    pile: ClassVar[Object] = Object(name="pile", type=pile_type)
    # Singleton discriminator objects, one per item type. Their kind feature is what
    # maps a held item back to the right bin/room at the raw-dynamics layer, which is
    # unchanged; at the symbolic layer the type now does that job.
    trash: ClassVar[Object] = Object(name="trash", type=trash_type)
    recycling: ClassVar[Object] = Object(name="recycling", type=recycling_type)

    action_space: ClassVar[Box] = Box(-np.inf, np.inf, (3,))

    def get_rooms(self) -> tuple[Object, ...]:
        """One Object per room, index feature = i. Built fresh every call (not cached)
        -- num_rooms can differ between instances (CLI override, test overrides), so
        caching risks a stale value; Object equality/hash are value-based (frozen
        pydantic), so rebuilding is correct, just not free (negligible at this scale)."""
        return tuple(Object(name=f"room_{i}", type=self.room_type) for i in range(self.num_rooms))

    def bin_for_kind(self, *, kind: int) -> Object:
        return self.recycling_bin if kind == self.RECYCLING_KIND else self.trash_bin

    def item_for_kind(self, *, kind: int) -> Object:
        return self.recycling if kind == self.RECYCLING_KIND else self.trash

    def bin_room_for_kind(self, *, kind: int) -> int:
        return self.recycling_bin_room if kind == self.RECYCLING_KIND else self.trash_bin_room

    def button_for_kind(self, *, kind: int) -> Object:
        return self.recycling_button if kind == self.RECYCLING_KIND else self.trash_button

    def button_room_for_kind(self, *, kind: int) -> int:
        """A button sits beside the bin it empties, so its room IS that bin's room."""
        return self.bin_room_for_kind(kind=kind)

    def required_force(self, *, throw_distance: float, item_weight: float) -> float:
        """The force a throw of `item_weight` into a bin `throw_distance` away must come
        within `throw_tolerance` of. The ONE place this relation is written down for this
        domain: both `_apply_throw` (the dynamics) and `skill_oracle_policy.py` (the
        privileged solver) call it. Deliberately not a State feature and not derivable
        from one -- see the class docstring and `TossingRoomEnvironment.required_force`.
        """
        return (
            self.reference_force
            + self.distance_coefficient * (throw_distance - self.reference_distance)
            + self.weight_coefficient * (item_weight - self.reference_weight)
        )

    def weight_schedule(self, *, weight_seed: int) -> np.ndarray:
        """This task's pre-sampled pickup weights, materialised once and cached.

        A pure function of `weight_seed` and this instance's own configuration, so the
        practice and evaluation Environments derive identical arrays without sharing
        anything -- which is what keeps `take_action` free of randomness and the
        two-instance split a genuine no-op on results.

        Public because a test and an analysis script both need to say what a run's
        weights *were*, and rederiving them by hand would be a second implementation of
        the thing under test."""
        cached = self._weight_schedules.get(weight_seed)
        if cached is None:
            cached = np.random.default_rng(weight_seed).uniform(
                self.pickup_weight_low, self.pickup_weight_high, size=self.weight_schedule_length
            )
            self._weight_schedules[weight_seed] = cached
        return cached

    def pre_sampled_seeds(self) -> frozenset[int]:
        """Which schedules have been materialised so far -- the observable that lets a
        test assert the array really is drawn by `build_initial_state` rather than lazily
        on the first pickup, i.e. inside `take_action`."""
        return frozenset(self._weight_schedules)

    def build_initial_state(
        self,
        *,
        weight_seed: int,
        recycling_count: int = 0,
        trash_count: int = 0,
    ) -> State:
        """The robot always starts in start_room with an empty hand; each bin sits in its
        configured room with its own button beside it. Only the task's weight seed and
        the initial bin counts vary between callers -- hard_reset uses canonical values
        with empty bins, Tasks draws a fresh weight seed per episode (and a prefilled item
        per bin for the empty-buckets goal).

        Both items start at `canonical_item_weight`, a PLACEHOLDER rather than a draw:
        nothing has been picked up yet, and no throw can read it (a throw needs a full
        hand, which only a pickup -- which writes the real weight -- can produce).

        This is also where the whole weight array is pre-sampled, so that `take_action`
        never consumes randomness. See the class docstring.

        A count above BIN_CAPACITY raises rather than being clamped: it is a caller bug,
        and silently accepting it would put the environment in a state each throw's
        bin-empty precondition and the in-bin predicates' once-per-throw reading both
        assume away."""
        for name, count in (("trash", trash_count), ("recycling", recycling_count)):
            if not 0 <= count <= self.BIN_CAPACITY:
                raise ValueError(
                    f"{name}_count={count}: a bin holds at most one item "
                    f"(BIN_CAPACITY={self.BIN_CAPACITY})"
                )
        # Eagerly, here, rather than on the first pickup: a lazy build would be an RNG
        # call inside take_action, which is exactly what the two-Environment split
        # forbids.
        self.weight_schedule(weight_seed=weight_seed)
        data: dict[Object, np.ndarray] = {
            self.robot: np.array([float(self.start_room), 0.0]),
            self.recycling_bin: np.array([
                float(recycling_count),
                float(self.recycling_bin_room),
                float(self.RECYCLING_KIND),
                float(self.throw_distance),
            ]),
            self.trash_bin: np.array([
                float(trash_count),
                float(self.trash_bin_room),
                float(self.TRASH_KIND),
                float(self.throw_distance),
            ]),
            self.trash_button: np.array([
                float(self.button_room_for_kind(kind=self.TRASH_KIND)),
                float(self.TRASH_KIND),
            ]),
            self.recycling_button: np.array([
                float(self.button_room_for_kind(kind=self.RECYCLING_KIND)),
                float(self.RECYCLING_KIND),
            ]),
            self.pile: np.array([float(self.start_room), float(weight_seed), 0.0]),
            self.trash: np.array([float(self.TRASH_KIND), float(self.canonical_item_weight)]),
            self.recycling: np.array([
                float(self.RECYCLING_KIND),
                float(self.canonical_item_weight),
            ]),
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
                    state=state,
                    next_state=next_state,
                    robot_room=robot_room,
                    holding=holding,
                    arg=arg,
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
                self._apply_press(next_state=next_state, robot_room=robot_room, arg=arg)
            # Any other skill_id is unknown -> no-op.

        self.set_state(state=next_state)
        return next_state

    def _apply_pickup(
        self, *, state: State, next_state: State, robot_room: int, holding: int, arg: int
    ) -> None:
        # Pickup only from the limitless pile at start_room, and only with an empty
        # hand; arg must name a real item kind.
        if (
            robot_room == self.start_room
            and holding == 0
            and arg in (self.TRASH_KIND, self.RECYCLING_KIND)
        ):
            next_state.set(obj=self.robot, feature_name="holding", feature_val=float(arg))
            # THE ONE BEHAVIOURAL DIFFERENCE FROM tossingroomsplit. The item the pile
            # just issued has a weight of its own, taken off this task's pre-sampled
            # array. Only a pickup that actually happened advances the cursor -- a
            # refused one is a silent no-op like every other out-of-context action, and
            # burning a weight on it would make the schedule two arms walk depend on how
            # many illegal actions each of them tried.
            cursor = int(round(state.get(obj=self.pile, feature_name="num_pickups")))
            weight = self._next_pickup_weight(state=state, cursor=cursor)
            next_state.set(
                obj=self.item_for_kind(kind=arg), feature_name="weight", feature_val=weight
            )
            next_state.set(obj=self.pile, feature_name="num_pickups", feature_val=float(cursor + 1))

    def _next_pickup_weight(self, *, state: State, cursor: int) -> float:
        """The `cursor`-th weight of the schedule this state's pile names.

        Raises rather than wrapping when the array runs out. That is deliberate and
        load-bearing: a wrapped schedule makes the training distribution periodic, and a
        sampler refitting forever on the same handful of weights looks exactly like one
        that has converged. `weight_schedule_length` is sized from the run's own step
        budget by the CLI, so reaching this is a configuration bug and should stop the
        run rather than quietly change what is being measured."""
        weight_seed = int(round(state.get(obj=self.pile, feature_name="weight_seed")))
        schedule = self.weight_schedule(weight_seed=weight_seed)
        if cursor >= len(schedule):
            raise RuntimeError(
                f"pickup {cursor} ran off the end of this task's pre-sampled weight "
                f"schedule (weight_schedule_length={len(schedule)}, weight_seed="
                f"{weight_seed}). The schedule never wraps -- a periodic training "
                "distribution would be indistinguishable from a converged sampler. Size "
                "--weight-schedule-length to at least the run's own pickup budget."
            )
        return float(schedule[cursor])

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
        count = state.get(obj=bin_obj, feature_name="count")
        # Capacity 1: a full bin REFUSES the throw outright -- the item stays in hand and
        # nothing happens, exactly like every other out-of-context action here. Refusing
        # rather than swallowing the item is what makes each throw's bin-empty
        # precondition exactly as strong as this guard, and it is why the count is
        # provably 0 below, so a landed throw flips the in-bin atom false -> true instead
        # of finding it already true. (A refusal is not a miss: nothing was consumed, so
        # this is not the free re-roll the release below exists to close. Emptying the
        # bin first costs a press, which the planner has to schedule.)
        if count >= self.BIN_CAPACITY:
            return
        # The two observable causes, read out of the State and combined by the
        # environment's own (unobservable) relation. Nothing in the State equals this.
        required = self.required_force(
            throw_distance=float(state.get(obj=bin_obj, feature_name="throw_distance")),
            item_weight=float(state.get(obj=item_obj, feature_name="weight")),
        )
        # Throwing always releases the item, whether or not it lands. It lands only in
        # the item's own bin room and only when the dial is within tolerance of the force
        # that bin/item pair requires. The release is what makes a miss cost something,
        # and it is what gives this domain its 1-attempt-per-period recycling budget: the
        # thrown item is gone (items carry only (kind, weight), with no position, so
        # "lying near the bin" is not representable), so trying again means a fresh item
        # from the pile -- an 8-step round trip for trash, and impossible for recycling,
        # since the one-way ledge has already closed behind the robot. Ported unchanged
        # from TossingRoomEnvironment._apply_throw; see that file for the history.
        next_state.set(obj=self.robot, feature_name="holding", feature_val=0.0)
        if robot_room == bin_room and abs(raw_force - required) < self.throw_tolerance:
            next_state.set(obj=bin_obj, feature_name="count", feature_val=count + 1.0)

    def _apply_press(self, *, next_state: State, robot_room: int, arg: int) -> None:
        """Empty the bin belonging to the button named by `arg`, and ONLY that bin --
        each bin has its own button, beside it. Naming the button in the action rather
        than inferring it from the room keeps this unambiguous under any layout,
        including a degenerate one that puts both bins in the same room."""
        if arg not in (self.TRASH_KIND, self.RECYCLING_KIND):
            return
        if robot_room == self.button_room_for_kind(kind=arg):
            next_state.set(obj=self.bin_for_kind(kind=arg), feature_name="count", feature_val=0.0)

    def get_valid_actions(self) -> list[Action]:
        # The force dimension is continuous and unbounded (matches Light Switch's
        # convention), so there is no finite/enumerable action list to return.
        return []

    def noop_action(self) -> Action:
        """`SKILL_NOOP` in slot 0, which `take_action`'s decode falls through to its
        "any other skill_id is unknown -> no-op" arm.

        Deliberately not a zero vector, even though zeros *is* inert here today: slot
        0 is the skill id and `SKILL_PICKUP == 0`, so zeros decodes as a real `Pickup`
        and survives only because its item-kind argument rounds to 0, which names no
        kind. That is a coincidence of a second field rather than a property of the
        no-op -- anything keying on the skill id already mislabels it, and giving
        `Pickup` a zero-valid argument would turn every no-op into a real pickup."""
        return np.array([float(self.SKILL_NOOP), 0.0, 0.0])

    def hard_reset(self) -> None:
        """Under `--practice-reset-policy never` this is the ONLY state practice ever
        gets, so `canonical_weight_seed` -- not a task's seed -- is what drives every
        practice weight for the whole run. It is wired from `--seed` by the CLI for
        exactly that reason. The cursor starts at 0 and, with no reset to rewind it,
        simply marches through this one array: the robot keeps meeting fresh weights
        even though it never meets a fresh task."""
        self.set_state(state=self.build_initial_state(weight_seed=self.canonical_weight_seed))
