from typing import ClassVar

import numpy as np

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, State

from .environment import TossingRoomEnvironment
from .predicates import (
    BIN_ACCEPTS_ITEM,
    BIN_EMPTY,
    BIN_IN_ROOM,
    BUTTON_FOR_BIN,
    BUTTON_IN_ROOM,
    CAN_MOVE_ROOM,
    HAND_EMPTY,
    HOLDING,
    HOLDING_RECYCLING,
    HOLDING_TRASH,
    ITEM_IN_BIN,
    PILE_IN_ROOM,
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


class TossingRoomSkills:
    """Lifted skill templates for Tossing Room (split throws) -- **the default arm**.

    `TossingRoomUnsplitSkills`, at the bottom of this file, is the other arm, selected by
    `TossingRoomEnvironment.unsplit_skills` (`--unsplit-skills`). Read this class first:
    the unsplit one is defined against it.

    **The one difference from `TossingRoomUnsplitSkills`, and the reason the split
    exists**: there is no `Throw`. There are `ThrowTrash` and `ThrowRecycling`, two
    `Skill`s with different `name`s, each binding its own item and bin.

    Why the *name* is the load-bearing thing: `EesMethod.sampler` keys its
    `LearnedSkillSampler` dict by `skill_name`, so two names mean two classifiers with
    the same architecture and independent weights. Under the unsplit arm the single
    `Throw` name pools both kinds' training rows into one classifier, and since
    `EesMethod.state_features` is `concat(state[obj] for obj in ground_skill.objects)`
    -- which for an item is `(kind, weight)` -- that classifier can and does transfer
    trash experience to recycling. Splitting the names removes the transfer.
    Competence is unaffected either way: `EesMethod.competence_model` is already keyed
    per *ground* skill, and `Throw(trash, ...)` and `Throw(recycling, ...)` were already
    two different groundings.

    Why *types* rather than a precondition: a shared `item_type` plus an
    `IsTrash(item)`-style precondition would still leave an `item` variable ranging over
    both kinds, and a later edit dropping the precondition would silently restore the
    shared binding with nothing to catch it. Distinct types are checked by
    `SkillGrounder._applicable_groundings` (exact `obj.type == variable.type`), by
    `GroundSkill`'s own validator, and by the PDDL handed to Fast Downward.

    Consequences, all deliberate:
      * `Pickup` splits too (`PickupTrash`/`PickupRecycling`) -- a `trash` object cannot
        bind a parameter typed `recycling`. Both are `param_dim=0`, so neither gets a
        sampler and nothing about learning changes; the split is representational.
      * `Press` splits too (`PressTrash`/`PressRecycling`), for the same reason applied
        to the bin and button types. The unsplit `Press` is already per-bin -- one lifted
        skill whose `?bin`/`?button` are pinned to a single pairing by
        `ButtonForBin`/`BinAcceptsItem`, so it has exactly two groundings. Here that
        is two lifted skills with one grounding each, which is the same relationship the
        throws have. `param_dim=0` for both, so again no sampler and no learning change.
      * `BinAcceptsItem` and `ButtonForBin` are **dropped**. Each exists on the unsplit
        arm only to stop a single lifted skill binding a mismatched object; here the types
        make both tautologies. See `predicates.py`.

    Everything else is the unsplit layer verbatim: `MoveRoom`'s `CanMoveRoom`
    precondition (not symmetric `Adjacent`, because `_apply_move` refuses the rightward
    ledge step), `Pickup`'s `PileInRoom` precondition (without which Fast Downward emits
    plans that walk past the pile and pick up in the bin room -- a silent no-op that
    solved 1/10 tasks; see docs/experiment-logs/2026-08-02-tossingroom-pickup-
    precondition.md), and the capacity-1 bin model: each throw requires its bin to be
    empty (because `_apply_throw` refuses a throw at a full bin) and deletes that atom
    (because a landed throw fills it). Neither press carries `ignore_effects`: one button
    per bin makes the in-bin delete an ordinary per-bin `delete_effect`.

    A static-method container, never instantiated, same as every other business-logic
    class in this project.
    """

    _robot: ClassVar[Variable] = Variable(name="robot", type=TossingRoomEnvironment.robot_type)
    _trash: ClassVar[Variable] = Variable(name="trash", type=TossingRoomEnvironment.trash_type)
    _recycling: ClassVar[Variable] = Variable(
        name="recycling", type=TossingRoomEnvironment.recycling_type
    )
    _room: ClassVar[Variable] = Variable(name="room", type=TossingRoomEnvironment.room_type)
    _from_room: ClassVar[Variable] = Variable(
        name="from_room", type=TossingRoomEnvironment.room_type
    )
    _to_room: ClassVar[Variable] = Variable(name="to_room", type=TossingRoomEnvironment.room_type)
    _trash_bin: ClassVar[Variable] = Variable(
        name="trash_bin", type=TossingRoomEnvironment.trash_bin_type
    )
    _recycling_bin: ClassVar[Variable] = Variable(
        name="recycling_bin", type=TossingRoomEnvironment.recycling_bin_type
    )
    _trash_button: ClassVar[Variable] = Variable(
        name="trash_button", type=TossingRoomEnvironment.trash_button_type
    )
    _recycling_button: ClassVar[Variable] = Variable(
        name="recycling_button", type=TossingRoomEnvironment.recycling_button_type
    )
    _pile: ClassVar[Variable] = Variable(name="pile", type=TossingRoomEnvironment.pile_type)

    PICKUP_TRASH: ClassVar[Skill] = Skill(
        name="PickupTrash",
        parameters=(_robot, _trash, _room, _pile),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(_pile, _room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING_TRASH, variables=(_robot, _trash))}),
        delete_effects=frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))}),
        param_dim=0,
    )
    PICKUP_RECYCLING: ClassVar[Skill] = Skill(
        name="PickupRecycling",
        parameters=(_robot, _recycling, _room, _pile),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(_pile, _room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=HOLDING_RECYCLING, variables=(_robot, _recycling))
        }),
        delete_effects=frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))}),
        param_dim=0,
    )
    MOVE_ROOM: ClassVar[Skill] = Skill(
        name="MoveRoom",
        parameters=(_robot, _from_room, _to_room),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _from_room)),
            # CanMoveRoom, not Adjacent: adjacency is symmetric but _apply_move
            # refuses the rightward step across the one-way ledge.
            LiftedAtom(predicate=CAN_MOVE_ROOM, variables=(_from_room, _to_room)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _to_room))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _from_room))
        }),
        param_dim=0,
    )
    THROW_TRASH: ClassVar[Skill] = Skill(
        name="ThrowTrash",
        parameters=(_robot, _trash, _trash_bin, _room),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING_TRASH, variables=(_robot, _trash)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=TRASH_BIN_IN_ROOM, variables=(_trash_bin, _room)),
            # No BinAcceptsItem: `_trash_bin` is typed `trash_bin`, so a mismatched bin
            # is not merely disallowed, it is unbindable. See predicates.py.
            #
            # A bin holds at most one item and _apply_throw REFUSES a throw at a full
            # one, so the model has to say so. This is also what makes TrashInBin an
            # honest add_effect: with the count pinned to 0 beforehand it goes
            # false -> true exactly once per throw, instead of already being true
            # because of an earlier throw's item (which EES scored as this throw's
            # success at any force).
            LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(_trash_bin,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=TRASH_IN_BIN, variables=(_trash, _trash_bin)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING_TRASH, variables=(_robot, _trash)),
            # A landed throw fills the bin. Omitting this would leave the planner
            # believing it can throw a second item into the same bin -- the same
            # model-above-reality shape as the defects this domain has already shipped.
            LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(_trash_bin,)),
        }),
        param_dim=1,
    )
    THROW_RECYCLING: ClassVar[Skill] = Skill(
        name="ThrowRecycling",
        parameters=(_robot, _recycling, _recycling_bin, _room),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING_RECYCLING, variables=(_robot, _recycling)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=RECYCLING_BIN_IN_ROOM, variables=(_recycling_bin, _room)),
            # Capacity 1, exactly as for ThrowTrash above.
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(_recycling_bin,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=RECYCLING_IN_BIN, variables=(_recycling, _recycling_bin)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING_RECYCLING, variables=(_robot, _recycling)),
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(_recycling_bin,)),
        }),
        param_dim=1,
    )
    # One press per bin. No ignore_effects on either, deliberately: the old single button
    # emptied BOTH bins, so every in-bin atom became false at once -- a universal delete
    # no per-item delete_effect could express, since that Press had no item parameter.
    # One button per bin makes the delete ordinary, and each skill's in-bin atom is the
    # only one its press can falsify.
    PRESS_TRASH: ClassVar[Skill] = Skill(
        name="PressTrash",
        parameters=(_robot, _trash_button, _room, _trash_bin, _trash),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=TRASH_BUTTON_IN_ROOM, variables=(_trash_button, _room)),
            # No ButtonForBin/BinAcceptsItem: `_trash_button`/`_trash_bin`/`_trash` are
            # typed per kind with one object each, so both pairings are unbindable the
            # wrong way round. See predicates.py.
        }),
        add_effects=frozenset({LiftedAtom(predicate=TRASH_BIN_EMPTY, variables=(_trash_bin,))}),
        delete_effects=frozenset({
            LiftedAtom(predicate=TRASH_IN_BIN, variables=(_trash, _trash_bin))
        }),
        param_dim=0,
    )
    PRESS_RECYCLING: ClassVar[Skill] = Skill(
        name="PressRecycling",
        parameters=(_robot, _recycling_button, _room, _recycling_bin, _recycling),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=RECYCLING_BUTTON_IN_ROOM, variables=(_recycling_button, _room)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=RECYCLING_BIN_EMPTY, variables=(_recycling_bin,))
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=RECYCLING_IN_BIN, variables=(_recycling, _recycling_bin))
        }),
        param_dim=0,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """Uniform(0, 1) per continuous dim -- only the two throws have one (the force).

        Deliberately the SAME base distribution for both throws, and the same one
        Tossing Room uses. The experiment compares how far each learned sampler moves
        away from this prior; giving one kind a head start here would be exactly the
        confound to avoid. The range is a plausible force band but is not concentrated on
        the force any particular task requires, so a random draw misses the
        throw_tolerance window with probability 0.8 on every task -- and there is no
        single force that works everywhere, since the requirement is an unobserved
        function of the bin's throw_distance and the item's weight. The oracle bypasses
        this entirely via `TossingRoomEnvironment.required_force`."""
        return rng.uniform(0.0, 1.0, size=ground_skill.skill.param_dim)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The lifted "option policy" layer: reads the bound objects (and, for a throw,
        the sampled force param) to produce one raw `[skill_id, arg, force]` action.
        Dispatches by Skill value equality (not identity), so a Method that reconstructs
        an equal-content Skill still works.

        Both pickups, both throws and both presses encode to the same raw skill id -- the
        RAW action space is unchanged from Tossing Room (see
        `TossingRoomEnvironment`); the kind is carried in `arg` and
        read off a bound object's own `kind` feature, so the two branches of each pair
        differ only in which object they read."""
        skills = TossingRoomSkills
        env = TossingRoomEnvironment
        skill = ground_skill.skill

        if skill in (skills.PICKUP_TRASH, skills.PICKUP_RECYCLING):
            _robot, item, _room, _pile = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_PICKUP), kind, 0.0])

        if skill == skills.MOVE_ROOM:
            _robot, _from_room, to_room = ground_skill.objects
            to_index = state.get(obj=to_room, feature_name="index")
            return np.array([float(env.SKILL_MOVE_ROOM), to_index, 0.0])

        if skill in (skills.THROW_TRASH, skills.THROW_RECYCLING):
            _robot, item, _bin, _room = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_THROW), kind, float(params[0])])

        if skill in (skills.PRESS_TRASH, skills.PRESS_RECYCLING):
            # arg names WHICH button: each bin has its own, and pressing empties only
            # that bin. Read off the bound button rather than the bin, so the action
            # says what was pressed.
            _robot, button, _room, _bin, _item = ground_skill.objects
            kind = state.get(obj=button, feature_name="kind")
            return np.array([float(env.SKILL_PRESS), kind, 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")


class TossingRoomUnsplitSkills:
    """The other arm: the four lifted skills `TossingRoomEnvironment.unsplit_skills`
    selects, where `TossingRoomSkills` above has seven.

    **`Throw` is one skill, and its `?item` ranges over both kinds** -- which is the whole
    point, because `EesMethod.sampler` keys its `LearnedSkillSampler` dict by skill NAME,
    so one name is one classifier fed by both kinds' rows. That is the trash -> recycling
    transfer channel the split removes, made switchable rather than forked into a fifth
    copy of the domain.

    **Why `Pickup` and `Press` collapse too, rather than only `Throw`.** `Type` carries no
    hierarchy, and `SkillGrounder._applicable_groundings` and `GroundSkill`'s validator
    both demand exact `obj.type == variable.type`. So an `?item` ranging over both kinds
    is only expressible if both item objects carry ONE type -- and a single lifted throw
    has a single `?bin`, which forces the same on the bins (dropping the bin parameter
    instead is not an option: the bin's `throw_distance` is part of the sampler's input
    row, and without a named bin neither the capacity-1 precondition nor the in-bin add
    effect is expressible). Once the item and bin types are shared, a `PickupTrash` typed
    over the shared item would ground with recycling as well, and a `PressTrash` with the
    recycling bin: the per-kind versions stop being *representable*, they are not merely
    redundant. The button type follows the bins for the same reason `Press` names both.

    **What comes back with them.** `BinAcceptsItem` and `ButtonForBin`, dropped from the
    split layer as tautologies, are live constraints here -- they are all that stops
    `Throw(trash -> recycling_bin)` (well-typed now, and refusable by `_apply_throw` at
    any force) and a press deleting the other bin's in-bin atom. `Press` still carries no
    `ignore_effects`: one button per bin keeps its delete an ordinary per-bin effect, with
    `?bin` pinned by `ButtonForBin` and `?item` by `BinAcceptsItem`.

    `MoveRoom` is *the same object* as the split layer's, not a copy: its parameters are
    typed `robot`/`room`, neither of which splits, so there is nothing for the flag to
    change and two definitions could only drift.

    Everything else -- `Pickup`'s `PileInRoom` precondition, the capacity-1 bin-empty
    precondition and delete on `Throw`, the `Uniform(0, 1)` force prior -- is
    `TossingRoomSkills` verbatim, deliberately: the two arms must differ in the skill
    decomposition and in nothing else.

    A static-method container, never instantiated, same as every other business-logic
    class in this project.
    """

    _robot: ClassVar[Variable] = Variable(name="robot", type=TossingRoomEnvironment.robot_type)
    _item: ClassVar[Variable] = Variable(name="item", type=TossingRoomEnvironment.item_type)
    _room: ClassVar[Variable] = Variable(name="room", type=TossingRoomEnvironment.room_type)
    _bin: ClassVar[Variable] = Variable(name="bin", type=TossingRoomEnvironment.bin_type)
    _button: ClassVar[Variable] = Variable(name="button", type=TossingRoomEnvironment.button_type)
    _pile: ClassVar[Variable] = Variable(name="pile", type=TossingRoomEnvironment.pile_type)

    PICKUP: ClassVar[Skill] = Skill(
        name="Pickup",
        parameters=(_robot, _item, _room, _pile),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            # Without this the model permits picking up ANYWHERE while _apply_pickup only
            # acts in the pile room, so the planner emitted plans that walk past the pile
            # and pick up in the bin room -- a silent no-op that solved 1/10 tasks. See
            # docs/experiment-logs/2026-08-02-tossingroom-pickup-precondition.md.
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(_pile, _room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _item))}),
        delete_effects=frozenset({LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,))}),
        param_dim=0,
    )
    # Shared outright with the split layer -- see the class docstring.
    MOVE_ROOM: ClassVar[Skill] = TossingRoomSkills.MOVE_ROOM
    THROW: ClassVar[Skill] = Skill(
        name="Throw",
        parameters=(_robot, _item, _bin, _room),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _item)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=BIN_IN_ROOM, variables=(_bin, _room)),
            # _apply_throw routes by the HELD item's kind and ignores the bound bin, so a
            # mismatched bin can never succeed at any force. With one bin type that
            # grounding is well-typed, so it has to be excluded here rather than by the
            # grounder -- this is exactly the constraint the split arm gets for free.
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(_item, _bin)),
            # A bin holds at most one item and _apply_throw REFUSES a throw at a full one,
            # so the model has to say so. It is also what makes ItemInBin an honest
            # add_effect: with the count pinned to 0 beforehand it goes false -> true
            # exactly once per throw.
            LiftedAtom(predicate=BIN_EMPTY, variables=(_bin,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=ITEM_IN_BIN, variables=(_item, _bin)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _item)),
            # A landed throw fills the bin. Omitting this would leave the planner
            # believing it can throw a second item into the same bin.
            LiftedAtom(predicate=BIN_EMPTY, variables=(_bin,)),
        }),
        param_dim=1,
    )
    PRESS: ClassVar[Skill] = Skill(
        name="Press",
        parameters=(_robot, _button, _room, _bin, _item),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=BUTTON_IN_ROOM, variables=(_button, _room)),
            # Each bin has its own button beside it and _apply_press empties only that
            # one, so ButtonForBin pins ?bin to the pressed button...
            LiftedAtom(predicate=BUTTON_FOR_BIN, variables=(_button, _bin)),
            # ...and BinAcceptsItem pins ?item to that bin, giving exactly one valid
            # grounding per button. Without a named item the ItemInBin delete below is not
            # expressible at all.
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(_item, _bin)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=BIN_EMPTY, variables=(_bin,))}),
        delete_effects=frozenset({LiftedAtom(predicate=ITEM_IN_BIN, variables=(_item, _bin))}),
        param_dim=0,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """The same Uniform(0, 1) prior the split throws draw from, and the same code
        path. The arms must differ in how experience is POOLED, not in what the untrained
        sampler proposes."""
        return TossingRoomSkills.sample_params(ground_skill=ground_skill, rng=rng)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The same raw encoding the split layer produces, from the same bound objects'
        `kind` features. The RAW action space does not change with the flag -- only which
        lifted skill named the action does -- so a trash throw is the identical
        `[SKILL_THROW, TRASH_KIND, force]` vector under either arm."""
        skills = TossingRoomUnsplitSkills
        env = TossingRoomEnvironment
        skill = ground_skill.skill

        if skill == skills.PICKUP:
            _robot, item, _room, _pile = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_PICKUP), kind, 0.0])

        if skill == skills.MOVE_ROOM:
            return TossingRoomSkills.compute_action(
                ground_skill=ground_skill, params=params, state=state
            )

        if skill == skills.THROW:
            _robot, item, _bin, _room = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_THROW), kind, float(params[0])])

        if skill == skills.PRESS:
            # arg names WHICH button, read off the bound button rather than the bin, so
            # the action says what was pressed.
            _robot, button, _room, _bin, _item = ground_skill.objects
            kind = state.get(obj=button, feature_name="kind")
            return np.array([float(env.SKILL_PRESS), kind, 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")
