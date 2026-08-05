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
    ITEM_IN_BIN,
    PILE_IN_ROOM,
    ROBOT_IN_ROOM,
)


class TossingRoomSkills:
    """Lifted skill templates for Tossing Room -- preconditions/add_effects/
    delete_effects are LiftedAtoms over each skill's own Variables, mirroring
    Light Switch's skills.py (and predicators' NSRT shape) so a future planning-based
    Method could task-plan over them. A static-method container, never instantiated,
    same as every other business-logic class in this project.

    These models are kept exactly as strong as the raw dynamics, no more. They were
    once deliberately more permissive -- Pickup said only "robot in some room + hand
    empty", justified by "no planner consumes these yet" -- and that justification
    expired the moment EES task-planned over them: Fast Downward emitted plans that
    walked past the pile and picked up in the bin room, a silent no-op, solving 1/10
    tasks. Pickup now requires PileInRoom (the pile is a state object precisely so a
    module-level Predicate can read it). See
    docs/experiment-logs/2026-08-02-tossingroom-pickup-precondition.md.

    Two of the models below encode the capacity-1 bin redesign, and both are exactly as
    strong as the dynamics they mirror: Throw requires BinEmpty (because _apply_throw
    refuses a throw at a full bin) and deletes it (because a landed throw fills it),
    and Press is per-bin -- ?bin pinned by ButtonForBin, ?item by BinAcceptsItem -- which
    is what let the old shared button's ignore_effects={ItemInBin} go away entirely.

    Light Switch's JumpToLight is NOT a precedent for weakening this: it is a
    deliberate trap whose option provably cannot achieve its effect, which EES is meant
    to discover. An operator that claims applicability the dynamics deny is a different
    thing -- it corrupts planning rather than being learnable.
    """

    _robot: ClassVar[Variable] = Variable(name="robot", type=TossingRoomEnvironment.robot_type)
    _item: ClassVar[Variable] = Variable(name="item", type=TossingRoomEnvironment.item_type)
    _room: ClassVar[Variable] = Variable(name="room", type=TossingRoomEnvironment.room_type)
    _from_room: ClassVar[Variable] = Variable(
        name="from_room", type=TossingRoomEnvironment.room_type
    )
    _to_room: ClassVar[Variable] = Variable(name="to_room", type=TossingRoomEnvironment.room_type)
    _bin: ClassVar[Variable] = Variable(name="bin", type=TossingRoomEnvironment.bin_type)
    _button: ClassVar[Variable] = Variable(name="button", type=TossingRoomEnvironment.button_type)
    _pile: ClassVar[Variable] = Variable(name="pile", type=TossingRoomEnvironment.pile_type)

    PICKUP: ClassVar[Skill] = Skill(
        name="Pickup",
        parameters=(_robot, _item, _room, _pile),
        preconditions=frozenset({
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            # Without this the model permits picking up ANYWHERE while
            # _apply_pickup only acts in the pile room, so the planner emitted
            # plans that walk past the pile and pick up in the bin room -- a
            # silent no-op. See TestPickupIsRestrictedToThePileRoom.
            LiftedAtom(predicate=PILE_IN_ROOM, variables=(_pile, _room)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=HOLDING, variables=(_robot, _item))}),
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
    THROW: ClassVar[Skill] = Skill(
        name="Throw",
        parameters=(_robot, _item, _bin, _room),
        preconditions=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _item)),
            LiftedAtom(predicate=ROBOT_IN_ROOM, variables=(_robot, _room)),
            LiftedAtom(predicate=BIN_IN_ROOM, variables=(_bin, _room)),
            # _apply_throw routes by the HELD item's kind and ignores the bound
            # bin, so a mismatched bin can never succeed at any force.
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(_item, _bin)),
            # A bin holds at most one item and _apply_throw REFUSES a throw at a full
            # one, so the model has to say so. This is also what makes ItemInBin an
            # honest add_effect: with the count pinned to 0 beforehand it goes
            # false -> true exactly once per throw, instead of already being true
            # because of somebody else's item (which EES scored as this throw's
            # success at any force).
            LiftedAtom(predicate=BIN_EMPTY, variables=(_bin,)),
        }),
        add_effects=frozenset({
            LiftedAtom(predicate=ITEM_IN_BIN, variables=(_item, _bin)),
            LiftedAtom(predicate=HAND_EMPTY, variables=(_robot,)),
        }),
        delete_effects=frozenset({
            LiftedAtom(predicate=HOLDING, variables=(_robot, _item)),
            # A landed throw fills the bin. Omitting this would leave the planner
            # believing it can throw a second item into the same bin -- the same
            # model-above-reality shape as the defects this domain has already shipped.
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
            # Each bin has its own button beside it, and _apply_press empties only that
            # one -- ButtonForBin pins ?bin to the pressed button.
            LiftedAtom(predicate=BUTTON_FOR_BIN, variables=(_button, _bin)),
            # ...and BinAcceptsItem pins ?item to that bin, giving exactly one valid
            # grounding per button rather than widening what Press applies to. Without a
            # named item the ItemInBin delete below is not expressible.
            LiftedAtom(predicate=BIN_ACCEPTS_ITEM, variables=(_item, _bin)),
        }),
        add_effects=frozenset({LiftedAtom(predicate=BIN_EMPTY, variables=(_bin,))}),
        delete_effects=frozenset({LiftedAtom(predicate=ITEM_IN_BIN, variables=(_item, _bin))}),
        # No ignore_effects, deliberately. The old single button emptied BOTH bins, so
        # every ItemInBin became false at once -- a universal delete no per-item
        # delete_effect could express, since that Press had no item parameter. One
        # button per bin makes the delete ordinary: ItemInBin(?item, ?bin) is the only
        # atom a press can falsify (a mismatched-kind ItemInBin is false in every
        # state), so the blanket ignore_effects -- which also told the planner that
        # emptying one bin empties the other -- is no longer needed and would now be
        # strictly weaker than the truth.
        param_dim=0,
    )

    @staticmethod
    def sample_params(*, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        """Uniform(0, 1) per continuous dim -- only Throw has one (the force). The
        range is a plausible force band but is NOT concentrated on any item's target,
        so a random draw usually misses the throw_tolerance window: that's the point,
        leaving the throw skill something a future learning Method could specialize.
        The oracle bypasses this entirely, setting force to the known target."""
        return rng.uniform(0.0, 1.0, size=ground_skill.skill.param_dim)

    @staticmethod
    def compute_action(*, ground_skill: GroundSkill, params: np.ndarray, state: State) -> Action:
        """The lifted "option policy" layer: reads the bound objects (and, for Throw,
        the sampled force param) to produce one raw [skill_id, arg, force] action --
        mirrors Light Switch's compute_action. Dispatches by Skill value equality (not
        identity), so a Method that reconstructs an equal-content Skill still works."""
        skills = TossingRoomSkills
        env = TossingRoomEnvironment
        skill = ground_skill.skill

        if skill == skills.PICKUP:
            _robot, item, _room, _pile = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_PICKUP), kind, 0.0])

        if skill == skills.MOVE_ROOM:
            _robot, _from_room, to_room = ground_skill.objects
            to_index = state.get(obj=to_room, feature_name="index")
            return np.array([float(env.SKILL_MOVE_ROOM), to_index, 0.0])

        if skill == skills.THROW:
            _robot, item, _bin, _room = ground_skill.objects
            kind = state.get(obj=item, feature_name="kind")
            return np.array([float(env.SKILL_THROW), kind, float(params[0])])

        if skill == skills.PRESS:
            # arg names WHICH button: each bin has its own, and pressing empties only
            # that bin. Read off the bound button rather than the bin, so the action
            # says what was pressed.
            _robot, button, _room, _bin, _item = ground_skill.objects
            kind = state.get(obj=button, feature_name="kind")
            return np.array([float(env.SKILL_PRESS), kind, 0.0])

        raise ValueError(f"Unknown skill: {skill.name}")
