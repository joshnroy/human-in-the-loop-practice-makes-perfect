"""What the real backend remembers of a pick that failed without touching the cube.

The robot side's `can_pick` is a dry run of the planned hover and grasp. It cannot see a
pick its own wrist servo then refuses to finish, so it goes on saying feasible while every
attempt fails the same way (2026-11-11: the cube near a wall of the close bin, the first
1 cm servo step refused because the gripper would touch the wall). That is the kitchen's
`KitchenPickPlannable` case on the real robot: an observed refusal is information.
`PickPlannable` is false while the cube and the bin stay where the failed pick left them,
and true again as soon as either is seen to have moved -- a pour, a person, a nudge.

Nothing here chooses for the planner: no cost, no mask, no forced skill.
"""

import math
from typing import Final

from pydantic import BaseModel, ConfigDict

from .contract import BinReply, ObserveReply, PickReply

# The gripper's reading at or under which it never closed. Open reads 0.009 and a held
# cube 0.47-0.52 (28/28 picks of the 2026-11-11 run); a gripper that closed on nothing
# reads past the held band.
GRIPPER_NEVER_CLOSED_READING: Final = 0.1
# The cube has moved when a look finds it this far from where the failed pick left it.
# The ceiling cameras repeat an unmoved cube within 1 mm, and 1 cm is the wrist servo's
# own smallest step: a shift under it leaves the servo the step it was refused.
CUBE_MOVED_M: Final = 0.01
# The bin's fit repeats within 1 mm and 0.4 degrees between looks.
BIN_MOVED_M: Final = 0.01
BIN_TURNED_RAD: Final = math.radians(2.0)


class RealPickRefusal(BaseModel):
    """A pick that never closed on the cube, and the scene the look after it found."""

    model_config = ConfigDict(frozen=True)

    # Map frame. The cube's yaw is left out: its fit turns by several degrees between
    # looks at an unmoved cube (0.151 -> 0.016 rad in the incident).
    cube: tuple[float, float]
    bin: BinReply | None
    note: str | None


class RealPickRefusalMemory:
    """When a failed pick is remembered, and when a look forgets it. A static-method
    container."""

    @staticmethod
    def never_closed(*, pick: PickReply) -> bool:
        """The pick moved, did not hold, and its gripper reads open: it never closed.
        A pick refused before it moved is `can_pick`'s own verdict, not remembered."""
        return not pick.held and not pick.refused and pick.gripper <= GRIPPER_NEVER_CLOSED_READING

    @staticmethod
    def recorded(
        *, before: ObserveReply, pick: PickReply, after: ObserveReply
    ) -> RealPickRefusal | None:
        """The refusal a failed `pick` leaves, between the looks `before` and `after` it,
        or None: the gripper closed (a miss or a slip stays retryable), or the pick moved
        the cube or the bin (a new scene, a new attempt)."""
        if not RealPickRefusalMemory.never_closed(pick=pick):
            return None
        if before.cube is None or before.holding:
            return None
        was = RealPickRefusal(cube=(before.cube[0], before.cube[1]), bin=before.bin, note=None)
        if not RealPickRefusalMemory.stands(refusal=was, reply=after):
            return None
        assert after.cube is not None
        return RealPickRefusal(cube=(after.cube[0], after.cube[1]), bin=after.bin, note=pick.note)

    @staticmethod
    def stands(*, refusal: RealPickRefusal, reply: ObserveReply) -> bool:
        """Whether the look `reply` finds the cube and the bin where `refusal` left them.
        A cube that is held or out of sight is not known to be there."""
        if reply.holding or reply.cube is None:
            return False
        if math.dist(refusal.cube, reply.cube[:2]) > CUBE_MOVED_M:
            return False
        if refusal.bin is None or reply.bin is None:
            return refusal.bin is None and reply.bin is None
        if math.dist((refusal.bin.x, refusal.bin.y), (reply.bin.x, reply.bin.y)) > BIN_MOVED_M:
            return False
        # The fit's yaw is its long side's: the same bin at yaw and yaw + pi.
        turn = (reply.bin.yaw - refusal.bin.yaw + math.pi / 2) % math.pi - math.pi / 2
        return abs(turn) <= BIN_TURNED_RAD
