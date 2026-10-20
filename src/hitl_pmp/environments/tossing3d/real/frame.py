"""The real room in the domain's frame: the map <-> domain conversion and the named spots.

The robot reports everything in its map frame. The sim's predicates read sides off one
axis -- an object is on the robot's side when its `x` is on the robot's side of
`barrier.x` -- so the domain frame is the one in which the tape is a line of constant
`x`: origin at the tape's centre line, `+x` across the tape away from the robot, `+y`
`+x` turned a quarter turn anticlockwise. The robot works at domain `x < 0`, as in the
sim. This module is the only place the two frames meet; the robot side owns all other
geometry (walls, drivable box, reach) and is asked, never re-derived.

Room facts (2026-10-20): the tape's robot-side edge is `y = 0.349 + 0.0267 x`, 0.05 m
thick, with the robot on map `-y`; the far-side bin spot is the box x 0.03..0.45,
y 1.24..1.56 (centre ~(0.24, 1.40)); the close-side spot is centred at ~(-0.30, -0.78)
with the bin's long side along y.
"""

import math
from typing import Final

from pydantic import BaseModel, ConfigDict

from hitl_pmp.environments.tossing3d.sides import Tossing3DSide


class RealRoomFrame(BaseModel):
    """A rigid map <-> domain transform, fixed by the tape."""

    model_config = ConfigDict(frozen=True)

    # The tape's centre line passes through here (map frame, metres).
    barrier_center: tuple[float, float]
    # Unit map-frame direction from the tape into the robot's side.
    robot_side: tuple[float, float]

    def model_post_init(self, __context: object) -> None:
        if not math.isclose(math.hypot(*self.robot_side), 1.0, abs_tol=1e-4):
            raise ValueError(f"robot_side must be a unit vector, got {self.robot_side}")

    def _axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        norm = math.hypot(*self.robot_side)
        ex = (-self.robot_side[0] / norm, -self.robot_side[1] / norm)
        return ex, (-ex[1], ex[0])

    def to_domain(self, *, x: float, y: float) -> tuple[float, float]:
        ex, ey = self._axes()
        dx, dy = x - self.barrier_center[0], y - self.barrier_center[1]
        return dx * ex[0] + dy * ex[1], dx * ey[0] + dy * ey[1]

    def to_map(self, *, x: float, y: float) -> tuple[float, float]:
        ex, ey = self._axes()
        return (
            self.barrier_center[0] + x * ex[0] + y * ey[0],
            self.barrier_center[1] + x * ex[1] + y * ey[1],
        )

    def yaw_to_domain(self, *, yaw: float) -> float:
        return math.remainder(yaw - self._rotation(), math.tau)

    def yaw_to_map(self, *, yaw: float) -> float:
        return math.remainder(yaw + self._rotation(), math.tau)

    def _rotation(self) -> float:
        ex, _ = self._axes()
        return math.atan2(ex[1], ex[0])


class BinSpot(BaseModel):
    """One place a person puts the bin. The one direction the robot throws at a bin
    there is the robot side's (`observe()["toss_direction"]`)."""

    model_config = ConfigDict(frozen=True)

    # Map frame: the bin's centre and the yaw of its long side.
    center: tuple[float, float]
    yaw: float
    description: str


# The tape centre is the robot-side edge moved half the 0.05 m thickness across it.
ROOM_FRAME: Final = RealRoomFrame(barrier_center=(0.0, 0.374), robot_side=(0.026690, -0.999644))

BIN_SPOTS: Final[dict[Tossing3DSide, BinSpot]] = {
    Tossing3DSide.ROBOT: BinSpot(
        center=(-0.30, -0.78),
        yaw=math.pi / 2,
        description="the close spot on the robot's side of the tape, long side along y",
    ),
    Tossing3DSide.OPPOSITE: BinSpot(
        center=(0.24, 1.40),
        yaw=0.0,
        description="the far spot across the tape, inside x 0.03..0.45, y 1.24..1.56, "
        "long side along x",
    ),
}

# Where a person puts the cube on every reset: on the robot's side, clear of both bin
# spots and of the ceiling cameras' overlap band (map y -0.15..0.18). Not a room fact
# given with the others; one named value so it is easy to move.
CUBE_SPOT: Final[tuple[float, float]] = (0.70, -0.60)
