"""The real room as `TossFeasibilityGeometry`, so the sim's toss pipeline runs on it.

`TossDirectionSelector` and `TossStandoffBand` decide where the robot stands to throw
from one `TossFeasibilityGeometry`: the robot's pose and footprint, the bin's pose, the
overhead obstacle rectangles and the room outline. On the simulator that geometry is
read out of KINDER's scene; here it is built from one `observe()` reply and the
`RealSceneLayout`, and the same code then runs on it unchanged:

- **The room** is the drivable box, as four thin wall rectangles tagged
  `tossing_room`, which is how `TossParameterFeasibility.room_polygon` recognizes walls.
  A stand whose footprint touches a wall is rejected by the overlap test; one whose
  centre is outside the box by the room test.
- **The barrier** is its own finite rectangle, named as the sim names it, so it
  collides like any obstacle and the selector's side invariant finds it.
- **The bin** is its fitted footprint, named `bin_0`, which the selector's clearance
  ranking excludes exactly as it does in the sim.
- **The barrier is also `uncrossable`.** The sim's barrier spans its room, so a stand
  across it is unreachable and its base planner says so. The real barrier is finite
  and the base could drive round it -- but the real robot never crosses it, so a stand
  whose footprint reaches across the barrier's line is rejected by the gate itself.

There is no base motion planner on this side, and none is needed: with the room, the
obstacles and the uncrossable line in the gate, the gate is the whole feasibility
check (`TossGeometrySource.base_plan_failure` answers `None`).
"""

import math

from hitl_pmp.environments.tossing3d.types import PlanarCollisionBox, TossFeasibilityGeometry

from .types import RealSceneSnapshot

# The sim's own object names, which the shared selector keys on.
BARRIER_NAME = "cuboid_barrier"
BIN_NAME = "bin_0"
ROOM_WALL_PREFIX = "tossing_room_wall"
ROOM_WALL_THICKNESS_M = 0.02


class RealTossGeometry:
    """A static-method container, never instantiated."""

    @staticmethod
    def of(*, snapshot: RealSceneSnapshot) -> TossFeasibilityGeometry | None:
        """`None` when the bin is not localized: no stand can be chosen without it."""
        observation = snapshot.observation
        layout = snapshot.layout
        if observation.bin is None:
            return None
        bin_ = observation.bin
        barrier = observation.barrier
        barrier_box = PlanarCollisionBox(
            name=BARRIER_NAME,
            center=(barrier.x, barrier.y),
            width=barrier.thickness,
            height=barrier.length,
            yaw=barrier.yaw,
        )
        return TossFeasibilityGeometry(
            robot_pose=(observation.robot.x, observation.robot.y, observation.robot.yaw),
            robot_size=layout.robot_size,
            robot_state_dtype="float64",
            bin_pose=(bin_.x, bin_.y, bin_.yaw),
            obstacles=(
                *RealTossGeometry.room_walls(
                    x_bounds=layout.drivable_x, y_bounds=layout.drivable_y
                ),
                barrier_box,
                PlanarCollisionBox(
                    name=BIN_NAME,
                    center=(bin_.x, bin_.y),
                    width=bin_.length,
                    height=bin_.width,
                    yaw=bin_.yaw,
                ),
            ),
            sampling_x_bounds=layout.drivable_x,
            sampling_y_bounds=layout.drivable_y,
            uncrossable=barrier_box,
        )

    @staticmethod
    def room_walls(
        *, x_bounds: tuple[float, float], y_bounds: tuple[float, float]
    ) -> tuple[PlanarCollisionBox, ...]:
        """Four thin walls whose centre lines are exactly the drivable box's edges."""
        (x_min, x_max), (y_min, y_max) = x_bounds, y_bounds
        width, height = x_max - x_min, y_max - y_min
        centre_x, centre_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        return (
            PlanarCollisionBox(
                name=f"{ROOM_WALL_PREFIX}_south",
                center=(centre_x, y_min),
                width=width,
                height=ROOM_WALL_THICKNESS_M,
                yaw=0.0,
            ),
            PlanarCollisionBox(
                name=f"{ROOM_WALL_PREFIX}_east",
                center=(x_max, centre_y),
                width=height,
                height=ROOM_WALL_THICKNESS_M,
                yaw=math.pi / 2,
            ),
            PlanarCollisionBox(
                name=f"{ROOM_WALL_PREFIX}_north",
                center=(centre_x, y_max),
                width=width,
                height=ROOM_WALL_THICKNESS_M,
                yaw=0.0,
            ),
            PlanarCollisionBox(
                name=f"{ROOM_WALL_PREFIX}_west",
                center=(x_min, centre_y),
                width=height,
                height=ROOM_WALL_THICKNESS_M,
                yaw=math.pi / 2,
            ),
        )

    @staticmethod
    def direction_azimuth(*, bin_yaw: float, rotation: float) -> float:
        """The contract's `direction` for a controller `rotation`.

        The controller's stand is `bin - standoff * (cos, sin)(bin_yaw + rotation)`, so
        the azimuth from the bin centre to the stand is that angle plus pi, wrapped to
        (-pi, pi].
        """
        angle = bin_yaw + rotation + math.pi
        return math.atan2(math.sin(angle), math.cos(angle))
