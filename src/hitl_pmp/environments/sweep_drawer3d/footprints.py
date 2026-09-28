"""Top-down 2D footprints for the gripper, cubes and drawer walls.

These are cheap pre-filters, run before any inverse kinematics: a grasp or push whose
fingers or palm already overlap a neighbour in the plane cannot succeed, so it is never
handed to the 3D planning scene, which remains the authority for everything it passes.
"""

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.geometry.base import BaseGeometry

from .types import GripperGeometry, SweepDrawerScene


class Footprints:
    """Static constructors and clearance queries over shapely polygons."""

    @staticmethod
    def rect(
        *, center: np.ndarray, yaw: float, u0: float, u1: float, v0: float, v1: float
    ) -> Polygon:
        """A rectangle spanning [u0, u1] x [v0, v1] in a frame at `center` rotated by `yaw`."""
        shape = affinity.rotate(box(u0, v0, u1, v1), yaw, use_radians=True, origin=(0, 0))
        return affinity.translate(shape, float(center[0]), float(center[1]))

    @staticmethod
    def cube(*, center: np.ndarray, yaw: float) -> Polygon:
        h = SweepDrawerScene.CUBE_HALF
        return Footprints.rect(center=center, yaw=yaw, u0=-h, u1=h, v0=-h, v1=h)

    @staticmethod
    def drawer_walls(*, drawer_pos: float) -> list[Polygon]:
        """The drawer's front wall (toward the robot) and side walls, which rise above the
        gripper palm when the fingers are at the drawer floor."""
        fx = SweepDrawerScene.DRAWER_FRONT_INNER_X + drawer_pos
        y = SweepDrawerScene.DRAWER_SIDE_INNER_Y
        return [
            box(fx, -1.0, fx + 0.05, 1.0),
            box(-1.0, y, 3.0, y + 0.1),
            box(-1.0, -y - 0.1, 3.0, -y),
        ]

    @staticmethod
    def clearance(*, shape: BaseGeometry, obstacles: list[Polygon]) -> float:
        """Smallest gap from `shape` to any obstacle; negative (-sqrt(overlap area)) on overlap."""
        best = 1.0
        for ob in obstacles:
            if shape.intersects(ob):
                best = min(best, -(float(shape.intersection(ob).area) ** 0.5))
            else:
                best = min(best, float(shape.distance(ob)))
        return best

    @staticmethod
    def support_distance(*, yaw: float, direction: np.ndarray) -> float:
        """Distance from a cube's centre to its footprint boundary along `direction`."""
        ca = abs(np.cos(yaw) * direction[0] + np.sin(yaw) * direction[1])
        sa = abs(-np.sin(yaw) * direction[0] + np.cos(yaw) * direction[1])
        return SweepDrawerScene.CUBE_HALF * float(ca + sa)

    @staticmethod
    def finger_clearance(
        *,
        center: np.ndarray,
        yaw: float,
        obstacles: list[Polygon],
        gap: float = 0.032,
        widen: float = 0.0,
    ) -> float:
        """Both partly-open fingers lowered around `center`, closing along `yaw`."""
        g = GripperGeometry
        best = 1.0
        for side in (-1.0, 1.0):
            u0, u1 = gap / 2, gap / 2 + g.FINGER_THICK
            if side < 0:
                u0, u1 = -u1, -u0
            finger = Footprints.rect(
                center=center,
                yaw=yaw,
                u0=u0,
                u1=u1,
                v0=-g.FINGER_HALF_WIDTH - widen / 2,
                v1=g.FINGER_HALF_WIDTH + widen / 2,
            )
            best = min(best, Footprints.clearance(shape=finger, obstacles=obstacles))
        return best

    @staticmethod
    def palm(*, center: np.ndarray, yaw: float, lean: float, alpha: float) -> Polygon:
        """The palm's footprint, shifted toward the lean side by a tilted approach."""
        v = np.array([-np.sin(yaw), np.cos(yaw)])
        c = np.asarray(center) + GripperGeometry.PALM_LEVER * np.sin(alpha) * lean * v
        hu, hv = GripperGeometry.PALM_HALF
        return Footprints.rect(center=c, yaw=yaw, u0=-hu, u1=hu, v0=-hv, v1=hv)

    @staticmethod
    def swept(*, start: np.ndarray, end: np.ndarray, radius: float) -> BaseGeometry:
        return LineString([tuple(start), tuple(end)]).buffer(radius)
