"""Top-down 2D footprints for the gripper, cubes and drawer walls.

These are cheap pre-filters, run before any inverse kinematics: a grasp or push whose
fingers or palm already overlap a neighbour in the plane cannot succeed, so it is never
handed to the 3D planning scene, which remains the authority for everything it passes.
"""

import numpy as np
import shapely
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
    def island_board() -> Polygon:
        """What a finger on a floor cube faces at the island: its bottom board, which
        stands 2 cm high and 2 cm back from the drawer faces above it."""
        return box(-1.0, -1.1, SweepDrawerScene.COUNTER_EDGE_X, 1.1)

    @staticmethod
    def island_faces(*, drawer_pos: float) -> Polygon:
        """What the palm over a floor cube faces: the drawer faces, which begin 3.6 cm
        above the floor -- carried out over the floor by the task's drawer when it is
        open, since the arm cannot reach in beneath it."""
        return box(-1.0, -1.1, SweepDrawerScene.DRAWER_FACE_X + max(drawer_pos, 0.0), 1.1)

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
    def finger_clearances(
        *,
        centres: np.ndarray,
        yaw: float,
        obstacles: list[Polygon],
        gap: float,
        widen: np.ndarray,
        inner: float | None = None,
    ) -> np.ndarray:
        """`finger_clearance` for many placements at once: one row of `centres` and one
        `widen` each, the same closing axis and opening. Overlap is reported as 0, not as
        a negative depth -- the grasp search asks only whether a placement is clear.

        With `inner`, the cube's extent from its centre along the closing axis, each
        finger's footprint runs from where it starts in to the cube: the strip its pad
        sweeps as it closes. A neighbour in that strip is taken along with the cube.
        """
        n = len(centres)
        if not obstacles:
            return np.ones(n)
        g = GripperGeometry
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-np.sin(yaw), np.cos(yaw)])
        half = g.FINGER_HALF_WIDTH + np.asarray(widen, dtype=float) / 2
        near, far = (gap / 2 if inner is None else inner), gap / 2 + g.FINGER_THICK
        along = np.array([near, far, far, near])
        across = np.array([-1.0, -1.0, 1.0, 1.0])
        # (placement, finger, corner, xy)
        corners = (
            np.asarray(centres, dtype=float)[:, None, None, :]
            + np.array([-1.0, 1.0])[None, :, None, None] * along[None, None, :, None] * u
            + (half[:, None, None] * across[None, None, :])[..., None] * v
        )
        fingers = shapely.polygons(corners.reshape(n * 2, 4, 2))
        gaps = shapely.distance(fingers[:, None], np.array(obstacles, dtype=object)[None, :])
        return np.minimum(gaps.min(axis=1).reshape(n, 2).min(axis=1), 1.0)

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
