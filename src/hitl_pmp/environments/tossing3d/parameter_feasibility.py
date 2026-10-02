"""Reject provably blocked toss base targets without predicting throw success.

`TossDirectionSelector` uses this one-sided check as the cheap pre-filter in front
of the real base planner, so the planner never runs on a stand these proofs rule out.
Accepting a stand means only that these geometric proofs did not reject it. No
search, simulator step or random draw is performed.
"""

import math

import numpy as np

from .types import PlanarCollisionBox, TossFeasibilityGeometry

# Ambiguous grazing contacts remain the controller's decision. A rejection needs
# strict overlap or strict separation, not a floating-point boundary coincidence.
_GEOMETRY_TOLERANCE = 1e-9

# The room fixture's collider name in the installed task (`tossing_room`); the
# outline itself is derived from those colliders' geometry, never written here.
_ROOM_COLLIDER_TAG = "tossing_room"

# Two wall segments meet if their endpoints are this close: generous against
# float32 collider poses, and still far too narrow for the 0.55 m base to pass.
_WALL_JOIN_TOLERANCE_M = 0.05


class TossParameterFeasibility:
    """A pure geometry gate on one `(standoff, rotation)` base target."""

    @staticmethod
    def rejection_reason_from_geometry(
        *, geometry: TossFeasibilityGeometry, params: np.ndarray
    ) -> str | None:
        """Check only the commanded base target and a proved path obstruction.

        `params` is the controller's `(distance, rotation)` pair: the stand pose, which
        is all this gate reads.

        The controller casts parameters to float32 before deriving its target;
        matching that cast matters for proposals adjacent to collision boundaries.
        The RRT bounds constrain random samples, not endpoints. Every generated
        path segment nevertheless stays in the convex hull of the sampling box,
        start, and goal, because extension is linear in x/y. A rectangle spanning
        that hull separates endpoints on opposite sides, even when the target
        itself is collision-free. A finite barrier that can be routed around is
        not rejected by that proof.
        """
        if params.shape != (2,) or not np.isfinite(params).all():
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            controller_params = np.asarray(params, dtype=np.float32)
        if not np.isfinite(controller_params).all():
            return None
        distance, rotation = (float(v) for v in controller_params)
        yaw = geometry.bin_pose[2] + rotation
        target = (
            float(geometry.bin_pose[0] - distance * np.cos(yaw)),
            float(geometry.bin_pose[1] - distance * np.sin(yaw)),
        )
        # SE2 canonicalizes the heading, then collision_fn writes the pose into
        # the copied robot state before constructing its collision rectangle.
        canonical_yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        with np.errstate(over="ignore", invalid="ignore"):
            collision_pose = np.asarray((*target, canonical_yaw), dtype=geometry.robot_state_dtype)
        if not np.isfinite(collision_pose).all():
            return None
        robot = PlanarCollisionBox(
            name="target_robot",
            center=(float(collision_pose[0]), float(collision_pose[1])),
            width=geometry.robot_size[0],
            height=geometry.robot_size[1],
            yaw=float(collision_pose[2]),
        )
        for obstacle in geometry.obstacles:
            if TossParameterFeasibility._strictly_overlap(first=robot, second=obstacle):
                return f"target_collision:{obstacle.name}"
        polygon = TossParameterFeasibility.room_polygon(geometry=geometry)
        if polygon is not None and not TossParameterFeasibility._inside(
            point=target, polygon=polygon
        ):
            return "outside_room"
        for obstacle in geometry.obstacles:
            if TossParameterFeasibility._separates(
                geometry=geometry, target=target, obstacle=obstacle
            ):
                return f"separating_obstacle:{obstacle.name}"
        return None

    @staticmethod
    def room_polygon(*, geometry: TossFeasibilityGeometry) -> list[tuple[float, float]] | None:
        """The room's floor outline, chained from its wall colliders' centre lines.

        Each wall collider is a thin rectangle; its long axis is a wall segment. The
        segments are chained end to end (endpoints within `_WALL_JOIN_TOLERANCE_M`,
        far narrower than the 0.55 m base) into one closed loop, so a stand centre
        outside it is unreachable. No closed loop -- missing, partial or unfamiliar
        walls -- means no room check, never a rejection on guessed geometry.
        """
        segments = []
        for obstacle in geometry.obstacles:
            if _ROOM_COLLIDER_TAG not in obstacle.name:
                continue
            length = max(obstacle.width, obstacle.height)
            axis = TossParameterFeasibility._axes(box=obstacle)[
                0 if obstacle.width >= obstacle.height else 1
            ]
            center = np.asarray(obstacle.center)
            segments.append((
                tuple(center - axis * length / 2),
                tuple(center + axis * length / 2),
            ))
        if len(segments) < 3:
            return None
        polygon = [segments[0][0], segments[0][1]]
        unused = segments[1:]
        while unused:
            tail = np.asarray(polygon[-1])
            joined = None
            for first, second in unused:
                if np.linalg.norm(np.asarray(first) - tail) <= _WALL_JOIN_TOLERANCE_M:
                    joined = (first, second)
                    break
                if np.linalg.norm(np.asarray(second) - tail) <= _WALL_JOIN_TOLERANCE_M:
                    joined = (second, first)
                    break
            if joined is None:
                return None
            polygon.append(joined[1])
            unused.remove(joined if joined in unused else (joined[1], joined[0]))
        if (
            np.linalg.norm(np.asarray(polygon[-1]) - np.asarray(polygon[0]))
            > _WALL_JOIN_TOLERANCE_M
        ):
            return None
        return [(float(x), float(y)) for x, y in polygon[:-1]]

    @staticmethod
    def _inside(*, point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        inside = False
        for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    @staticmethod
    def _axes(*, box: PlanarCollisionBox) -> tuple[np.ndarray, np.ndarray]:
        cosine, sine = math.cos(box.yaw), math.sin(box.yaw)
        return np.array([cosine, sine]), np.array([-sine, cosine])

    @staticmethod
    def _strictly_overlap(*, first: PlanarCollisionBox, second: PlanarCollisionBox) -> bool:
        first_axes = TossParameterFeasibility._axes(box=first)
        second_axes = TossParameterFeasibility._axes(box=second)
        displacement = np.asarray(first.center) - np.asarray(second.center)
        for axis in (*first_axes, *second_axes):
            radius = sum(
                size * abs(float(np.dot(direction, axis))) / 2
                for size, direction in (
                    (first.width, first_axes[0]),
                    (first.height, first_axes[1]),
                    (second.width, second_axes[0]),
                    (second.height, second_axes[1]),
                )
            )
            if abs(float(np.dot(displacement, axis))) >= radius - _GEOMETRY_TOLERANCE:
                return False
        return True

    @staticmethod
    def _separates(
        *,
        geometry: TossFeasibilityGeometry,
        target: tuple[float, float],
        obstacle: PlanarCollisionBox,
    ) -> bool:
        center = np.asarray(obstacle.center)
        start = np.asarray(geometry.robot_pose[:2]) - center
        end = np.asarray(target) - center
        hull_vertices = (
            np.asarray([
                *((x, y) for x in geometry.sampling_x_bounds for y in geometry.sampling_y_bounds),
                geometry.robot_pose[:2],
                target,
            ])
            - center
        )
        # Interpolated RRT poses also round on writes into the robot state. Bound
        # their projection error over the whole hull before claiming separation;
        # a sub-precision obstacle or a marginal span cannot establish that proof.
        rounding_slack = max(
            _GEOMETRY_TOLERANCE,
            2
            * float(np.finfo(geometry.robot_state_dtype).eps)
            * max(1.0, float(np.abs(hull_vertices + center).max())),
        )
        first, second = TossParameterFeasibility._axes(box=obstacle)
        for normal, tangent, thickness, length in (
            (first, second, obstacle.width, obstacle.height),
            (second, first, obstacle.height, obstacle.width),
        ):
            start_side = float(np.dot(start, normal))
            end_side = float(np.dot(end, normal))
            half_thickness = thickness / 2 + rounding_slack
            opposite = (start_side < -half_thickness and end_side > half_thickness) or (
                end_side < -half_thickness and start_side > half_thickness
            )
            if (
                opposite
                and thickness > 2 * rounding_slack
                and float(np.abs(hull_vertices @ tangent).max()) < length / 2 - rounding_slack
            ):
                return True
        return False
