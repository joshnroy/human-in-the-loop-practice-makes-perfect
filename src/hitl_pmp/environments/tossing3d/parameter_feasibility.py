"""Reject provably blocked toss base targets without predicting throw success.

The learned sampler uses this one-sided check before selecting parameters.
Accepting a proposal means only that these geometric proofs did
not reject it. No search, simulator step, random draw, or speed/timing tuning is
performed, and missing geometry is accepted.
"""

import math

import numpy as np

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import State

from .kinder_backend import KinderBackend
from .skills import Tossing3DSkills
from .types import PlanarCollisionBox, TossFeasibilityGeometry, Tossing3DState

# Ambiguous grazing contacts remain the controller's decision. A rejection needs
# strict overlap or strict separation, not a floating-point boundary coincidence.
_GEOMETRY_TOLERANCE = 1e-9


class TossParameterFeasibility:
    """A pure geometry gate shared by execution and diagnostic proposal generators."""

    @staticmethod
    def rejection_reason(
        *, state: State, ground_skill: GroundSkill, params: np.ndarray
    ) -> str | None:
        if ground_skill.skill != Tossing3DSkills.MOVE_TO_TOSS_LOCATION_AND_TOSS:
            return None
        if not isinstance(state, Tossing3DState) or state.object_centric is None:
            return None
        geometry = KinderBackend.toss_feasibility_geometry(snapshot=state.object_centric)
        if geometry is None:
            return None
        return TossParameterFeasibility.rejection_reason_from_geometry(
            geometry=geometry, params=params
        )

    @staticmethod
    def rejection_reason_from_geometry(
        *, geometry: TossFeasibilityGeometry, params: np.ndarray
    ) -> str | None:
        """Check only the commanded base target and a proved path obstruction.

        The controller casts parameters to float32 before deriving its target;
        matching that cast matters for proposals adjacent to collision boundaries.
        The RRT bounds constrain random samples, not endpoints. Every generated
        path segment nevertheless stays in the convex hull of the sampling box,
        start, and goal, because extension is linear in x/y. A rectangle spanning
        that hull separates endpoints on opposite sides, even when the target
        itself is collision-free. A finite barrier that can be routed around is
        not rejected by that proof.
        """
        if params.shape != (4,) or not np.isfinite(params).all():
            return None
        with np.errstate(over="ignore", invalid="ignore"):
            controller_params = np.asarray(params[:2], dtype=np.float32)
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
        for obstacle in geometry.obstacles:
            if TossParameterFeasibility._separates(
                geometry=geometry, target=target, obstacle=obstacle
            ):
                return f"separating_obstacle:{obstacle.name}"
        return None

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
