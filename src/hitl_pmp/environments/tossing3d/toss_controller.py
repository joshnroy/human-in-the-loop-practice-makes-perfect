"""The move-and-toss controller Tossing3D dispatches, and the one place its base plan's
planning arguments are built.

`KinderBackend.toss_base_plan_failure`, the check every toss proposal passes before it
is accepted, asks for a base plan with exactly `base_motion_arguments`, the arguments this
controller's own `_plan_base_motion` plans with. The check then cannot accept a stand the
controller refuses. Before, the check rebuilt a subset of them by hand: no 3-D check of
the carried arm and cube, no planning-scene bounding boxes or static obstacles, and so a
coarser densification. A stand could pass it and then be refused when dispatched.

Upstream's `MoveToTossLocationAndTossController._plan_base_motion`, reproduced so that its
arguments can be built without planning; `test_toss_check_mirrors_controller.py` holds it
equal to upstream's at the pinned commit. A scene with a no-drive zone overrides the two
hooks (`kitchen_recycling3d.controllers.RecyclingTossController`).

Imports KINDER at module scope, so it is imported lazily, as `kinder_backend` does.
"""

from typing import Any

import numpy as np
from bilevel_planning.trajectory_samplers.trajectory_sampler import TrajectorySamplingFailure
from kinder.envs.dynamic3d.object_types import MujocoTidyBotRobotObjectType
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    MoveToTossLocationAndTossController,
)
from kinder_models.dynamic3d.utils import (
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    PyBulletSim,
    get_overhead_object_se2_pose,
    get_target_robot_pose_from_parameters,
    run_base_motion_planning,
)

# The names `KinderBackend` declares for the toss's objects; repeated, not imported, since
# `kinder_backend` imports this module.
BIN_NAME = "bin_0"
CUBE_NAME = "cube_0"
BARRIER_NAME = "cuboid_barrier"


class Tossing3DTossController(MoveToTossLocationAndTossController):
    """Upstream's move-and-toss, with its base plan's arguments built by one method."""

    @classmethod
    def for_snapshot(cls, *, snapshot: Any, pybullet_sim: Any = None) -> Any:
        """Ground on `snapshot`'s robot, cube and barrier, as `KinderBackend` does. With no
        sim, one is built from `snapshot`, as upstream's `reset` builds one."""
        (robot,) = snapshot.get_objects(MujocoTidyBotRobotObjectType)
        objects = (
            robot,
            snapshot.get_object_from_name(CUBE_NAME),
            snapshot.get_object_from_name(BARRIER_NAME),
        )
        return cls(
            objects, pybullet_sim=PyBulletSim(snapshot) if pybullet_sim is None else pybullet_sim
        )

    def base_motion_arguments(
        self,
        *,
        x: Any,
        params: np.ndarray,
        extend_xy_magnitude: float = 0.025,
        extend_rot_magnitude: float = np.pi / 8,
        disable_collision_objects: list[str] | None = None,
    ) -> dict[str, Any]:
        """`run_base_motion_planning`'s keyword arguments for a toss stand, the arm riding
        at the retract configuration with the held cube attached by the observed grasp."""
        if disable_collision_objects is None:
            disable_collision_objects = [self.objects[1].name]
        sim = self._pybullet_sim
        assert sim is not None
        held_cube, held_object = self._configure_held_cube(x)
        sim.set_state(x, held_object=held_cube)
        carried = list(self._return_home_arm_config)
        bodies = sim.get_collision_bodies(held_object=held_object)
        refused = self.base_pose_refused

        def body_collision_fn(*args: Any) -> bool:
            (pose,) = args
            if refused(pose=pose):
                return True
            return bool(
                sim.robot_collides_at_base(
                    pose, carried, held_object=held_object, collision_bodies=bodies
                )
            )

        target_base_pose = get_target_robot_pose_from_parameters(
            get_overhead_object_se2_pose(x, x.get_object_from_name(BIN_NAME)),
            params[0],
            params[1],
        )
        x_bounds, y_bounds = self.sampling_bounds()
        return {
            "state": x,
            "target_base_pose": target_base_pose,
            "x_bounds": x_bounds,
            "y_bounds": y_bounds,
            "seed": 0,
            "extend_xy_magnitude": extend_xy_magnitude,
            "extend_rot_magnitude": extend_rot_magnitude,
            "disable_collision_objects": disable_collision_objects,
            "bounding_boxes": sim.bounding_boxes,
            "static_obstacles": sim.static_base_obstacles,
            "body_collision_fn": body_collision_fn,
        }

    def sampling_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Where the base planner samples: the world."""
        return WORLD_X_BOUNDS, WORLD_Y_BOUNDS

    def base_pose_refused(self, *, pose: Any) -> bool:
        """A base pose refused before any collision check: none, in the world."""
        del pose
        return False

    def _plan_base_motion(self, *args: Any) -> Any:
        x, current_params, extend_xy_magnitude, extend_rot_magnitude, disabled = args
        plan = run_base_motion_planning(
            **self.base_motion_arguments(
                x=x,
                params=current_params,
                extend_xy_magnitude=extend_xy_magnitude,
                extend_rot_magnitude=extend_rot_magnitude,
                disable_collision_objects=disabled,
            )
        )
        if plan is None:
            raise TrajectorySamplingFailure("Base motion planning failed")
        return plan
