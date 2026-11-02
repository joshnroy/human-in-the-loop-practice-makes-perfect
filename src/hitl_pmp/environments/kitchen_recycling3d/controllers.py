"""Object-aware grasp refinement over pinned KINDER motion primitives.

This module is simulation-only. No global vendor constants or cube controllers change.
"""

import numpy as np
from kinder_models.dynamic3d.tossing.parameterized_skills import (
    PickCubeController,
    PickCubeGraspCandidate,
    plan_checked_arm_motion,
)
from kinder_models.dynamic3d.utils import (
    get_target_robot_pose_from_parameters,
    inverse_kinematics_fixed_effort,
    run_base_motion_planning,
)
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import InverseKinematicsError
from pybullet_helpers.motion_planning import check_collisions_with_held_object
from scipy.spatial.transform import Rotation
from spatialmath import SE2
from tomsgeoms2d.structs import Rectangle

from .domain import ObjectObservation, grasp_yaws, projected_bounds


class RecyclingPickController(PickCubeController):
    """Keep collision-checked pick/retrace; replace cube poses and attachment geometry."""

    def configure(self, *, episode, spec, calibration):
        self.episode = episode
        self.spec = spec
        self.calibration = calibration
        self.grasp_transform = Pose(
            (-0.005, 0.0, calibration.grasp_offset), (2**-0.5, 2**-0.5, 0.0, 0.0)
        )

    def grasp_candidates(self, *args):
        (state,) = args
        target = self.objects[1]
        observation = ObjectObservation(
            name=self.spec.name,
            position=tuple(state.get(target, axis) for axis in "xyz"),
            quaternion=tuple(state.get(target, key) for key in ("qx", "qy", "qz", "qw")),
        )
        candidates = []
        for yaw in grasp_yaws(spec=self.spec, observation=observation):
            relative = Rotation.from_euler("z", -yaw) * Rotation.from_quat(observation.quaternion)
            lower, upper = projected_bounds(spec=self.spec, rotation=relative)
            width = upper[1] - lower[1]
            if width >= 0.085:
                continue
            opening = min(0.085, width + 0.012)
            candidates.append(
                PickCubeGraspCandidate(tuple(Rotation.from_euler("z", yaw).as_quat()), 0.0, opening)
            )
        alternatives = []
        for candidate in candidates:
            yaw = Rotation.from_quat(candidate.grasp_quat).as_euler("xyz")[2]
            for offset in (np.pi / 2, -np.pi / 2):
                face = tuple(Rotation.from_euler("z", yaw + offset).as_quat())
                alternatives.append(
                    PickCubeGraspCandidate(face, 0.0, candidate.opening, yaw_offset=-offset)
                )
        candidates.extend(alternatives)
        return candidates

    def rescue_sides(self, *args):
        return []

    def _robot_is_close_to_pose(self, *args, atol=0.004):
        """Preserve the millimetre clearance left by a cylinder in the open gripper."""
        (pose,) = args
        return super()._robot_is_close_to_pose(pose, atol=atol)

    def _robot_is_close_to_conf(self, *args, atol=0.04):
        """Pass through intermediate waypoints, but converge before phase handoffs."""
        (conf,) = args
        plan = self.plans.get(self.current_phase)
        if plan and np.array_equal(conf[:7], plan[-1][:7]):
            atol = 0.008
        return super()._robot_is_close_to_conf(conf, atol=atol)

    def _step_trajectory_phase(self, *args):
        """Damp the endpoint command instead of oscillating across its tolerance."""
        phase, next_phase = args
        action = super()._step_trajectory_phase(phase, next_phase)
        if self._plan_step_idx[phase] == len(self.plans[phase]) - 1:
            action[3:10] *= 0.5
        return action

    def _plan_base_motion(self, *args):
        """Keep the chassis clear of every object, including the not-yet-held target."""
        state, target, extend_xy_magnitude, extend_rot_magnitude, grasp_quat = args
        simulator = self._pybullet_sim
        assert simulator is not None
        simulator.set_state(state)
        carried = self._with_fingers(self.home_joints.tolist(), 0.0)
        bodies = simulator.get_collision_bodies()
        yaw = 2 * float(np.arctan2(grasp_quat[2], grasp_quat[3]))
        target_pose = get_target_robot_pose_from_parameters(
            SE2(state.get(target, "x"), state.get(target, "y"), yaw),
            self.TARGET_DISTANCE,
            self.TARGET_ROTATION,
        )
        footprints = []
        for spec in self.episode.config.objects:
            observation = self.episode.observations[spec.name]
            lower, upper = projected_bounds(
                spec=spec, rotation=Rotation.from_quat(observation.quaternion)
            )
            center = np.asarray(observation.position[:2]) + (lower[:2] + upper[:2]) / 2
            extent = upper[:2] - lower[:2] + 0.02
            footprints.append(Rectangle.from_center(*center, *extent, rotation_about_center=0.0))

        def collides(*args):
            (pose,) = args
            try:
                self.episode.check_drive(start=(pose.x, pose.y), end=(pose.x, pose.y))
            except PermissionError:
                return True
            return simulator.robot_collides_at_base(pose, carried, collision_bodies=bodies)

        left, bottom, right, top = self.episode.config.kitchen_bounds
        radius = self.episode.config.robot_radius
        return run_base_motion_planning(
            state=state,
            target_base_pose=target_pose,
            x_bounds=(left + radius, right - radius),
            y_bounds=(bottom + radius, top - radius),
            seed=0,
            extend_xy_magnitude=extend_xy_magnitude,
            extend_rot_magnitude=extend_rot_magnitude,
            bounding_boxes=simulator.bounding_boxes,
            static_obstacles=(*simulator.static_base_obstacles, *footprints),
            body_collision_fn=collides,
        )

    def _plan_grasp_and_lift(self, *args, endpoints_only=False):
        state, start_joints, candidate, held_object_collision_threshold = args
        simulator = self._pybullet_sim
        assert simulator is not None
        simulator.set_state(state)
        start_joints = self._with_fingers(start_joints, candidate.finger_state)
        target = self.objects[1]
        self._planned_cube_position = np.array([state.get(target, axis) for axis in "xyz"])
        rotation = Rotation.from_quat([state.get(target, key) for key in ("qx", "qy", "qz", "qw")])
        offset = self.calibration.grasp_offset
        if self.spec.kind in ("bottle", "can"):
            if abs(rotation.apply([0, 0, 1])[2]) < 0.7:
                offset = min(offset, 0.02)
            else:
                _, upper = projected_bounds(spec=self.spec, rotation=rotation)
                offset = max(offset, upper[2] - 0.015)
        self.grasp_transform = Pose((-0.005, 0.0, offset), self.grasp_transform.orientation)
        grasp = multiply_poses(
            Pose(tuple(self._planned_cube_position), candidate.grasp_quat), self.grasp_transform
        )
        try:
            joints = self._with_fingers(
                inverse_kinematics_fixed_effort(simulator.robot, grasp, set_joints=False),
                candidate.finger_state,
            )
        except InverseKinematicsError:
            return False
        if self._crowds_a_bin_wall(joints):
            return False
        target_body = simulator._cubes[target.name]
        obstacles = simulator.get_collision_bodies(held_object=target_body)

        def clearance(*args):
            (configuration,) = args
            return not check_collisions_with_held_object(
                simulator.robot,
                obstacles,
                simulator.physics_client_id,
                held_object=None,
                base_link_to_held_obj=None,
                joint_state=configuration,
                distance_threshold=self.GRASP_OBSTACLE_CLEARANCE,
            )

        descent = plan_checked_arm_motion(
            simulator.robot,
            start_joints,
            joints,
            collision_bodies=simulator.get_collision_bodies(),
            additional_state_constraint_fn=clearance,
            physics_client_id=simulator.physics_client_id,
            endpoints_only=endpoints_only,
        )
        if descent is None:
            return False
        object_pose = Pose(
            tuple(self._planned_cube_position),
            tuple(state.get(target, key) for key in ("qx", "qy", "qz", "qw")),
        )
        simulator.base_link_to_held_obj = multiply_poses(grasp.invert(), object_pose)
        lift = plan_checked_arm_motion(
            simulator.robot,
            descent[0][-1],
            self._with_fingers(self.home_joints.tolist(), candidate.finger_state),
            collision_bodies=obstacles,
            held_object=target_body,
            base_link_to_held_obj=simulator.base_link_to_held_obj,
            distance_threshold=held_object_collision_threshold,
            physics_client_id=simulator.physics_client_id,
            endpoints_only=endpoints_only,
        )
        if lift is None:
            return False
        if not endpoints_only:
            self.plans[self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE] = descent[1]
            self.plans[self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME] = lift[1]
            self._plan_step_idx[self.PickCubeControllerPhase.MOVE_ARM_DOWN_AROUND_CUBE] = 0
            self._plan_step_idx[self.PickCubeControllerPhase.LIFT_CUBE_TO_HOME] = 0
        return True
