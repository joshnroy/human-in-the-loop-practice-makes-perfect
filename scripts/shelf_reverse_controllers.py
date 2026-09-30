"""Individual Shelf reverse skills; native forward controllers remain unchanged.

These prototype skills use the native base/arm planner and controller execution.
They never modify the physical state, predicates, or native task configuration.
"""

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.inverse_kinematics import (
    inverse_kinematics,
    check_collisions_with_held_object,
)
from pybullet_helpers.motion_planning import run_motion_planning
from spatialmath import SE2
from prpl_utils.utils import get_signed_angle_distance

from kinder_models.dynamic3d.shelf.parameterized_skills import (
    PickShelfController,
    PlaceShelfController,
)
from kinder_models.dynamic3d.utils import (
    GRASP_TRANSFORM_TO_OBJECT,
    WORLD_X_BOUNDS,
    WORLD_Y_BOUNDS,
    _ARM_MAX_ACCELERATION,
    _ARM_MAX_VELOCITY,
    _compute_per_joint_profile,
    run_base_motion_planning,
    wrap_arm_joint_difference,
)


class ReversePlanning:
    @staticmethod
    def support_lift(sim, target, joints, object_id, transform):
        """Separate existing upward support contact, then require native clearance."""
        bodies = sim.get_collision_bodies(object_id)
        if check_collisions_with_held_object(
            sim.robot, bodies, sim.physics_client_id, None, None, joints
        ):
            raise ValueError("Robot is not clear at retrieval grasp")
        check_collisions_with_held_object(
            sim.robot, bodies, sim.physics_client_id, object_id, transform, joints
        )
        supports = {}
        fixture_ids = set(sim._static_colliders.values()) | {sim._cupboard1_shelf_id}
        for body in bodies:
            for contact in p.getClosestPoints(
                object_id, body, 1e-6, physicsClientId=sim.physics_client_id
            ):
                if body not in fixture_ids or contact[7][2] < 0.95 or contact[8] < -0.001:
                    raise ValueError(
                        "Initial retrieval contact is not shallow upward shelf support"
                    )
                supports[body] = min(supports.get(body, 0.0), contact[8])
        trajectory = [joints]
        for dz in np.linspace(0.00025, 0.03, 120):
            pose = Pose(
                (target.position[0], target.position[1], target.position[2] + dz),
                target.orientation,
            )
            q = inverse_kinematics(sim.robot, pose, set_joints=True)
            if check_collisions_with_held_object(
                sim.robot, bodies, sim.physics_client_id, object_id, transform, q
            ):
                # Existing support is the only permitted contact and may only separate.
                if check_collisions_with_held_object(
                    sim.robot, bodies, sim.physics_client_id, None, None, q
                ):
                    raise ValueError("Robot collision during support separation")
                for body in bodies:
                    for contact in p.getClosestPoints(
                        object_id, body, 1e-6, physicsClientId=sim.physics_client_id
                    ):
                        if (
                            body not in supports
                            or contact[7][2] < 0.95
                            or contact[8] < supports[body] - 1e-6
                        ):
                            raise ValueError("New or deepening contact during support separation")
                        supports[body] = contact[8]
            trajectory.append(q)
        if check_collisions_with_held_object(
            sim.robot, bodies, sim.physics_client_id, object_id, transform, trajectory[-1]
        ):
            raise ValueError("Lift did not fully separate shelf contact")
        return trajectory

    @staticmethod
    def pose(state, obj):
        return Pose(
            tuple(state.get(obj, f) for f in ("x", "y", "z")),
            tuple(state.get(obj, f) for f in ("qx", "qy", "qz", "qw")),
        )

    @staticmethod
    def prepare(controller, state, base, target_ee, *, carrying):
        controller._last_state = state
        sim = controller._pybullet_sim
        robot, cube = controller.objects[:2]
        controller._current_base_motion_plan = run_base_motion_planning(
            state=state,
            target_base_pose=base,
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=[cube.name] if carrying else None,
        )
        if controller._current_base_motion_plan is None:
            raise ValueError("No native base path for reverse skill")
        sim.set_state(state)
        if carrying:
            sim.base_link_to_held_obj = multiply_poses(
                sim.get_ee_pose().invert(), ReversePlanning.pose(state, cube)
            )
        plan_state = state.copy()
        plan_state.set(robot, "pos_base_x", base.x)
        plan_state.set(robot, "pos_base_y", base.y)
        plan_state.set(robot, "pos_base_rot", base.theta())
        sim.set_state(plan_state, cube if carrying else None)
        target = inverse_kinematics(sim.robot, target_ee, set_joints=False)
        object_id = sim._cubes[cube.name]
        if not carrying:
            sim.base_link_to_held_obj = multiply_poses(
                target_ee.invert(), ReversePlanning.pose(state, cube)
            )
        held = {"held_object": object_id, "base_link_to_held_obj": sim.base_link_to_held_obj}
        approach = run_motion_planning(
            sim.robot,
            sim.get_robot_joints(),
            target,
            collision_bodies=sim.get_collision_bodies(object_id if carrying else None),
            seed=0,
            physics_client_id=sim.physics_client_id,
            **(held if carrying else {}),
        )
        lift = []
        retract_start = target
        if not carrying and getattr(controller, "separate_support", False):
            lift = ReversePlanning.support_lift(
                sim, target_ee, target, object_id, sim.base_link_to_held_obj
            )
            retract_start = lift[-1]
        retract = run_motion_planning(
            sim.robot,
            retract_start,
            controller.home_joints.tolist(),
            collision_bodies=sim.get_collision_bodies(None if carrying else object_id),
            seed=0,
            physics_client_id=sim.physics_client_id,
            **({} if carrying else held),
        )
        if approach is None or retract is None:
            raise ValueError(
                f"No native collision-free arm approach/retract: approach={approach is not None}, retract={retract is not None}"
            )
        if hasattr(controller, "physical_joint_limits"):
            for q in [*approach, *lift, *retract]:
                for value, limits in zip(q[:7], controller.physical_joint_limits):
                    if limits is not None and not limits[0] <= value <= limits[1]:
                        raise ValueError("Reverse plan exceeds actual MuJoCo joint limits")
        controller._current_arm_joint_plan = approach
        controller._current_retract_plan = retract
        if lift:
            controller._checked_retract_waypoints = [*lift[1::8], lift[-1], *retract[1:]]
        start = np.array(controller._get_current_robot_arm_conf()[:7])
        finish = np.array(target[:7])
        controller._approach_trajectory, controller._approach_traj_dir = _compute_per_joint_profile(
            start, finish, _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        controller._approach_start_joints = start.copy()
        controller._approach_step_idx = 0
        controller._retract_trajectory, controller._retract_traj_dir = _compute_per_joint_profile(
            finish, controller.home_joints[:7], _ARM_MAX_VELOCITY, _ARM_MAX_ACCELERATION
        )
        controller._retract_start_joints = finish.copy()
        controller._retract_step_idx = 0


class RetrieveShelfController(PickShelfController):
    """Retrieve one actual shelf cube; this is distinct from native OnGround pick."""

    separate_support = True

    def step(self):
        if not self._navigated:
            return super().step()
        action = np.zeros(11, dtype=np.float32)
        base = self._get_current_robot_pose()
        target_base = self._current_base_motion_plan[-1]
        action[:3] = [
            target_base.x - base.x,
            target_base.y - base.y,
            get_signed_angle_distance(target_base.theta(), base.theta()),
        ]
        current = np.asarray(self._get_current_robot_arm_conf()[:7])
        if not self._pre_grasp:
            queue = self._checked_approach_waypoints
            action[-1] = 0.0
        elif not self._closed_gripper:
            action[-1] = 1.0
            grip = self._last_state.get(self.objects[0], "pos_gripper")
            if grip > 0.2 and abs(grip - self._last_gripper_state) < 0.002:
                self._grasp_settle += 1
            else:
                self._grasp_settle = 0
            if self._grasp_settle >= 5:
                self._closed_gripper = True
            self._last_gripper_state = grip
            return action
        else:
            queue = self._checked_retract_waypoints
            action[-1] = 1.0
        while (
            queue
            and np.max(np.abs(wrap_arm_joint_difference(np.asarray(queue[0][:7]) - current)))
            < 0.008
        ):
            queue.pop(0)
        if not queue:
            if not self._pre_grasp:
                self._pre_grasp = True
            else:
                self._lifted = True
            return action
        action[3:10] = 0.5 * wrap_arm_joint_difference(np.asarray(queue[0][:7]) - current)
        return action

    def reset(self, state, params):
        self._current_params = np.asarray(params)
        base = SE2(*params[:3])
        target = multiply_poses(
            ReversePlanning.pose(state, self.objects[1]), GRASP_TRANSFORM_TO_OBJECT
        )
        if len(params) >= 7:
            obj = self.objects[1]
            target = Pose(
                (state.get(obj, "x") - 0.005, state.get(obj, "y"), state.get(obj, "z") + 0.005),
                tuple(params[3:7]),
            )
        if len(params) == 10:
            target = Pose(tuple(params[3:6]), tuple(params[6:10]))
        ReversePlanning.prepare(self, state, base, target, carrying=False)
        self._checked_approach_waypoints = list(self._current_arm_joint_plan[1:])
        self._grasp_settle = 0


class ReturnCubeToStartController(PlaceShelfController):
    """Place one held cube at its saved valid native ground-start pose."""

    def step(self):
        if not self._navigated:
            return super().step()
        if not self._pre_place and hasattr(self, "supported_in_initial_region"):
            current_object = ReversePlanning.pose(self._last_state, self.objects[1])
            angle = (
                Rotation.from_quat(current_object.orientation).inv()
                * Rotation.from_quat(self._desired_object.orientation)
            ).magnitude()
            xy_error = np.linalg.norm(
                np.asarray(current_object.position[:2])
                - np.asarray(self._desired_object.position[:2])
            )
            if (
                self.supported_in_initial_region(self._last_state)
                and angle < 0.001745
                and xy_error < 0.005
            ):
                self.release_observations["supported"] = {
                    f: self._last_state.get(self.objects[1], f)
                    for f in ("x", "y", "z", "qx", "qy", "qz", "qw")
                }
                self._pre_place = True
                self._checked_approach_waypoints.clear()
                # The support contact determines the actual release pose.
                sim = self._pybullet_sim
                sim.set_state(self._last_state)
                retract = run_motion_planning(
                    sim.robot,
                    sim.get_robot_joints(),
                    self.home_joints.tolist(),
                    collision_bodies=sim.get_collision_bodies(),
                    seed=0,
                    physics_client_id=sim.physics_client_id,
                )
                if retract is None:
                    raise ValueError("No native retract plan from actual supported release pose")
                self._checked_retract_waypoints = list(retract[1:])
        action = np.zeros(11, dtype=np.float32)
        base = self._get_current_robot_pose()
        target_base = self._current_base_motion_plan[-1]
        action[:3] = [
            target_base.x - base.x,
            target_base.y - base.y,
            get_signed_angle_distance(target_base.theta(), base.theta()),
        ]
        current = np.asarray(self._get_current_robot_arm_conf()[:7])
        if not self._pre_place:
            queue = self._checked_approach_waypoints
            action[-1] = 1.0
        elif not self._open_gripper:
            action[-1] = 0.0
            if self._last_state.get(self.objects[0], "pos_gripper") < 0.001:
                self._release_settle += 1
            else:
                self._release_settle = 0
            if self._release_settle >= 5:
                self._open_gripper = True
                self.release_observations["opened"] = {
                    f: self._last_state.get(self.objects[1], f)
                    for f in ("x", "y", "z", "qx", "qy", "qz", "qw")
                }
            return action
        else:
            queue = self._checked_retract_waypoints
            action[-1] = 0.0
        while (
            queue
            and np.max(np.abs(wrap_arm_joint_difference(np.asarray(queue[0][:7]) - current)))
            < 0.008
        ):
            queue.pop(0)
        if not queue:
            if not self._pre_place:
                return self._precision_lower(action)
            else:
                self._returned = True
            return action
        action[3:10] = 0.5 * wrap_arm_joint_difference(np.asarray(queue[0][:7]) - current)
        return action

    def _precision_lower(self, action):
        """Align the measured held cube above the floor before lowering it."""
        sim = self._pybullet_sim
        state = self._last_state
        cube = self.objects[1]
        sim.set_state(state)
        actual_object = ReversePlanning.pose(state, cube)
        target = Pose(
            (
                self._desired_object.position[0],
                self._desired_object.position[1],
                self._lower_target_z,
            ),
            self._desired_object.orientation,
        )
        xy_error = np.linalg.norm(
            np.asarray(actual_object.position[:2]) - np.asarray(target.position[:2])
        )
        angle = (
            Rotation.from_quat(actual_object.orientation).inv()
            * Rotation.from_quat(target.orientation)
        ).magnitude()
        if (
            xy_error < 0.003
            and angle < 0.001
            and abs(actual_object.position[2] - self._lower_target_z) < 0.003
        ):
            self._lower_target_z = max(
                self._desired_object.position[2], self._lower_target_z - 0.001
            )
            target = Pose(
                (target.position[0], target.position[1], self._lower_target_z), target.orientation
            )
        transform = multiply_poses(sim.get_ee_pose().invert(), actual_object)
        target_ee = multiply_poses(target, transform.invert())
        joints = inverse_kinematics(sim.robot, target_ee, set_joints=False)
        for value, limits in zip(joints[:7], self.physical_joint_limits):
            if limits is not None and not limits[0] <= value <= limits[1]:
                raise ValueError("Precision placement target exceeds actual MuJoCo joint limits")
        held = sim._cubes[cube.name]
        if check_collisions_with_held_object(
            sim.robot,
            sim.get_collision_bodies(held),
            sim.physics_client_id,
            held,
            transform,
            joints,
        ):
            raise ValueError("Precision placement target has native collision")
        action[3:10] = 0.5 * wrap_arm_joint_difference(
            np.asarray(joints[:7]) - np.asarray(self._get_current_robot_arm_conf()[:7])
        )
        action[-1] = 1.0
        return action

    def reset(self, state, params):
        self._current_params = np.asarray(params)
        base = SE2(*params[:3])
        target_object = Pose(tuple(params[3:6]), tuple(params[6:10]))
        self._desired_object = target_object
        self._lower_target_z = target_object.position[2] + 0.04
        target_object = Pose(
            (target_object.position[0], target_object.position[1], self._lower_target_z),
            target_object.orientation,
        )
        self._pybullet_sim.set_state(state)
        transform = multiply_poses(
            self._pybullet_sim.get_ee_pose().invert(), ReversePlanning.pose(state, self.objects[1])
        )
        target_ee = multiply_poses(target_object, transform.invert())
        ReversePlanning.prepare(self, state, base, target_ee, carrying=True)
        self._checked_approach_waypoints = list(self._current_arm_joint_plan[1:])
        self._checked_retract_waypoints = list(self._current_retract_plan[1:])
        self._release_settle = 0
        self.release_observations = {}
