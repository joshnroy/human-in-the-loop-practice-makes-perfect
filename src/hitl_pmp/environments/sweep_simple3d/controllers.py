"""New floor skills using existing collision-aware planning and physical execution."""

from typing import Any

import numpy as np
from pydantic import Field

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError, Motion
from hitl_pmp.environments.sweep_drawer3d.planning_scene import PlanningScene
from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives


class FloorPlanningScene(PlanningScene):
    def plan_base(
        self, *, target: tuple[float, float, float], margin: float = 0.02
    ) -> list[tuple[float, float, float]] | None:
        del margin
        from kinder_models.dynamic3d.utils import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            run_base_motion_planning,
        )
        from spatialmath import SE2

        path = run_base_motion_planning(
            state=self.session.state,
            target_base_pose=SE2(*target),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=(
                ["wiper_0"]
                if self.session.gripper() > 0.2 and self.session.position(name="wiper_0")[2] > 0.08
                else []
            ),
        )
        if path is None:
            return None
        return [(float(p.x), float(p.y), float(p.theta())) for p in path]


class FloorPrimitives(Primitives):
    """Floor pickup uses the tested generic handle grasp with a learned base stance."""

    distance: float = Field(default=0.7, ge=0.55, le=0.85)
    heading_offset: float = Field(default=0.0, ge=-np.pi / 12, le=np.pi / 12)

    def wiper_grasp_offsets(self) -> tuple[float, ...]:
        return 0.0, 0.03, 0.06

    def wiper_approach_angles(self) -> tuple[float, ...]:
        return np.pi / 2, 1.2, 1.8

    def wiper_grasp_yaw(self, *, axis: np.ndarray) -> float:
        del axis
        base = self.session.base()
        wiper = self.session.position(name="wiper_0")
        return float(np.arctan2(base[1] - wiper[1], base[0] - wiper[0]) - np.pi / 2)

    def wiper_stow_goal(self) -> np.ndarray:
        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        base = self.session.base()
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        for yaw in (self.session.yaw(name="wiper_0"), base[2]):
            for radius in (0.45, 0.55, 0.35):
                for height in (0.35, 0.45, 0.25):
                    target_body = Pose.from_rpy(
                        (
                            base[0] + radius * np.cos(base[2]),
                            base[1] + radius * np.sin(base[2]),
                            height,
                        ),
                        (0.0, 0.0, yaw),
                    )
                    target = multiply_poses(target_body, held_tf.invert())
                    for solution in ikfast_closest_inverse_kinematics(
                        self.scene.robot, world_from_target=target
                    )[:12]:
                        path = self.scene.plan_arm(
                            goal=solution[:7],
                            bodies=self.scene.bodies(),
                            held=self.scene.wiper_body,
                            held_tf=held_tf,
                        )
                        if path is not None:
                            return np.asarray(solution[:7])
        raise ExecutionError("No collision-free upright tool transport pose")

    def wiper_handle_geometry(self) -> tuple[int, int]:
        import mujoco

        model = self.session.mj_model
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        handle = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
            key=lambda g: float(model.geom_size[g][2]),
        )
        return handle, 2

    def wiper_in_hand(self, *, gripper: np.ndarray, wiper: np.ndarray) -> bool:
        del wiper
        handle, _ = self.wiper_handle_geometry()
        center = self.session.mj_data.geom_xpos[handle]
        return self.session.gripper() > 0.2 and float(np.linalg.norm(gripper - center)) < 0.15

    def stances(self, *, target: np.ndarray, where: str) -> list[tuple[float, float, float]]:
        del where
        base = self.session.base()
        bearing = float(np.arctan2(base[1] - target[1], base[0] - target[0]))
        bearing += self.heading_offset
        return [
            (
                float(target[0] + self.distance * np.cos(bearing)),
                float(target[1] + self.distance * np.sin(bearing)),
                float((bearing + 2 * np.pi) % (2 * np.pi) - np.pi),
            )
        ]

    def sweep_cube(self, *, cube: str, region: str, distance: float, heading_offset: float) -> str:
        from pybullet_helpers.geometry import Pose, multiply_poses

        ee = self.scene.ee_now()
        observed_wiper = self.session.position(name="wiper_0")
        if not self.wiper_in_hand(gripper=np.asarray(ee.position), wiper=observed_wiper):
            raise ExecutionError("Sweep requires the physical wiper handle in the hand")
        self.stow_wiper()
        ee = self.scene.ee_now()
        observed_wiper = self.session.position(name="wiper_0")
        held_tf = multiply_poses(
            ee.invert(), Pose(tuple(observed_wiper), self.session.quaternion(name="wiper_0"))
        )
        core = self.session.env.unwrapped._object_centric_env
        box = core.task_config["regions"][region]["ranges"][0]
        target = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        initial = self.session.position(name=cube).copy()
        delta = target - initial[:2]
        length = float(np.linalg.norm(delta))
        if length < 0.02:
            return "Cube already at target center"
        direction = delta / length
        angle = float(np.arctan2(direction[1], direction[0]))
        robot_box = core.task_config["regions"]["robot_task_init_region"]["ranges"][0]
        aisle = np.array([(robot_box[0] + robot_box[2]) / 2, (robot_box[1] + robot_box[3]) / 2])
        stance_bearing = (
            float(np.arctan2(aisle[1] - initial[1], aisle[0] - initial[0])) + heading_offset
        )
        stance_angle = float((stance_bearing + 2 * np.pi) % (2 * np.pi) - np.pi)
        stance = (
            float(initial[0] + distance * np.cos(stance_bearing)),
            float(initial[1] + distance * np.sin(stance_bearing)),
            stance_angle,
        )
        self.motion.drive_to(target=stance, grip=1.0)
        bodies = self.scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
        wiper_start = initial[:2] - 0.045 * direction
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        approach = None
        attempts = []
        for tool_yaw in (angle - np.pi / 2, angle + np.pi / 2):
            floor_pose = Pose.from_rpy((*wiper_start, 0.001), (0.0, 0.0, tool_yaw))
            floor_ee = multiply_poses(floor_pose, held_tf.invert())
            for lift_height in (0.14, 0.24):
                hover = Pose(
                    tuple(np.array(floor_ee.position) + [0, 0, lift_height]), floor_ee.orientation
                )
                self.scene.sync()
                solutions = ikfast_closest_inverse_kinematics(
                    self.scene.robot, world_from_target=hover
                )
                attempts.append({
                    "yaw": tool_yaw,
                    "hover": hover.position,
                    "solutions": len(solutions),
                })
                for solution in solutions[:12]:
                    candidate_lower = self.scene.linear_path(
                        start=np.asarray(solution[:7]),
                        target=floor_ee,
                        bodies=bodies,
                        finger_state=0.5,
                        max_jump=0.6,
                    )
                    if candidate_lower is None or any(
                        self.scene.in_collision(
                            joints=self.scene.fingers(arm=q, state=0.5),
                            bodies=bodies,
                            held=self.scene.wiper_body,
                            held_tf=held_tf,
                        )
                        for q in candidate_lower
                    ):
                        continue
                    approach = self.scene.plan_arm(
                        goal=solution[:7],
                        bodies=bodies,
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    if approach is not None:
                        break
                if approach is not None:
                    break
            if approach is not None:
                break
        if approach is None or not self.motion.follow(path=approach, grip=1.0):
            raise ExecutionError(f"No collision-free floor sweep approach: {attempts}")
        lower = self.scene.linear_path(
            start=self.session.arm(), target=floor_ee, bodies=bodies, finger_state=0.5, max_jump=0.6
        )
        if lower is None or any(
            self.scene.in_collision(
                joints=self.scene.fingers(arm=q, state=0.5),
                bodies=bodies,
                held=self.scene.wiper_body,
                held_tf=held_tf,
            )
            for q in lower
        ):
            raise ExecutionError("No collision-free floor sweep descent")
        if not self.motion.follow(path=lower, grip=1.0, final_tol=0.005):
            raise ExecutionError("Floor sweep descent did not converge")
        # Finger compliance can change the grasp transform during reorientation.
        # Close the loop on the observed blade height rather than a stale transform.
        for _ in range(6):
            height = float(self.session.position(name="wiper_0")[2])
            if height <= 0.003:
                break
            current_ee = self.scene.ee_now()
            correction = Pose(
                tuple(np.array(current_ee.position) - [0, 0, min(height - 0.001, 0.02)]),
                current_ee.orientation,
            )
            path = self.scene.linear_path(
                start=self.session.arm(),
                target=correction,
                bodies=bodies,
                finger_state=0.5,
                max_jump=0.6,
            )
            if path is None or not self.motion.follow(path=path, grip=1.0, final_tol=0.003):
                raise ExecutionError("Cannot establish observed floor contact")
        if self.session.position(name="wiper_0")[2] > 0.006:
            raise ExecutionError("Wiper remains above the floor")
        base_origin = np.array(self.session.base()[:2])
        for progress in np.arange(0.012, length + 0.10, 0.012):
            current = self.session.position(name=cube)
            if core._ground_fixture.check_in_region(current, region, core._robot_env):
                break
            next_xy = base_origin + progress * direction
            target_base = (float(next_xy[0]), float(next_xy[1]), stance_angle)
            # Replan against observed moving cubes rather than their stale pre-sweep poses.
            base_path = self.scene.plan_base(target=target_base)
            if base_path is None:
                raise ExecutionError("Base route blocked during contact sweep")
            if not self.motion.drive(path=base_path, grip=1.0, max_ticks=30):
                raise ExecutionError("Contact sweep base did not converge")
            for _ in range(3):
                wiper_now = Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                )
                from scipy.spatial.transform import Rotation

                upright_error = float(
                    np.linalg.norm(Rotation.from_quat(wiper_now.orientation).as_euler("xyz")[:2])
                )
                if wiper_now.position[2] <= 0.008 and upright_error < 0.035:
                    break
                ee_now = self.scene.ee_now()
                live_tf = multiply_poses(ee_now.invert(), wiper_now)
                desired = Pose.from_rpy(
                    (wiper_now.position[0], wiper_now.position[1], 0.001), (0.0, 0.0, tool_yaw)
                )
                correction = multiply_poses(desired, live_tf.invert())
                path = self.scene.linear_path(
                    start=self.session.arm(),
                    target=correction,
                    bodies=bodies,
                    finger_state=0.5,
                    max_jump=0.6,
                )
                if path is None or not self.motion.follow(path=path, grip=1.0, final_tol=0.003):
                    q = self.scene.ik(pose=correction, seed=self.session.arm())
                    collision = (
                        None
                        if q is None
                        else self.scene.in_collision(
                            joints=self.scene.fingers(arm=q, state=0.5), bodies=bodies
                        )
                    )
                    raise ExecutionError(
                        f"Lost checked floor contact: height={wiper_now.position[2]:.4f}, "
                        f"tilt={upright_error:.4f}, path={path is not None}, "
                        f"IK={q is not None}, collision={collision}"
                    )
            if not self.wiper_in_hand(
                gripper=np.asarray(self.scene.ee_now().position),
                wiper=self.session.position(name="wiper_0"),
            ):
                raise ExecutionError("Wiper lost during contact sweep")
        current_ee = self.scene.ee_now()
        lift = self.scene.linear_path(
            start=self.session.arm(),
            target=Pose(
                tuple(np.array(current_ee.position) + [0, 0, 0.16]), current_ee.orientation
            ),
            bodies=bodies,
            finger_state=0.5,
            max_jump=0.6,
        )
        if lift is None or not self.motion.follow(path=lift, grip=1.0):
            raise ExecutionError("No checked lift after floor sweep")
        displacement = float(np.linalg.norm(self.session.position(name=cube)[:2] - initial[:2]))
        if not core._ground_fixture.check_in_region(
            self.session.position(name=cube), region, core._robot_env
        ):
            raise ExecutionError(
                f"Sweep ended outside native target region; cube displacement {displacement:.3f} m"
            )
        return f"Cube displacement {displacement:.3f} m"

    def stow_wiper(self) -> None:
        from pybullet_helpers.geometry import Pose, multiply_poses

        goal = self.wiper_stow_goal()
        if np.max(np.abs(self.session.arm() - goal)) < 0.03:
            return
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        path = self.scene.plan_arm(
            goal=goal, bodies=self.scene.bodies(), held=self.scene.wiper_body, held_tf=held_tf
        )
        if path is None or not self.motion.follow(path=path, grip=1.0, final_tol=0.025):
            raise ExecutionError("No collision-free stow for the physically held wiper")

    def place_wiper_at_start(self) -> str:
        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        self.stow_wiper()
        position, orientation = self.session.initial_pose(name="wiper_0")
        stance = self.stances(target=position, where="floor")[0]
        self.motion.drive_to(target=stance, grip=1.0)
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        body_target = Pose((float(position[0]), float(position[1]), 0.001), orientation)
        ee_target = multiply_poses(body_target, held_tf.invert())
        hover = Pose(
            tuple(np.asarray(ee_target.position) + [0.0, 0.0, 0.15]), ee_target.orientation
        )
        self.scene.sync()
        path = None
        bodies = self.scene.bodies()
        for solution in ikfast_closest_inverse_kinematics(
            self.scene.robot, world_from_target=hover
        )[:12]:
            path = self.scene.plan_arm(
                goal=solution[:7], bodies=bodies, held=self.scene.wiper_body, held_tf=held_tf
            )
            if path is not None:
                break
        if path is None or not self.motion.follow(path=path, grip=1.0):
            raise ExecutionError("No collision-free wiper placement approach")
        path = self.scene.linear_path(
            start=self.session.arm(),
            target=ee_target,
            bodies=bodies,
            finger_state=0.5,
            max_jump=0.6,
        )
        if path is None or not self.motion.follow(path=path, grip=1.0, final_tol=0.005):
            raise ExecutionError("No collision-free wiper placement descent")
        self.motion.set_gripper(command=0.0)
        retreat = self.scene.linear_path(
            start=self.session.arm(), target=hover, bodies=bodies, max_jump=0.6
        )
        if retreat is None or not self.motion.follow(path=retreat, grip=0.0):
            raise ExecutionError("No collision-free retreat from released wiper")
        return "Wiper physically released at its original native floor start"

    @staticmethod
    def create(*, session: Any, distance: float, heading_offset: float) -> "FloorPrimitives":
        scene = FloorPlanningScene(session=session)
        return FloorPrimitives(
            session=session,
            scene=scene,
            motion=Motion(session=session, scene=scene),
            distance=distance,
            heading_offset=heading_offset,
        )
