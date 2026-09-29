"""New floor skills using existing collision-aware planning and physical execution."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError, Motion
from hitl_pmp.environments.sweep_drawer3d.planning_scene import ArmMath, PlanningScene
from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives


class FloorPrimitives(Primitives):
    """Floor pickup uses the tested generic handle grasp with a learned base stance."""

    narrow_contact: bool = False
    contact_stroke_length: float = 0.132
    contact_step: float = 0.003
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
        from scipy.spatial.transform import Rotation

        for yaw in (self.session.yaw(name="wiper_0"), base[2]):
            for rpy in ((0.0, 0.0, yaw), (0.0, np.pi / 2, yaw), (np.pi / 2, 0.0, yaw)):
                center_offset = Rotation.from_euler("xyz", rpy).apply([0.0, 0.0, 0.17])
                for radius in (0.1, 0.0, 0.2):
                    for height in (0.75, 0.85, 0.65):
                        center = np.array([
                            base[0] + radius * np.cos(base[2]),
                            base[1] + radius * np.sin(base[2]),
                            height,
                        ])
                        target_body = Pose.from_rpy(tuple(center - center_offset), rpy)
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
        raise ExecutionError("No collision-free compact tool transport pose")

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
        del gripper, wiper
        return self.session.gripper() > 0.2 and FloorGrip.has_bilateral_contact(
            session=self.session
        )

    def require_handle(self, *, phase: str) -> None:
        if not self.wiper_in_hand(
            gripper=np.asarray(self.scene.ee_now().position),
            wiper=self.session.position(name="wiper_0"),
        ):
            raise ExecutionError(f"Physical bilateral handle grasp lost during {phase}")

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
        core = self.session.env.unwrapped._object_centric_env
        stalls = 0
        for stroke in range(24):
            if core._ground_fixture.check_in_region(
                self.session.position(name=cube), region, core._robot_env
            ):
                return f"Native target attained after {stroke} checked strokes"
            before = {f"cube_{i}": self.session.position(name=f"cube_{i}").copy() for i in range(5)}
            self._sweep_cube_stroke(
                cube=cube, region=region, distance=distance, heading_offset=heading_offset
            )
            displacement = max(
                float(np.linalg.norm(self.session.position(name=name)[:2] - position[:2]))
                for name, position in before.items()
            )
            stalls = stalls + 1 if displacement < 0.001 else 0
            if stalls >= 3:
                raise ExecutionError("Three consecutive checked strokes made no cube progress")
        raise ExecutionError("Native target not attained within 24 checked strokes")

    def _sweep_cube_stroke(
        self, *, cube: str, region: str, distance: float, heading_offset: float
    ) -> str:
        from pybullet_helpers.geometry import Pose, multiply_poses

        ee = self.scene.ee_now()
        observed_wiper = self.session.position(name="wiper_0")
        if not self.wiper_in_hand(gripper=np.asarray(ee.position), wiper=observed_wiper):
            raise ExecutionError("Sweep requires the physical wiper handle in the hand")
        core = self.session.env.unwrapped._object_centric_env
        box = core.task_config["regions"][region]["ranges"][0]
        target = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        initial = self.session.position(name=cube).copy()
        # The native island ends immediately before the goal-side aisle. Route
        # the blade around that corner rather than diagonally through its width.
        if region == "sweep_region" and initial[1] < target[1] - 0.03:
            target[0] = initial[0]
        elif region == "blocks_init_region" and initial[0] < target[0] - 0.03:
            target[1] = initial[1]
        delta = target - initial[:2]
        length = float(np.linalg.norm(delta))
        if length < 0.02:
            return "Cube already at target center"
        direction = delta / length
        angle = float(np.arctan2(direction[1], direction[0]))
        transverse = np.array([-direction[1], direction[0]])
        behind = 0.0
        for other in (f"cube_{i}" for i in range(5)):
            relative = self.session.position(name=other)[:2] - initial[:2]
            if abs(float(relative @ transverse)) <= (0.025 if self.narrow_contact else 0.16):
                behind = max(behind, -float(relative @ direction))
        wiper_start = initial[:2] - (behind + (0.20 if self.narrow_contact else 0.06)) * direction
        robot_box = core.task_config["regions"]["robot_task_init_region"]["ranges"][0]
        aisle = np.array([(robot_box[0] + robot_box[2]) / 2, (robot_box[1] + robot_box[3]) / 2])
        nominal_bearing = (
            angle + np.pi
            if abs(direction[1]) > abs(direction[0])
            else float(np.arctan2(aisle[1] - wiper_start[1], aisle[0] - wiper_start[0]))
        )
        stance_bearing = nominal_bearing + heading_offset
        stance_angle = float((stance_bearing + 2 * np.pi) % (2 * np.pi) - np.pi)
        stance = (
            float(wiper_start[0] + distance * np.cos(stance_bearing)),
            float(wiper_start[1] + distance * np.sin(stance_bearing)),
            stance_angle,
        )
        self.transport_wiper(target=stance)
        self.require_handle(phase="base transport")
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        bodies = self.scene.bodies(without_cubes=tuple(f"cube_{i}" for i in range(5)))
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        approach = None
        attempts: list[dict[str, Any]] = []
        tool_yaws = (
            (angle, angle + np.pi)
            if self.narrow_contact
            else (angle - np.pi / 2, angle + np.pi / 2)
        )
        for tool_yaw in tool_yaws:
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
                    "approach_paths": 0,
                    "descent_paths": 0,
                    "checked_descents": 0,
                })
                for solution in solutions[:12]:
                    candidate = self.scene.plan_arm(
                        goal=solution[:7],
                        bodies=bodies,
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    if candidate is None:
                        continue
                    attempts[-1]["approach_paths"] += 1
                    candidate_lower = self.scene.linear_path(
                        start=np.asarray(candidate[-1]),
                        target=floor_ee,
                        bodies=bodies,
                        finger_state=0.5,
                        max_jump=0.6,
                    )
                    if candidate_lower is not None:
                        attempts[-1]["descent_paths"] += 1
                    if candidate_lower is None or any(
                        self.scene.in_collision(
                            joints=self.scene.planning_fingers(arm=q, state=0.5),
                            bodies=bodies,
                            held=self.scene.wiper_body,
                            held_tf=held_tf,
                        )
                        for q in candidate_lower
                    ):
                        continue
                    attempts[-1]["checked_descents"] += 1
                    approach = candidate
                    break
                if approach is not None:
                    break
            if approach is not None:
                break
        if approach is None or not self.motion.follow(path=approach, grip=1.0, final_tol=0.003):
            raise ExecutionError(f"No collision-free floor sweep approach: {attempts}")
        self.require_handle(phase="floor approach")
        lower = self.scene.linear_path(
            start=self.session.arm(), target=floor_ee, bodies=bodies, finger_state=0.5, max_jump=0.6
        )
        if lower is None or any(
            self.scene.in_collision(
                joints=self.scene.planning_fingers(arm=q, state=0.5),
                bodies=bodies,
                held=self.scene.wiper_body,
                held_tf=held_tf,
            )
            for q in lower
        ):
            raise ExecutionError("No collision-free floor sweep descent")
        if not self.motion.follow(path=lower, grip=1.0, final_tol=0.005):
            raise ExecutionError("Floor sweep descent did not converge")
        self.require_handle(phase="floor descent")
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
        self.session._write(
            record={
                "kind": "contact_diagnostic",
                "t": self.session.ticks,
                "cube": self.session.position(name=cube).tolist(),
                "wiper": self.session.position(name="wiper_0").tolist(),
                "wiper_yaw": self.session.yaw(name="wiper_0"),
                "target_wiper": wiper_start.tolist(),
                "target_yaw": tool_yaw,
            }
        )
        base_origin = np.array(self.session.base()[:2])
        contact_arm = self.session.arm().copy()
        contact_ended = False
        for progress in np.arange(
            self.contact_step,
            min(length + 0.10, self.contact_stroke_length + self.contact_step),
            self.contact_step,
        ):
            current = self.session.position(name=cube)
            if (
                core._ground_fixture.check_in_region(current, region, core._robot_env)
                or np.linalg.norm(current[:2] - target) < 0.025
            ):
                break
            next_xy = base_origin + progress * direction
            target_base = (float(next_xy[0]), float(next_xy[1]), stance_angle)
            # Replan against observed moving cubes rather than their stale pre-sweep poses.
            base_path = self.scene.plan_base(target=target_base)
            if base_path is None:
                raise ExecutionError("Base route blocked during contact sweep")
            if not self.motion.drive(
                path=base_path, grip=1.0, max_ticks=30, arm=contact_arm, tol=0.0005
            ):
                raise ExecutionError("Contact sweep base did not converge")
            for _ in range(8):
                wiper_now = Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                )
                from scipy.spatial.transform import Rotation

                upright_error = float(
                    np.linalg.norm(Rotation.from_quat(wiper_now.orientation).as_euler("xyz")[:2])
                )
                blade_bottom = self.blade_bottom_height()
                # Keep at least half the native 20-mm cube height in blade contact.
                if (
                    blade_bottom <= float(self.session.position(name=cube)[2])
                    and upright_error < 0.3
                ):
                    break
                ee_now = self.scene.ee_now()
                live_tf = multiply_poses(ee_now.invert(), wiper_now)
                desired = Pose.from_rpy(
                    (
                        wiper_now.position[0],
                        wiper_now.position[1],
                        min(wiper_now.position[2], 0.002),
                    ),
                    (0.0, 0.0, tool_yaw),
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
                    self.require_handle(phase="contact correction")
                    self.session._write(
                        record={
                            "kind": "contact_stroke_ended",
                            "t": self.session.ticks,
                            "path_found": path is not None,
                            "height": wiper_now.position[2],
                            "tilt": upright_error,
                            "blade_bottom": blade_bottom,
                        }
                    )
                    contact_ended = True
                    break
                contact_arm = self.session.arm().copy()
            if contact_ended:
                break
            if not self.wiper_in_hand(
                gripper=np.asarray(self.scene.ee_now().position),
                wiper=self.session.position(name="wiper_0"),
            ):
                raise ExecutionError("Wiper lost during contact sweep")
        if contact_ended:
            retreat_base = np.asarray(self.session.base()[:2]) - 0.03 * direction
            retreat_path = self.scene.plan_base(
                target=(float(retreat_base[0]), float(retreat_base[1]), stance_angle)
            )
            if retreat_path is None or not self.motion.drive(
                path=retreat_path, grip=1.0, max_ticks=80
            ):
                raise ExecutionError("Cannot leave blade/cube contact along a checked route")
            self.require_handle(phase="contact retreat")
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
        if lift is None:
            self.stow_wiper()
        elif not self.motion.follow(path=lift, grip=1.0):
            raise ExecutionError("Checked lift after floor sweep did not converge")
        self.require_handle(phase="stroke lift")
        displacement = float(np.linalg.norm(self.session.position(name=cube)[:2] - initial[:2]))
        return f"Cube displacement {displacement:.3f} m"

    def blade_bottom_height(self) -> float:
        """Highest point of the native blade bottom face in the current pose."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body),
            key=lambda geom: float(model.geom_size[geom][0]),
        )
        vertical = data.geom_xmat[blade].reshape(3, 3)[2]
        half = model.geom_size[blade]
        return float(
            data.geom_xpos[blade][2]
            + abs(vertical[0]) * half[0]
            + abs(vertical[1]) * half[1]
            - vertical[2] * half[2]
        )

    def transport_wiper(self, *, target: tuple[float, float, float]) -> None:
        """Reuse native base paths, checking the actual carried arm/tool along each."""
        from pybullet_helpers.geometry import Pose, multiply_poses

        for stow in (False, True):
            if stow:
                self.stow_wiper()
            arm = self.session.arm().copy()
            self.scene.sync()
            held_tf = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )
            path = self.scene.plan_base(target=target)
            if path is None:
                continue
            clear = True
            for base in path:
                self.scene.sync(base=base)
                if self.scene.in_collision(
                    joints=self.scene.planning_fingers(arm=arm, state=0.5),
                    bodies=self.scene.bodies(),
                    held=self.scene.wiper_body,
                    held_tf=held_tf,
                ):
                    clear = False
                    break
            self.scene.sync()
            if clear:
                if not self.motion.drive(path=path, grip=1.0, arm=arm):
                    raise ExecutionError("Checked carried-tool base motion did not converge")
                self.require_handle(phase="checked base transport")
                return
        raise ExecutionError("No native base path clears the carried arm/tool")

    def stow_wiper(self) -> None:
        self.require_handle(phase="before transport")
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
        self.require_handle(phase="upright transport")

    def place_wiper_at_start(self) -> str:
        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        self.stow_wiper()
        position, orientation = self.session.initial_pose(name="wiper_0")
        stance = self.stances(target=position, where="floor")[0]
        self.motion.drive_to(target=stance, grip=1.0)
        self.require_handle(phase="base transport")
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


class FloorPlanningScene(PlanningScene):
    max_tool_tilt: float = 1.1
    _native_chassis: list[tuple[int, int]] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        super().model_post_init(__context)
        import mujoco
        import pybullet

        model = self.session.mj_model
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_link")
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != chassis or not (
                model.geom_contype[geom] + model.geom_conaffinity[geom]
            ):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise ValueError("Native chassis collision geometry is not a mesh")
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            shape = pybullet.createCollisionShape(
                pybullet.GEOM_MESH,
                vertices=model.mesh_vert[start : start + count].tolist(),
                physicsClientId=self.cid,
            )
            body = pybullet.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=shape, physicsClientId=self.cid
            )
            self._native_chassis.append((geom, body))
        self.sync()

    def bodies(
        self,
        *,
        without_cubes: Iterable[str] = (),
        without_drawer: bool = False,
        without: Iterable[int] = (),
    ) -> set[int]:
        bodies = super().bodies(
            without_cubes=without_cubes, without_drawer=without_drawer, without=without
        )
        bodies.discard(self.chassis_body)
        bodies.update(body for _, body in self._native_chassis if body not in without)
        return bodies

    def planning_fingers(self, *, arm: Any, state: float = 0.0) -> list[float]:
        if state <= 0.2 or self.session.gripper() <= 0.2:
            return super().fingers(arm=arm, state=state)
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        names = (
            "left_driver",
            "right_driver",
            "left_spring_link",
            "right_spring_link",
            "left_follower",
            "right_follower",
        )
        actual = [
            float(
                data.qpos[
                    model.jnt_qposadr[
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_JOINT, "robot_" + name + "_joint"
                        )
                    ]
                ]
            )
            for name in names
        ]
        return [float(value) for value in arm[:7]] + actual

    def in_collision(
        self,
        *,
        joints: Any,
        bodies: set[int],
        held: int | None = None,
        held_tf: Any = None,
        margin: float = 0.0,
    ) -> bool:
        import pybullet

        chassis_bodies = {body for _, body in self._native_chassis} & bodies
        if super().in_collision(
            joints=joints, bodies=bodies - chassis_bodies, held=held, held_tf=held_tf, margin=margin
        ):
            return True
        for body in chassis_bodies:
            contacts = pybullet.getClosestPoints(
                self.robot.robot_id, body, distance=margin, physicsClientId=self.cid
            )
            # Native adjacent chassis/arm-mount bodies are collision-excluded.
            if any(contact[3] != 0 for contact in contacts):
                return True
            if held is not None and pybullet.getClosestPoints(
                held, body, distance=margin, physicsClientId=self.cid
            ):
                return True
        if held is None:
            return False
        # Fingers intentionally contact the tool; the palm and arm must not.
        contacts = pybullet.getClosestPoints(
            held, self.robot.robot_id, distance=0.01, physicsClientId=self.cid
        )
        return any(contact[4] <= 10 and contact[8] < 0.005 for contact in contacts)

    def sync(self, *, base: tuple[float, float, float] | None = None) -> None:
        import mujoco
        from pybullet_helpers.geometry import Pose, multiply_poses

        data, model = self.session.mj_data, self.session.mj_model
        arm_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_gen3/base_link")
        quaternion = data.xquat[arm_root]
        physical_arm_root = Pose(tuple(data.xpos[arm_root]), tuple(quaternion[[1, 2, 3, 0]]))
        bx, by, yaw = self.session.base()
        physical_base = Pose.from_rpy((bx, by, 0.0), (0.0, 0.0, yaw))
        self._sim._base_to_arm_pose = multiply_poses(physical_base.invert(), physical_arm_root)
        super().sync(base=base)
        from pybullet_helpers.geometry import set_pose
        from scipy.spatial.transform import Rotation

        selected_base = (
            physical_base
            if base is None
            else Pose.from_rpy((base[0], base[1], 0.0), (0.0, 0.0, base[2]))
        )
        for geom, body in self._native_chassis:
            native_pose = Pose(
                tuple(data.geom_xpos[geom]),
                tuple(Rotation.from_matrix(data.geom_xmat[geom].reshape(3, 3)).as_quat()),
            )
            planned_pose = multiply_poses(selected_base, physical_base.invert(), native_pose)
            set_pose(body, planned_pose, self.cid)

    def ik(self, *, pose: Any, seed: Any) -> np.ndarray | None:
        import pybullet
        from scipy.spatial.transform import Rotation

        analytic = super().ik(pose=pose, seed=seed)
        if analytic is not None and np.max(np.abs(ArmMath.wrap(delta=analytic - seed))) < 0.4:
            return analytic
        self.robot.set_joints(self.planning_fingers(arm=seed))
        solved = pybullet.calculateInverseKinematics(
            self.robot.robot_id,
            self.robot.end_effector_id,
            targetPosition=pose.position,
            targetOrientation=pose.orientation,
            maxNumIterations=200,
            residualThreshold=1e-6,
            physicsClientId=self.cid,
        )
        candidate = np.asarray(seed) + ArmMath.wrap(delta=np.asarray(solved[:7]) - seed)
        achieved = self.fk(arm=candidate)
        position_error = np.linalg.norm(np.asarray(achieved.position) - pose.position)
        angle_error = (
            Rotation.from_quat(achieved.orientation).inv() * Rotation.from_quat(pose.orientation)
        ).magnitude()
        if self.within_arm_limits(arm=candidate) and position_error < 0.001 and angle_error < 0.01:
            return candidate
        return analytic

    def plan_arm(
        self,
        *,
        goal: Any,
        bodies: set[int],
        start: Any = None,
        base: Any = None,
        held: int | None = None,
        held_tf: Any = None,
    ) -> list[np.ndarray] | None:
        if goal is None:
            return None
        planning_bodies = bodies - {body for _, body in self._native_chassis}
        if held != self.wiper_body or held_tf is None:
            path = super().plan_arm(
                goal=goal,
                bodies=planning_bodies,
                start=start,
                base=base,
                held=held,
                held_tf=held_tf,
            )
            if path is None or any(
                self.in_collision(
                    joints=self.planning_fingers(arm=q), bodies=bodies, held=held, held_tf=held_tf
                )
                for q in path
            ):
                return None
            return path
        from pybullet_helpers.geometry import multiply_poses
        from scipy.spatial.transform import Rotation

        self.sync(base=base)
        q0 = self.session.arm() if start is None else np.asarray(start)
        target = self.fk(arm=goal)
        path = self.linear_path(
            start=q0, target=target, bodies=bodies, finger_state=0.5, max_jump=0.6
        )
        if path is None:
            from pybullet_helpers.geometry import Pose

            origin = self.fk(arm=q0)
            for lift in (0.1, 0.2):
                q = q0.copy()
                candidate = []
                for pose in (
                    Pose(tuple(np.asarray(origin.position) + [0, 0, lift]), origin.orientation),
                    Pose(
                        (target.position[0], target.position[1], origin.position[2] + lift),
                        target.orientation,
                    ),
                    target,
                ):
                    leg = self.linear_path(
                        start=q, target=pose, bodies=bodies, finger_state=0.5, max_jump=0.6
                    )
                    if leg is None:
                        break
                    candidate.extend(leg)
                    q = leg[-1]
                else:
                    path = candidate
                    break
        if path is None:
            path = super().plan_arm(
                goal=goal,
                bodies=planning_bodies,
                start=start,
                base=base,
                held=held,
                held_tf=held_tf,
            )
        if path is None:
            return None
        initial_tool = multiply_poses(self.fk(arm=q0), held_tf)
        goal_tool = multiply_poses(target, held_tf)
        endpoint_tilts = [
            np.arccos(np.clip(Rotation.from_quat(p.orientation).as_matrix()[2, 2], -1, 1))
            for p in (initial_tool, goal_tool)
        ]
        allowed_tilt = max(self.max_tool_tilt, *[t + 0.05 for t in endpoint_tilts])
        previous = np.asarray(q0)
        for waypoint in path:
            waypoint = previous + ArmMath.wrap(delta=waypoint - previous)
            steps = max(1, int(np.ceil(np.max(np.abs(waypoint - previous)) / 0.05)))
            for fraction in np.linspace(0, 1, steps + 1):
                joints = previous + fraction * (waypoint - previous)
                tool_pose = multiply_poses(self.fk(arm=joints), held_tf)
                tilt = np.arccos(
                    np.clip(Rotation.from_quat(tool_pose.orientation).as_matrix()[2, 2], -1, 1)
                )
                if tilt > allowed_tilt or self.in_collision(
                    joints=self.planning_fingers(arm=joints, state=0.5),
                    bodies=bodies,
                    held=held,
                    held_tf=held_tf,
                ):
                    return None
            previous = waypoint
        return path

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
                if self.session.gripper() > 0.2
                and FloorGrip.has_bilateral_contact(session=self.session)
                else []
            ),
        )
        if path is None:
            return None
        return [(float(p.x), float(p.y), float(p.theta())) for p in path]


class FloorGrip:
    @staticmethod
    def has_bilateral_contact(*, session: Any) -> bool:
        import mujoco

        model, data = session.mj_model, session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        wiper_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body}
        pads = set()
        for contact in data.contact:
            if not wiper_geoms.intersection((contact.geom1, contact.geom2)):
                continue
            other = contact.geom2 if contact.geom1 in wiper_geoms else contact.geom1
            pads.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other]))
        return {"robot_left_pad", "robot_right_pad"} <= pads
