"""New floor skills using existing collision-aware planning and physical execution."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from pybullet_helpers.geometry import Pose
from pydantic import Field, PrivateAttr

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError, Motion
from hitl_pmp.environments.sweep_drawer3d.planning_scene import ArmMath, PlanningScene
from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives


class FloorPrimitives(Primitives):
    """Floor pickup uses the tested generic handle grasp with a learned base stance."""

    scene: "FloorPlanningScene"
    narrow_contact: bool = False
    contact_stroke_length: float = 0.10
    contact_step: float = 0.003
    distance: float = Field(default=0.7, ge=0.55, le=0.85)
    heading_offset: float = Field(default=0.0, ge=-np.pi / 12, le=np.pi / 12)
    _ground_clearance_hold: np.ndarray | None = PrivateAttr(default=None)

    def wiper_grasp_offsets(self) -> tuple[float, ...]:
        # A low cross-handle grasp shortens the contact-force lever arm.
        return (-0.09,)

    def wiper_approach_angles(self) -> tuple[float, ...]:
        return np.pi / 2, 1.2, 1.8

    def wiper_grasp_yaw(self, *, axis: np.ndarray) -> float:
        del axis
        base = self.session.base()
        wiper = self.session.position(name="wiper_0")
        return float(np.arctan2(base[1] - wiper[1], base[0] - wiper[0]))

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

        from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene

        home = np.asarray(SweepDrawerScene.HOME)
        if (
            self.scene.plan_arm(
                goal=home,
                bodies=self.scene.bodies(),
                held=self.scene.wiper_body,
                held_tf=held_tf,
                allow_joint_fallback=False,
            )
            is not None
        ):
            return home
        for allow_joint_fallback in (False, True):
            for yaw in (self.session.yaw(name="wiper_0"), base[2]):
                for rpy in ((0.0, 0.0, yaw), (0.0, np.pi / 2, yaw), (np.pi / 2, 0.0, yaw)):
                    center_offset = Rotation.from_euler("xyz", rpy).apply([0.0, 0.0, 0.17])
                    for radius in (0.25, 0.35, 0.1, 0.0):
                        for height in (1.1, 1.0, 0.85, 0.75, 0.65):
                            center = np.array([
                                base[0] + radius * np.cos(base[2]),
                                base[1] + radius * np.sin(base[2]),
                                height,
                            ])
                            target_body = Pose.from_rpy(tuple(center - center_offset), rpy)
                            target = multiply_poses(target_body, held_tf.invert())
                            solutions = ikfast_closest_inverse_kinematics(
                                self.scene.robot, world_from_target=target
                            )
                            for solution in [
                                q for q in solutions if self.scene.within_arm_limits(arm=q[:7])
                            ][:24]:
                                path = self.scene.plan_arm(
                                    goal=solution[:7],
                                    bodies=self.scene.bodies(),
                                    held=self.scene.wiper_body,
                                    held_tf=held_tf,
                                    allow_joint_fallback=allow_joint_fallback,
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
        for stroke in range(36):
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
        if core._ground_fixture.check_in_region(
            self.session.position(name=cube), region, core._robot_env
        ):
            return "Native target attained after 36 checked strokes"
        raise ExecutionError("Native target not attained within 36 checked strokes")

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
        westward_goal_leg = region == "sweep_region" and abs(direction[0]) > abs(direction[1])
        narrow_contact = self.narrow_contact or westward_goal_leg
        transverse = np.array([-direction[1], direction[0]])
        blade_anchor = initial[:2].copy()
        if not narrow_contact:
            projections = [
                float(self.session.position(name=f"cube_{i}")[:2] @ transverse) for i in range(5)
            ]
            if max(projections) - min(projections) <= 0.28:
                midpoint = (max(projections) + min(projections)) / 2
                blade_anchor += (midpoint - float(blade_anchor @ transverse)) * transverse
        behind = 0.0
        for other in (f"cube_{i}" for i in range(5)):
            relative = self.session.position(name=other)[:2] - blade_anchor
            if abs(float(relative @ transverse)) <= (0.025 if narrow_contact else 0.16):
                behind = max(behind, -float(relative @ direction))
        wiper_start = blade_anchor - (behind + (0.20 if narrow_contact else 0.025)) * direction
        # Keep the chassis behind the blade on both legs. Facing the initial
        # aisle during a westward stroke folds the cross-grasp wrist into it.
        # The fixed eastward recovery instead stands north: west of a goal
        # cube is occupied by the native island, even though the blade fits.
        if westward_goal_leg:
            # The right counter blocks a broad-blade approach from the east.
            # A blade-end push from this checked north-side stance fits the aisle.
            nominal_bearing = 2 * np.pi / 3
        elif region == "blocks_init_region" and abs(direction[0]) > abs(direction[1]):
            nominal_bearing = np.pi / 2
        else:
            nominal_bearing = angle + np.pi
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
            (angle + np.pi, angle) if narrow_contact else (angle - np.pi / 2, angle + np.pi / 2)
        )
        for tool_yaw, preserve_tilt in [
            (yaw, preserve) for preserve in (False, True) for yaw in tool_yaws
        ]:
            floor_pose = self.floor_tool_pose(
                xy=wiper_start, yaw=tool_yaw, preserve_tilt=preserve_tilt
            )
            floor_ee = multiply_poses(floor_pose, held_tf.invert())
            for lift_height in (0.14, 0.24):
                hover = Pose(
                    tuple(np.array(floor_ee.position) + [0, 0, lift_height]), floor_ee.orientation
                )
                self.scene.sync()
                solutions = ikfast_closest_inverse_kinematics(
                    self.scene.robot, world_from_target=hover
                )
                solutions = [q for q in solutions if self.scene.within_arm_limits(arm=q[:7])]
                attempts.append({
                    "yaw": tool_yaw,
                    "preserve_tilt": preserve_tilt,
                    "hover": hover.position,
                    "solutions": len(solutions),
                    "approach_paths": 0,
                    "descent_paths": 0,
                    "checked_descents": 0,
                })
                for solution in solutions[:24]:
                    # Reject unreachable floor endpoints before spending time on
                    # a hover approach that cannot complete this descent.
                    candidate_lower = self.scene.floor_descent(
                        start=np.asarray(solution[:7]),
                        target=floor_ee,
                        bodies=bodies,
                        held_tf=held_tf,
                    )
                    if candidate_lower is None:
                        continue
                    candidate = self.scene.plan_arm(
                        goal=solution[:7],
                        bodies=bodies,
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    if candidate is None:
                        continue
                    attempts[-1]["approach_paths"] += 1
                    candidate_lower = self.scene.floor_descent(
                        start=np.asarray(candidate[-1]),
                        target=floor_ee,
                        bodies=bodies,
                        held_tf=held_tf,
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
        if approach is None:
            raise ExecutionError(f"No collision-free floor sweep approach: {attempts}")
        self.session._write(
            record={
                "kind": "floor_approach_selected",
                "t": self.session.ticks,
                "tool_yaw": tool_yaw,
                "preserve_tilt": preserve_tilt,
                "floor_position": floor_pose.position,
                "floor_orientation": floor_pose.orientation,
                "attempts": attempts,
            }
        )
        if not self.motion.follow(
            path=approach,
            grip=1.0,
            final_tol=0.03,
            tick_guard=lambda: self.require_handle(phase="floor approach"),
        ):
            raise ExecutionError(
                "Checked floor approach did not reach the joint tracking tolerance"
            )
        self.require_handle(phase="floor approach")
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        floor_ee = multiply_poses(floor_pose, held_tf.invert())
        lower = self.scene.floor_descent(
            start=self.session.arm(), target=floor_ee, bodies=bodies, held_tf=held_tf
        )
        if lower is None or not self.scene.held_path_clear(
            path=lower,
            start=self.session.arm(),
            bodies=bodies,
            held=self.scene.wiper_body,
            held_tf=held_tf,
            allowed_tilt=self.scene.max_tool_tilt,
        ):
            lower = candidate_lower
        if lower is None or not self.scene.held_path_clear(
            path=lower,
            start=self.session.arm(),
            bodies=bodies,
            held=self.scene.wiper_body,
            held_tf=held_tf,
            allowed_tilt=self.scene.max_tool_tilt,
        ):
            raise ExecutionError("No collision-free floor sweep descent from observed grasp")
        if not self.motion.follow(
            path=lower,
            grip=1.0,
            final_tol=0.005,
            tick_guard=lambda: self.require_handle(phase="floor descent"),
        ):
            raise ExecutionError("Floor sweep descent did not converge")
        self.require_handle(phase="floor descent")
        # Close the loop on actual blade overlap, bounded by native floor clearance.
        for _ in range(6):
            height = self.blade_bottom_height(cube=cube, narrow=narrow_contact)
            center_height = self.cube_contact_ceiling(cube=cube)
            if height <= center_height:
                break
            descent = min(
                height - center_height + 0.002, max(self.blade_minimum_height() - 0.001, 0.0), 0.02
            )
            if descent < 0.0001:
                if self.level_blade(bodies=bodies):
                    continue
                break
            ee = self.scene.ee_now()
            live_tf = multiply_poses(
                ee.invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )
            correction = Pose(tuple(np.asarray(ee.position) - [0.0, 0.0, descent]), ee.orientation)
            path = self.scene.floor_descent(
                start=self.session.arm(), target=correction, bodies=bodies, held_tf=live_tf
            )
            if path is None or not self.motion.follow(
                path=path,
                grip=1.0,
                max_ticks=30,
                final_tol=0.005,
                tick_guard=lambda: self.require_handle(phase="initial blade-height correction"),
            ):
                if self.level_blade(bodies=bodies):
                    continue
                break
        if self.blade_bottom_height(cube=cube, narrow=narrow_contact) > self.cube_contact_ceiling(
            cube=cube
        ):
            raise ExecutionError("Observed blade edge does not overlap the target cube height")
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
            ground_corrected = False
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
            carried_tf = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )
            for waypoint in base_path:
                self.scene.sync(base=waypoint)
                self.scene.capture_path_rejections = True
                carried_clear = self.scene.held_path_clear(
                    path=[contact_arm],
                    start=self.session.arm(),
                    bodies=bodies,
                    held=self.scene.wiper_body,
                    held_tf=carried_tf,
                    allowed_tilt=self.scene.max_tool_tilt,
                )
                self.scene.capture_path_rejections = False
                if not carried_clear:
                    self.session._write(
                        record={
                            "kind": "contact_path_rejection",
                            "t": self.session.ticks,
                            "actual_arm": self.session.arm().tolist(),
                            "target_arm": contact_arm.tolist(),
                            "waypoint": list(waypoint),
                            "held_position": list(carried_tf.position),
                            "held_orientation": list(carried_tf.orientation),
                            "checker": self.scene._last_path_rejection,
                            "native_qpos": self.session.mj_data.qpos.tolist(),
                        }
                    )
                    self.scene.sync()
                    rejection = self.scene._last_path_rejection or {}
                    if (rejection.get("collision") or {}).get("reason") == "native_tool_ground":
                        if self.raise_blade_clear_of_ground(bodies=bodies):
                            assert self._ground_clearance_hold is not None
                            contact_arm = self._ground_clearance_hold.copy()
                            ground_corrected = True
                            break
                        # End and unload this contact stroke before a stale arm
                        # hold target would drive the blade into the floor.
                        contact_ended = True
                        break
                    raise ExecutionError("Carried arm/tool route blocked during contact sweep")
            self.scene.sync()
            if contact_ended:
                break
            if ground_corrected:
                continue
            if not self.motion.drive(
                path=base_path, grip=1.0, max_ticks=30, arm=contact_arm, tol=0.0005
            ):
                raise ExecutionError("Contact sweep base did not converge")
            self.require_handle(phase="contact base step")
            for _ in range(8):
                wiper_now = Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                )
                from scipy.spatial.transform import Rotation

                upright_error = float(
                    np.linalg.norm(Rotation.from_quat(wiper_now.orientation).as_euler("xyz")[:2])
                )
                blade_bottom = self.blade_bottom_height(cube=cube, narrow=narrow_contact)
                # Require positive native cube/blade overlap with a 2-mm margin.
                if blade_bottom <= self.cube_contact_ceiling(cube=cube):
                    break
                if self.blade_minimum_height() <= 0.0011:
                    self.session._write(
                        record={
                            "kind": "contact_stroke_ended",
                            "t": self.session.ticks,
                            "reason": "no downward clearance",
                            "blade_bottom": blade_bottom,
                        }
                    )
                    contact_ended = True
                    break
                ee_now = self.scene.ee_now()
                # Control the measured blade overlap without requiring an upright
                # handle during contact; a leaning grasp can still sweep correctly.
                correction = Pose(
                    tuple(
                        np.asarray(ee_now.position)
                        - [
                            0.0,
                            0.0,
                            min(
                                blade_bottom - self.cube_contact_ceiling(cube=cube) + 0.002,
                                0.02,
                                max(self.blade_minimum_height() - 0.001, 0.0),
                            ),
                        ]
                    ),
                    ee_now.orientation,
                )
                path = self.scene.linear_path(
                    start=self.session.arm(),
                    target=correction,
                    bodies=bodies,
                    finger_state=0.5,
                    max_jump=0.6,
                )
                live_tf = multiply_poses(ee_now.invert(), wiper_now)
                if path is not None and not self.scene.held_path_clear(
                    path=path,
                    start=self.session.arm(),
                    bodies=bodies,
                    held=self.scene.wiper_body,
                    held_tf=live_tf,
                    allowed_tilt=self.scene.max_tool_tilt,
                ):
                    path = None
                if path is None or not self.motion.follow(
                    path=path,
                    grip=1.0,
                    tol=0.005,
                    final_tol=0.003,
                    max_ticks=30,
                    tick_guard=lambda: self.require_handle(phase="contact correction"),
                ):
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
        retreat_origin = np.asarray(self.session.base()[:2])
        for retreat in (0.03, 0.06, 0.10):
            retreat_base = retreat_origin - retreat * direction
            retreat_path = self.scene.plan_base(
                target=(float(retreat_base[0]), float(retreat_base[1]), stance_angle)
            )
            if retreat_path is None:
                raise ExecutionError("Cannot unload blade/cube contact along a checked route")
            retreat_arm = self.session.arm().copy()
            retreat_tf = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )
            for waypoint in retreat_path:
                self.scene.sync(base=waypoint)
                if not self.scene.held_path_clear(
                    path=[retreat_arm],
                    start=retreat_arm,
                    bodies=bodies,
                    held=self.scene.wiper_body,
                    held_tf=retreat_tf,
                    allowed_tilt=self.scene.max_tool_tilt,
                ):
                    self.scene.sync()
                    raise ExecutionError("Carried arm/tool route blocked during contact retreat")
            self.scene.sync()
            if not self.motion.drive(path=retreat_path, grip=1.0, max_ticks=80, arm=retreat_arm):
                raise ExecutionError("Checked contact retreat did not converge")
            self.require_handle(phase="contact retreat")
            if not self.wiper_loaded_by_cube():
                break
        if self.wiper_loaded_by_cube():
            raise ExecutionError("Blade remains loaded by cubes after checked retreat")
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
        lift_tf = multiply_poses(
            current_ee.invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        if lift is not None and not self.scene.held_path_clear(
            path=lift,
            start=self.session.arm(),
            bodies=bodies,
            held=self.scene.wiper_body,
            held_tf=lift_tf,
            allowed_tilt=self.scene.max_tool_tilt,
        ):
            lift = None
        if lift is None:
            self.stow_wiper()
        elif not self.motion.follow(path=lift, grip=1.0):
            raise ExecutionError("Checked lift after floor sweep did not converge")
        self.require_handle(phase="stroke lift")
        displacement = float(np.linalg.norm(self.session.position(name=cube)[:2] - initial[:2]))
        return f"Cube displacement {displacement:.3f} m"

    def wiper_loaded_by_cube(self) -> bool:
        """Use native contact pairs rather than assuming the lowest corner is grounded."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        wiper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        cubes = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"cube_{i}") for i in range(5)}
        for contact in data.contact[: data.ncon]:
            a, b = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            if (a == wiper and b in cubes) or (b == wiper and a in cubes):
                return True
        return False

    def raise_blade_clear_of_ground(self, *, bodies: set[int]) -> bool:
        """Lift through a checked path and verify actual clearance before contact."""
        from pybullet_helpers.geometry import multiply_poses

        ee = self.scene.ee_now()
        held = multiply_poses(
            ee.invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        target = Pose(tuple(np.asarray(ee.position) + [0.0, 0.0, 0.02]), ee.orientation)
        path = self.scene.floor_descent(
            start=self.session.arm(), target=target, bodies=bodies, held_tf=held
        )
        execution_terminated = path is not None and self.motion.follow(
            path=path,
            grip=1.0,
            tol=0.03,
            final_tol=0.005,
            max_ticks=180,
            tick_guard=lambda: self.require_handle(phase="ground-clearance correction"),
            stop_condition=lambda: self.blade_minimum_height() >= 0.005,
        )
        # The servo has a load-dependent joint residual. The correction's goal
        # is measured floor clearance, not an exact unloaded joint posture.
        success = path is not None and self.blade_minimum_height() >= 0.005
        self._ground_clearance_hold = self.session.arm().copy() if success else None
        self.session._write(
            record={
                "kind": "ground_clearance_correction",
                "t": self.session.ticks,
                "path_found": path is not None,
                "clearance_attained": success,
                "execution_terminated": execution_terminated,
                "joint_target_reached": bool(path) and float(np.max(np.abs(
                    ArmMath.wrap(delta=np.asarray(path[-1]) - self.session.arm())
                ))) < 0.005,
                "path_waypoints": None if path is None else len(path),
                "goal_arm": None if not path else np.asarray(path[-1]).tolist(),
                "actual_arm": self.session.arm().tolist(),
                "max_joint_residual": None
                if not path
                else float(
                    np.max(np.abs(ArmMath.wrap(delta=np.asarray(path[-1]) - self.session.arm())))
                ),
                "blade_minimum_height": self.blade_minimum_height(),
            }
        )
        return success

    def level_blade(self, *, bodies: set[int]) -> bool:
        """Rotate a small amount only after unloading the blade from all cubes."""
        if self.wiper_loaded_by_cube():
            self.session._write(
                record={
                    "kind": "blade_leveling_skipped",
                    "t": self.session.ticks,
                    "reason": "cube_loaded",
                }
            )
            return False
        from itertools import product

        import mujoco
        from pybullet_helpers.geometry import multiply_poses
        from scipy.spatial.transform import Rotation

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
            key=lambda g: float(model.geom_size[g][0]),
        )
        observed = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
        yaw = float(np.arctan2(observed.as_matrix()[1, 0], observed.as_matrix()[0, 0]))
        delta = (Rotation.from_euler("z", yaw) * observed.inv()).as_rotvec()
        angle = float(np.linalg.norm(delta))
        if angle < 0.01:
            self.session._write(
                record={
                    "kind": "blade_leveling_skipped",
                    "t": self.session.ticks,
                    "reason": "below_turn_threshold",
                    "turn_needed": angle,
                }
            )
            return False
        tool = Pose(tuple(data.xpos[body]), tuple(observed.as_quat()))
        ee = self.scene.ee_now()
        held_tf = multiply_poses(ee.invert(), tool)
        geom_rotation = data.geom_xmat[blade].reshape(3, 3)
        world = np.array([
            data.geom_xpos[blade] + geom_rotation @ (model.geom_size[blade] * signs)
            for signs in product((-1.0, 1.0), repeat=3)
        ])
        local = (world - data.xpos[body]) @ observed.as_matrix()
        support = int(np.argmin(world[:, 2]))
        for turn in (0.08, 0.04, 0.02):
            rotation = Rotation.from_rotvec(delta * min(1.0, turn / angle)) * observed
            position = world[support] - rotation.apply(local[support])
            corners = rotation.apply(local) + position
            position[2] += max(0.0, 0.001 - float(corners[:, 2].min()))
            target = multiply_poses(
                Pose(tuple(position), tuple(rotation.as_quat())), held_tf.invert()
            )
            path = self.scene.floor_descent(
                start=self.session.arm(), target=target, bodies=bodies, held_tf=held_tf
            )
            before_tilt = float(np.arccos(np.clip(observed.as_matrix()[2, 2], -1.0, 1.0)))
            self.session._write(
                record={
                    "kind": "blade_leveling_candidate",
                    "t": self.session.ticks,
                    "turn": turn,
                    "before_tilt": before_tilt,
                    "path_found": path is not None,
                    "path_waypoints": None if path is None else len(path),
                    "goal_arm": None if not path else np.asarray(path[-1]).tolist(),
                    "target_tool_position": position.tolist(),
                    "target_tool_orientation": rotation.as_quat().tolist(),
                }
            )
            if path is None:
                continue
            converged = None
            execution_error = None
            try:
                converged = self.motion.follow(
                    path=path,
                    grip=1.0,
                    tol=0.005,
                    final_tol=0.005,
                    max_ticks=30,
                    tick_guard=lambda: self.require_handle(phase="blade leveling"),
                )
                self.require_handle(phase="blade leveling")
            except Exception as error:
                execution_error = repr(error)
                raise
            finally:
                after = data.xmat[body].reshape(3, 3)
                after_tilt = float(np.arccos(np.clip(after[2, 2], -1.0, 1.0)))
                residual = (
                    ArmMath.wrap(delta=np.asarray(path[-1]) - self.session.arm()) if path else None
                )
                self.session._write(
                    record={
                        "kind": "blade_leveling_outcome",
                        "t": self.session.ticks,
                        "turn": turn,
                        "before_tilt": before_tilt,
                        "after_tilt": after_tilt,
                        "converged": converged,
                        "max_joint_residual": None
                        if residual is None
                        else float(np.max(np.abs(residual))),
                        "actual_arm": self.session.arm().tolist(),
                        "error": execution_error,
                    }
                )
            if not converged:
                return False
            return after_tilt < before_tilt - 0.002
        return False

    def cube_contact_ceiling(self, *, cube: str) -> float:
        """Highest blade edge that still overlaps the native cube by two millimeters."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
        geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == body)
        vertical = data.geom_xmat[geom].reshape(3, 3)[2]
        return float(data.geom_xpos[geom][2] + np.abs(vertical) @ model.geom_size[geom] - 0.002)

    def blade_minimum_height(self) -> float:
        """Exact native box support height above the floor."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
            key=lambda g: float(model.geom_size[g][0]),
        )
        vertical = data.geom_xmat[blade].reshape(3, 3)[2]
        return float(data.geom_xpos[blade][2] - np.abs(vertical) @ model.geom_size[blade])

    def floor_tool_pose(self, *, xy: np.ndarray, yaw: float, preserve_tilt: bool = True) -> Pose:
        """Preserve observed tilt and seat the native blade at its actual support height."""
        from itertools import product

        import mujoco
        from scipy.spatial.transform import Rotation

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
            key=lambda g: float(model.geom_size[g][0]),
        )
        observed = data.xmat[body].reshape(3, 3)
        observed_yaw = float(np.arctan2(observed[1, 0], observed[0, 0]))
        target_rotation = (
            Rotation.from_rotvec([0.0, 0.0, yaw - observed_yaw]).as_matrix() @ observed
        )
        tilt = float(np.arccos(np.clip(target_rotation[2, 2], -1.0, 1.0)))
        if not preserve_tilt or tilt > self.scene.max_tool_tilt:
            target_rotation = Rotation.from_euler("z", yaw).as_matrix()
        blade_rotation = data.geom_xmat[blade].reshape(3, 3)
        world_corners = np.array([
            data.geom_xpos[blade] + blade_rotation @ (model.geom_size[blade] * signs)
            for signs in product((-1.0, 1.0), repeat=3)
        ])
        body_corners = (world_corners - data.xpos[body]) @ observed
        bottom = float((body_corners @ target_rotation.T)[:, 2].min())
        return Pose(
            (float(xy[0]), float(xy[1]), 0.001 - bottom),
            tuple(Rotation.from_matrix(target_rotation).as_quat()),
        )

    def blade_bottom_height(self, *, cube: str | None = None, narrow: bool = False) -> float:
        """Native bottom-edge height over the selected cube lateral footprint."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max(
            (geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body),
            key=lambda geom: float(model.geom_size[geom][0]),
        )
        rotation = data.geom_xmat[blade].reshape(3, 3)
        vertical = rotation[2]
        half = model.geom_size[blade]
        if cube is not None:
            from itertools import product

            cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube)
            cube_geom = next(g for g in range(model.ngeom) if model.geom_bodyid[g] == cube_body)
            cube_rotation = data.geom_xmat[cube_geom].reshape(3, 3)
            cube_corners = np.array([
                data.geom_xpos[cube_geom] + cube_rotation @ (model.geom_size[cube_geom] * signs)
                for signs in product((-1.0, 1.0), repeat=3)
            ])
            local = (cube_corners - data.geom_xpos[blade]) @ rotation
            lateral = 1 if narrow else 0
            front_axis = 1 - lateral
            lower = max(-half[lateral], float(local[:, lateral].min()))
            upper = min(half[lateral], float(local[:, lateral].max()))
            if lower <= upper:
                front = float(np.sign(local[:, front_axis].mean())) * half[front_axis]
                return float(
                    data.geom_xpos[blade][2]
                    + max(vertical[lateral] * lower, vertical[lateral] * upper)
                    + vertical[front_axis] * front
                    - vertical[2] * half[2]
                )
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
                self.session._write(
                    record={
                        "kind": "transport_rejected",
                        "stowed": stow,
                        "reason": "native_base_path",
                        "target": target,
                    }
                )
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
                    self.session._write(
                        record={
                            "kind": "transport_rejected",
                            "stowed": stow,
                            "reason": "carried_collision",
                            "base": base,
                            "target": target,
                        }
                    )
                    clear = False
                    break
            self.scene.sync()
            if clear:
                if not self.motion.drive(
                    path=path,
                    grip=1.0,
                    arm=arm,
                    max_ticks=2500,
                    max_translation_step=0.01,
                    max_yaw_step=0.005,
                    translation_step_change=0.002,
                    yaw_step_change=0.001,
                    tick_guard=lambda: self.require_handle(phase="checked base transport step"),
                ):
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
        self.transport_wiper(target=stance)
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
    capture_path_rejections: bool = False
    _last_path_rejection: dict[str, Any] | None = PrivateAttr(default=None)
    _last_collision_rejection: dict[str, Any] | None = PrivateAttr(default=None)
    _native_chassis: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _distance_data: Any = PrivateAttr(default=None)
    _native_tool_corners: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        super().model_post_init(__context)
        from itertools import product

        import mujoco
        import pybullet
        from scipy.spatial.transform import Rotation

        model = self.session.mj_model
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        corners = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] != tool or not (
                model.geom_contype[geom] + model.geom_conaffinity[geom]
            ):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Native Simple tool floor check requires box geometry")
            rotation = Rotation.from_quat(model.geom_quat[geom][[1, 2, 3, 0]])
            corners.extend(
                model.geom_pos[geom]
                + rotation.apply([
                    model.geom_size[geom] * signs for signs in product((-1.0, 1.0), repeat=3)
                ])
            )
        self._native_tool_corners = np.asarray(corners)
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

    def collision_pair_detail(self, *, contact: Any) -> dict[str, Any]:
        """Describe an already queried Bullet pair without changing planning state."""
        import pybullet

        labels = {body: name for name, body in self._sim._static_colliders.items()}
        labels.update({body: f"native_chassis_{geom}" for geom, body in self._native_chassis})
        labels[self.robot.robot_id] = "robot"
        labels[self.wiper_body] = "wiper"
        detail: dict[str, Any] = {"distance": float(contact[8])}
        for side, body, link in (("a", contact[1], contact[3]), ("b", contact[2], contact[4])):
            detail[f"body_{side}"] = int(body)
            detail[f"name_{side}"] = labels.get(body, str(body))
            detail[f"link_{side}"] = int(link)
            if body == self.robot.robot_id and link >= 0:
                detail[f"link_name_{side}"] = pybullet.getJointInfo(
                    body, link, physicsClientId=self.cid
                )[12].decode()
        return detail

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

        if self.capture_path_rejections:
            self._last_collision_rejection = None
        if held == self.wiper_body and held_tf is not None:
            from pybullet_helpers.geometry import multiply_poses
            from scipy.spatial.transform import Rotation

            tool_pose = multiply_poses(self.fk(arm=np.asarray(joints[:7])), held_tf)
            minimum = float(
                (
                    Rotation.from_quat(tool_pose.orientation).apply(self._native_tool_corners)
                    + tool_pose.position
                )[:, 2].min()
            )
            # Native soft contacts may place the observed tool fractionally below
            # the plane. Permit escape from that state, never a deeper path.
            observed = Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            )
            observed_minimum = float(
                (
                    Rotation.from_quat(observed.orientation).apply(self._native_tool_corners)
                    + observed.position
                )[:, 2].min()
            )
            floor_limit = min(0.0, observed_minimum) - 1e-6
            if minimum < floor_limit:
                if self.capture_path_rejections:
                    self._last_collision_rejection = dict(
                        reason="native_tool_ground",
                        minimum_height=minimum,
                        minimum_allowed_height=floor_limit,
                    )
                return True
        chassis_bodies = {body for _, body in self._native_chassis} & bodies
        arm = np.asarray(joints[:7])
        # Collision geometry of an observed physical pose is separate from
        # admissibility of a proposed arm target. Native soft-limit contact may
        # place the observed arm slightly beyond the nominal bound; every IK
        # candidate still goes through strict within_arm_limits filtering.
        measured_roundoff = (
            not self.within_arm_limits(arm=arm) and np.max(np.abs(arm - self.session.arm())) <= 1e-7
        )
        if measured_roundoff:
            from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object

            blocked = check_collisions_with_held_object(
                self.robot,
                bodies - chassis_bodies,
                self.cid,
                held,
                held_tf,
                joints,
                distance_threshold=margin,
            )
        else:
            blocked = super().in_collision(
                joints=joints,
                bodies=bodies - chassis_bodies,
                held=held,
                held_tf=held_tf,
                margin=margin,
            )
        if blocked:
            if self.capture_path_rejections:
                pairs = []
                for first, second in self.robot.self_collision_link_ids:
                    pairs.extend(
                        pybullet.getClosestPoints(
                            self.robot.robot_id,
                            self.robot.robot_id,
                            distance=margin,
                            linkIndexA=first,
                            linkIndexB=second,
                            physicsClientId=self.cid,
                        )
                    )
                for body in bodies - chassis_bodies:
                    pairs.extend(
                        pybullet.getClosestPoints(
                            self.robot.robot_id,
                            body,
                            distance=margin,
                            physicsClientId=self.cid,
                        )
                    )
                    if held is not None:
                        pairs.extend(
                            pybullet.getClosestPoints(
                                held,
                                body,
                                distance=margin,
                                physicsClientId=self.cid,
                            )
                        )
                self._last_collision_rejection = {
                    "planning_joints": np.asarray(joints).tolist(),
                    "reason": "upstream",
                    "within_arm_limits": self.within_arm_limits(arm=arm),
                    "observed_roundoff": bool(measured_roundoff),
                    "pairs": [self.collision_pair_detail(contact=pair) for pair in pairs],
                }
            return True
        for body in chassis_bodies:
            contacts = pybullet.getClosestPoints(
                self.robot.robot_id, body, distance=margin, physicsClientId=self.cid
            )
            # Native adjacent chassis/arm-mount bodies are collision-excluded.
            chassis_geom = next(geom for geom, proxy in self._native_chassis if proxy == body)
            for contact in contacts:
                link = contact[3]
                if link == 0:
                    continue
                padding = float(
                    pybullet.getDynamicsInfo(self.robot.robot_id, link, physicsClientId=self.cid)[
                        11
                    ]
                ) + float(pybullet.getDynamicsInfo(body, -1, physicsClientId=self.cid)[11])
                native_distance = None
                if contact[8] >= -padding - 1e-5:
                    native_distance = self.native_chassis_distance(
                        link=link,
                        chassis_geom=chassis_geom,
                        joints=joints,
                        distance_threshold=margin,
                    )
                    if native_distance is not None and native_distance > margin:
                        continue
                if self.capture_path_rejections:
                    if native_distance is None:
                        native_distance = self.native_chassis_distance(
                            link=link,
                            chassis_geom=chassis_geom,
                            joints=joints,
                            distance_threshold=margin,
                        )
                    self._last_collision_rejection = {
                        "planning_joints": np.asarray(joints).tolist(),
                        "reason": "arm_chassis",
                        "pair": self.collision_pair_detail(contact=contact),
                        "chassis_geom": int(chassis_geom),
                        "padding": padding,
                        "native_distance": native_distance,
                    }
                return True
            held_contacts = (
                pybullet.getClosestPoints(held, body, distance=margin, physicsClientId=self.cid)
                if held is not None
                else ()
            )
            if held_contacts:
                if self.capture_path_rejections:
                    self._last_collision_rejection = {
                        "planning_joints": np.asarray(joints).tolist(),
                        "reason": "held_chassis",
                        "pair": self.collision_pair_detail(contact=held_contacts[0]),
                    }
                return True
        if held is None:
            return False
        # Fingers intentionally contact the tool; the palm and arm must not.
        contacts = pybullet.getClosestPoints(
            held, self.robot.robot_id, distance=0.01, physicsClientId=self.cid
        )
        for contact in contacts:
            if contact[4] <= 10 and contact[8] < 0.005:
                if self.capture_path_rejections:
                    self._last_collision_rejection = {
                        "planning_joints": np.asarray(joints).tolist(),
                        "reason": "held_arm",
                        "pair": self.collision_pair_detail(contact=contact),
                    }
                return True
        return False

    def native_chassis_distance(
        self, *, link: int, chassis_geom: int, joints: Any, distance_threshold: float = 0.0
    ) -> float | None:
        """Query capped native clearance at the collision threshold, without mutation."""
        import mujoco
        import pybullet

        model = self.session.mj_model
        if link < 0:
            return None
        name = pybullet.getJointInfo(self.robot.robot_id, link, physicsClientId=self.cid)[
            12
        ].decode()
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_" + name)
        if body >= 0:
            bodies = {body}
        elif name in {
            "robotiq_arg2f_base_link",
            *(f"{side}_{part}" for side in ("left", "right") for part in (
                "outer_knuckle", "outer_finger", "inner_finger",
                "inner_finger_pad", "inner_knuckle",
            )),
        }:
            # The URDF and native Robotiq assets name their articulated links
            # differently. Query the WHOLE native gripper conservatively, not
            # an assumed one-to-one alias or a missing-body collision exemption.
            root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_base")
            if root < 0:
                return None
            bodies = {root}
            for candidate in range(root + 1, model.nbody):
                if model.body_parentid[candidate] in bodies:
                    bodies.add(candidate)
        else:
            return None
        geoms = [
            g
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in bodies
            and model.geom_contype[g] + model.geom_conaffinity[g] > 0
        ]
        if not geoms:
            return None
        if self._distance_data is None:
            self._distance_data = mujoco.MjData(model)
        data = self._distance_data
        data.qpos[:] = self.session.mj_data.qpos
        for number, value in enumerate(joints[:7], start=1):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot_joint_{number}")
            data.qpos[model.jnt_qposadr[joint]] = value
        names = (
            "left_driver",
            "right_driver",
            "left_spring_link",
            "right_spring_link",
            "left_follower",
            "right_follower",
        )
        for name, value in zip(names, joints[7:], strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_" + name + "_joint")
            data.qpos[model.jnt_qposadr[joint]] = value
        # Arm/chassis distance is invariant to their shared rigid base transform.
        mujoco.mj_kinematics(model, data)
        # A larger positive search cap can yield an ambiguous zero for separated
        # convex meshes in pinned MuJoCo 3.3.7. We need only the threshold test:
        # a positive capped return proves separation, not an exact distance.
        distance_cap = max(0.0, distance_threshold) + 1e-6
        return min(
            float(mujoco.mj_geomDistance(model, data, geom, chassis_geom, distance_cap, None))
            for geom in geoms
        )

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
        return super().ik(pose=pose, seed=seed)

    def plan_arm(
        self,
        *,
        goal: Any,
        bodies: set[int],
        start: Any = None,
        base: Any = None,
        held: int | None = None,
        held_tf: Any = None,
        allow_joint_fallback: bool = True,
    ) -> list[np.ndarray] | None:
        if goal is None or not self.within_arm_limits(arm=goal):
            return None
        if held != self.wiper_body or held_tf is None:
            path = self.native_joint_path(
                goal=goal,
                bodies=bodies,
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
        if self.in_collision(
            joints=self.planning_fingers(arm=goal, state=0.5),
            bodies=bodies,
            held=held,
            held_tf=held_tf,
        ):
            return None
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
        if path is None and not allow_joint_fallback:
            return None
        if path is None:
            path = self.native_joint_path(
                goal=goal,
                bodies=bodies,
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
        if self.held_path_clear(
            path=path,
            start=q0,
            bodies=bodies,
            held=held,
            held_tf=held_tf,
            allowed_tilt=allowed_tilt,
        ):
            return path
        if not allow_joint_fallback:
            return None
        alternative = self.native_joint_path(
            goal=goal, bodies=bodies, start=start, base=base, held=held, held_tf=held_tf
        )
        if alternative is not None and self.held_path_clear(
            path=alternative,
            start=q0,
            bodies=bodies,
            held=held,
            held_tf=held_tf,
            allowed_tilt=allowed_tilt,
        ):
            return alternative
        return None

    def floor_descent(
        self, *, start: Any, target: Any, bodies: set[int], held_tf: Any
    ) -> list[np.ndarray] | None:
        """Keep native collision checks when Cartesian IK changes branch near the floor."""
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        path = self.linear_path(
            start=np.asarray(start),
            target=target,
            bodies=bodies,
            finger_state=0.5,
            max_jump=0.6,
        )
        if path is not None and self.held_path_clear(
            path=path,
            start=start,
            bodies=bodies,
            held=self.wiper_body,
            held_tf=held_tf,
            allowed_tilt=self.max_tool_tilt,
        ):
            return path
        self.sync()
        solutions = ikfast_closest_inverse_kinematics(self.robot, world_from_target=target)
        candidates = [np.asarray(q[:7]) for q in solutions if self.within_arm_limits(arm=q[:7])]
        candidates.sort(key=lambda q: float(np.linalg.norm(q - start)))
        checked_candidates = 0
        for goal in candidates:
            if self.in_collision(
                joints=self.planning_fingers(arm=goal, state=0.5),
                bodies=bodies,
                held=self.wiper_body,
                held_tf=held_tf,
            ):
                continue
            # The budget limits path searches, not collision-rejected IK branches.
            # Otherwise eight blocked branches can hide a reachable ninth one.
            checked_candidates += 1
            if checked_candidates > 8:
                break
            path = self.native_joint_path(
                goal=goal,
                start=start,
                bodies=bodies,
                held=self.wiper_body,
                held_tf=held_tf,
            )
            if path is not None and self.held_path_clear(
                path=path,
                start=start,
                bodies=bodies,
                held=self.wiper_body,
                held_tf=held_tf,
                allowed_tilt=self.max_tool_tilt,
            ):
                return path
        return None

    def native_joint_path(
        self,
        *,
        goal: Any,
        bodies: set[int],
        start: Any = None,
        base: Any = None,
        held: int | None = None,
        held_tf: Any = None,
    ) -> list[np.ndarray] | None:
        """Reuse BiRRT with native joint limits and chassis/held-object constraints."""
        from pybullet_helpers.motion_planning import run_motion_planning

        if not self.within_arm_limits(arm=goal):
            return None
        self.sync(base=base)
        initial = self.session.arm() if start is None else np.asarray(start)
        planning_bodies = bodies - {body for _, body in self._native_chassis}
        path = run_motion_planning(
            self.robot,
            self.planning_fingers(arm=initial, state=0.5 if held is not None else 0.0),
            self.planning_fingers(arm=goal, state=0.5 if held is not None else 0.0),
            collision_bodies=planning_bodies,
            seed=0,
            physics_client_id=self.cid,
            held_object=held,
            base_link_to_held_obj=held_tf,
            additional_state_constraint_fn=lambda q: (
                not self.in_collision(joints=q, bodies=bodies, held=held, held_tf=held_tf)
            ),
        )
        if path is None:
            return None
        return [np.asarray(q[:7], dtype=float) for q in path]

    def held_path_clear(
        self,
        *,
        path: Any,
        start: Any,
        bodies: set[int],
        held: int,
        held_tf: Any,
        allowed_tilt: float,
    ) -> bool:
        from pybullet_helpers.geometry import multiply_poses
        from scipy.spatial.transform import Rotation

        if self.capture_path_rejections:
            self._last_path_rejection = None
        previous = np.asarray(start)
        for waypoint_index, waypoint in enumerate(path):
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
                    if self.capture_path_rejections:
                        self._last_path_rejection = {
                            "waypoint_index": waypoint_index,
                            "fraction": float(fraction),
                            "tilt": float(tilt),
                            "allowed_tilt": float(allowed_tilt),
                            "joints": joints.tolist(),
                            "reason": "tilt" if tilt > allowed_tilt else "collision",
                            "collision": (
                                None if tilt > allowed_tilt else self._last_collision_rejection
                            ),
                        }
                    return False
            previous = waypoint
        return True

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
