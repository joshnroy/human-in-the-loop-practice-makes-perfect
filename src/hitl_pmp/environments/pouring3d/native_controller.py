"""Bounded Cartesian robot skills through native KinDER motors.

IK writes only a separate MjData. Robot skills use motor actions and contacts.
Human intervention is handled separately by Environment.reset_movables().
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class PourParameters(BaseModel):
    offset_x: float = Field(default=0.0, ge=-0.4, le=0.4, allow_inf_nan=False)
    offset_y: float = Field(default=0.0, ge=-0.4, le=0.4, allow_inf_nan=False)
    angle: float = Field(default=145.0, ge=0, le=150, allow_inf_nan=False)
    tilt_seconds: float = Field(default=10.0, ge=2, le=15, allow_inf_nan=False)
    hold_seconds: float = Field(default=4.0, ge=0.2, le=8, allow_inf_nan=False)


class NativePouringController(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Any
    scene: Any
    model: Any
    data: Any
    scratch: Any
    arm_q: np.ndarray
    base_q: np.ndarray
    arm_v: np.ndarray
    last_q: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    site: int
    frequency: float
    home: np.ndarray
    tray_home: np.ndarray
    tray_home_rotation: Rotation = Field(default_factory=Rotation.identity)
    base_home: np.ndarray
    base_target: np.ndarray
    phase: str = "ready"
    held: str | None = None
    callback: Callable[..., None] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    steps: int = 0
    grip: float = 0.0
    hand_offset: np.ndarray | None = None
    rotation_offset: Any = None
    lost_contacts: int = 0
    posture: float = 0.0
    posture_goal: float = 0.0
    posture_weight: float = 0.015
    pitcher_approach_clearance: float = 0.25
    pitcher_open_grip: float = 0.0
    pitcher_hover_backoff: float = 0.0
    pitcher_lift_height: float = 0.28
    placement_base_side_step: float = 0.0
    placement_base_backoff: float = 0.0
    placement_clearance: float = 0.02
    pour_clearance: float = 0.45
    handle_roll_degrees: float = 90.0
    handle_contact_depth: float = 0.04
    handle_lean_degrees: float = 0.0
    handle_lean_axis: str = "y"
    ik_kinematics_only: bool = False

    @classmethod
    def create(cls, *, env: Any) -> "NativePouringController":
        import mujoco

        scene = env.unwrapped._object_centric_env
        model, data = scene._model_and_data()
        robot = scene.robot_name
        joints = [model.joint(f"{robot}_joint_{i}").id for i in range(1, 8)]
        arm_q = model.jnt_qposadr[joints]
        base_q = np.array([
            model.joint(f"{robot}_joint_{axis}").qposadr[0] for axis in ("x", "y", "th")
        ])
        return cls(
            env=env,
            scene=scene,
            model=model,
            data=data,
            scratch=mujoco.MjData(model),
            arm_q=arm_q,
            arm_v=model.jnt_dofadr[joints],
            base_q=base_q,
            last_q=data.qpos[arm_q].copy(),
            lower=np.array([
                model.jnt_range[j, 0] if model.jnt_limited[j] else -12 for j in joints
            ]),
            upper=np.array([model.jnt_range[j, 1] if model.jnt_limited[j] else 12 for j in joints]),
            site=model.site(f"{robot}_pinch_site").id,
            frequency=scene._robot_env.control_frequency,
            home=data.xpos[model.body("pitcher").id].copy(),
            tray_home=data.xpos[model.body("tray").id].copy(),
            tray_home_rotation=Rotation.from_quat(data.xquat[model.body("tray").id][[1, 2, 3, 0]]),
            base_home=data.qpos[base_q].copy(),
            base_target=data.qpos[base_q].copy(),
        )

    def body_pose(self, *, name: str) -> tuple[np.ndarray, Rotation]:
        body = self.model.body(name).id
        return (
            self.data.xpos[body].copy(),
            Rotation.from_quat(self.data.xquat[body][[1, 2, 3, 0]]),
        )

    def hand_pose(self) -> tuple[np.ndarray, Rotation]:
        return (
            self.data.site_xpos[self.site].copy(),
            Rotation.from_matrix(self.data.site_xmat[self.site].reshape(3, 3)),
        )

    def obstacle_residual(self) -> np.ndarray:
        """Optional collision-clearance cost, evaluated only in scratch state."""
        return np.zeros(0)

    def update_ik_scratch(self) -> None:
        """IK only needs world geometry/site transforms, not contact dynamics."""
        import mujoco

        if self.ik_kinematics_only:
            mujoco.mj_kinematics(self.model, self.scratch)
        else:
            mujoco.mj_forward(self.model, self.scratch)

    def inverse_kinematics(self, *, position: np.ndarray, rotation: Rotation) -> np.ndarray:

        self.posture += float(np.clip(self.posture_goal - self.posture, -0.02, 0.02))
        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qpos[self.base_q] = self.base_target

        def residual(q: np.ndarray) -> np.ndarray:  # noqa: PLR0917
            self.scratch.qpos[self.arm_q] = q
            self.update_ik_scratch()
            actual = Rotation.from_matrix(self.scratch.site_xmat[self.site].reshape(3, 3))
            return np.r_[
                self.scratch.site_xpos[self.site] - position,
                0.22 * (rotation * actual.inv()).as_rotvec(),
                0.002 * (q - self.last_q),
                self.posture_weight * (q[4] - self.posture),
                self.obstacle_residual(),
            ]

        fit = least_squares(
            residual,
            np.clip(self.last_q, self.lower + 1e-6, self.upper - 1e-6),
            bounds=(self.lower, self.upper),
            max_nfev=100,
        )
        continuous = self.lower == -12
        delta = fit.x[continuous] - self.last_q[continuous]
        fit.x[continuous] = self.last_q[continuous] + np.arctan2(np.sin(delta), np.cos(delta))
        error = float(np.linalg.norm(residual(fit.x)[:6]))
        if error > 0.008 or np.max(np.abs(fit.x - self.last_q)) > 0.5:
            # Search redundant configurations in scratch state, never physical outcomes.
            candidates = []
            for joint in (0, 2, 4, 6):
                for delta in (-0.25, 0.25):
                    seed = self.last_q.copy()
                    seed[joint] += delta
                    candidate = least_squares(
                        residual,
                        np.clip(seed, self.lower + 1e-6, self.upper - 1e-6),
                        bounds=(self.lower, self.upper),
                        max_nfev=120,
                    )
                    delta = candidate.x[continuous] - self.last_q[continuous]
                    candidate.x[continuous] = self.last_q[continuous] + np.arctan2(
                        np.sin(delta), np.cos(delta)
                    )
                    candidate_error = float(np.linalg.norm(residual(candidate.x)[:6]))
                    if candidate_error < 0.008 and np.max(np.abs(candidate.x - self.last_q)) <= 0.5:
                        candidates.append((np.linalg.norm(candidate.x - self.last_q), candidate))
            if candidates:
                fit = min(candidates, key=lambda entry: entry[0])[1]
                error = float(np.linalg.norm(residual(fit.x)[:6]))
        if error > 0.008:
            raise ValueError(
                f"IK infeasible in {self.phase}: residual {error:.4f}; "
                f"joint degrees {np.round(np.rad2deg(fit.x), 1).tolist()}"
            )
        if np.max(np.abs(fit.x - self.last_q)) > 0.5:
            raise ValueError(f"Discontinuous IK branch in {self.phase}")
        self.last_q = fit.x.copy()
        return fit.x

    def payload_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Estimated load at the hand; controllers may supply a fixed load model."""
        force = np.zeros(3)
        moment = np.zeros(3)
        if self.held is not None:
            names = [self.held]
            if self.held == "pitcher":
                names += [
                    name
                    for name in self.scene.cube_names
                    if self.scene.cube_in_vessel(name, "pitcher")
                ]
            for name in names:
                body = self.model.body(name).id
                weight = -self.model.body_subtreemass[body] * self.model.opt.gravity
                force += weight
                moment += np.cross(
                    self.data.subtree_com[body] - self.data.site_xpos[self.site], weight
                )
        return force, moment

    def motor_step(self, *, target_q: np.ndarray, grip: float) -> None:
        import mujoco

        motor_target = target_q.copy()
        if self.held is not None:
            force, moment = self.payload_wrench()
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
            torque = jacp[:, self.arm_v].T @ force + jacr[:, self.arm_v].T @ moment
            motor_target = np.clip(
                motor_target + torque / self.scene._robot_env.arm_kp, self.lower, self.upper
            )
        action = np.r_[
            np.clip(self.base_target - self.data.qpos[self.base_q], -0.08, 0.08),
            np.clip(motor_target - self.data.qpos[self.arm_q], -0.08, 0.08),
            grip,
        ]
        if self.env.action_space.shape == (18,):
            action = np.r_[action, np.zeros(7)]
        self.env.step(action)
        self.steps += 1
        if self.callback is not None:
            self.callback(controller=self)
        if self.held == "pitcher":
            self.lost_contacts = (
                self.lost_contacts + 1
                if len(self.finger_contacts(name="pitcher", handle_only=True)) < 2
                else 0
            )
            position, _ = self.body_pose(name="pitcher")
            hand, _ = self.hand_pose()
            if self.lost_contacts > 3 or np.linalg.norm(position - hand) > 0.35:
                self.held = None
                raise ValueError(f"Physical pitcher grasp lost during {self.phase}")

    def move_base(self, *, target: np.ndarray, phase: str, seconds: float) -> None:
        self.phase = phase
        start = self.data.qpos[self.base_q].copy()
        count = max(1, round(seconds * self.frequency))
        for index in range(1, count + 1):
            u = index / count
            self.base_target = start + u * u * (3 - 2 * u) * (target - start)
            self.motor_step(target_q=self.last_q, grip=self.grip)
        for _ in range(10):
            self.motor_step(target_q=self.last_q, grip=self.grip)
        if np.linalg.norm(self.data.qpos[self.base_q] - target) > 0.02:
            raise ValueError(f"Base tracking failed in {phase}")

    def move(
        self,
        *,
        phase: str,
        position: np.ndarray,
        rotation: Rotation,
        seconds: float,
        grip: float,
    ) -> None:
        self.phase = phase
        start_pos, start_rot = self.hand_pose()
        relative = (rotation * start_rot.inv()).as_rotvec()
        count = max(1, round(seconds * self.frequency))
        for index in range(1, count + 1):
            u = index / count
            smooth = u * u * (3 - 2 * u)
            desired = start_pos + smooth * (position - start_pos)
            orient = Rotation.from_rotvec(smooth * relative) * start_rot
            q = self.inverse_kinematics(position=desired, rotation=orient)
            self.motor_step(target_q=q, grip=grip)
        for _ in range(round(15 * self.frequency)):
            self.motor_step(target_q=self.last_q, grip=grip)
            actual_pos, actual_rot = self.hand_pose()
            if (
                np.linalg.norm(actual_pos - position) < 0.015
                and (rotation * actual_rot.inv()).magnitude() < 0.08
            ):
                self.grip = grip
                return
        raise ValueError(
            f"Motor tracking failed in {phase}: {np.linalg.norm(actual_pos - position):.3f} m"
        )

    def pause(self, *, phase: str, seconds: float, grip: float | None = None) -> None:
        self.phase = phase
        target = self.last_q.copy()
        for _ in range(round(seconds * self.frequency)):
            self.motor_step(target_q=target, grip=self.grip if grip is None else grip)
        if grip is not None:
            self.grip = grip

    def finger_contacts(self, *, name: str, handle_only: bool = False) -> set[str]:
        contacts = self.scene.get_gripper_object_contacts(name)
        if handle_only:
            contacts = [
                contact
                for contact in contacts
                if any(
                    str(contact.get(key, "")).startswith(f"{name}_handle_")
                    for key in ("geom1", "geom2")
                )
            ]
        return {
            side
            for side in ("left", "right")
            if any(
                f"_{side}_pad" in str(c.get("geom1")) or f"_{side}_pad" in str(c.get("geom2"))
                for c in contacts
            )
        }

    def pitcher_grasp_pose(self) -> tuple[np.ndarray, Rotation, Rotation]:
        origin, rotation = self.body_pose(name="pitcher")
        upright = Rotation.from_euler(
            self.handle_lean_axis, -self.handle_lean_degrees, degrees=True
        ) * Rotation.from_euler("y", np.pi)
        down = (
            upright * Rotation.from_euler("z", self.handle_roll_degrees, degrees=True)
            if self.handle_lean_degrees
            else Rotation.from_euler("z", -np.pi / 2) * upright
        )
        contact = origin + rotation.apply(self.scene.get_object("pitcher").handle_grasp_position)
        target = contact + (
            -down.apply([0, 0, self.handle_contact_depth])
            if self.handle_lean_degrees
            else np.array([0, 0, 0.04])
        )
        return target, upright, down

    def pick_pitcher(self) -> None:
        if not self.ready():
            raise ValueError("Pitcher pick requires a ready five-cube scene")
        self.move_base(target=self.base_home, phase="approach pitcher base", seconds=4)
        origin, _ = self.body_pose(name="pitcher")
        handle, upright, down = self.pitcher_grasp_pose()
        self.move(
            phase="approach pitcher",
            position=handle + [-self.pitcher_hover_backoff, 0, self.pitcher_approach_clearance],
            rotation=upright,
            seconds=8,
            grip=self.pitcher_open_grip,
        )
        self.move(
            phase="align fingers across handle",
            position=handle + [-self.pitcher_hover_backoff, 0, self.pitcher_approach_clearance],
            rotation=down,
            seconds=8,
            grip=self.pitcher_open_grip,
        )
        self.move(
            phase="descend to pitcher handle",
            position=handle,
            rotation=down,
            seconds=4,
            grip=self.pitcher_open_grip,
        )
        self.pause(phase="close fingers on pitcher handle", seconds=3, grip=1)
        if len(self.finger_contacts(name="pitcher", handle_only=True)) != 2:
            raise ValueError("Pitcher handle grasp lacks handle contact on both fingers")
        self.move(
            phase="lift pitcher by handle",
            position=handle + [0, 0, self.pitcher_lift_height],
            rotation=down,
            seconds=6,
            grip=1,
        )
        position, actual_rotation = self.body_pose(name="pitcher")
        if position[2] < origin[2] + self.pitcher_lift_height - 0.08:
            raise ValueError("Pitcher grasp did not lift the physical vessel")
        if len(self.finger_contacts(name="pitcher", handle_only=True)) != 2:
            raise ValueError("Handle grasp was not retained through the lift")
        self.held = "pitcher"
        hand, orientation = self.hand_pose()
        self.hand_offset = actual_rotation.inv().apply(hand - position)
        self.rotation_offset = actual_rotation.inv() * orientation

    def move_pitcher(
        self,
        *,
        phase: str,
        position: np.ndarray,
        rotation: Rotation,
        seconds: float,
    ) -> None:
        if self.held != "pitcher" or self.hand_offset is None:
            raise ValueError("No retained pitcher grasp")
        self.move(
            phase=phase,
            position=position + rotation.apply(self.hand_offset),
            rotation=rotation * self.rotation_offset,
            seconds=seconds,
            grip=1,
        )

    def pour(self, *, parameters: PourParameters) -> None:
        tray, _ = self.body_pose(name="tray")
        target = tray + [parameters.offset_x, parameters.offset_y, 0.45]
        self.move_base(
            target=np.array([target[0] - 0.55, target[1], 0.0]),
            phase="position base for pour",
            seconds=5,
        )
        self.posture_goal = 1.0
        # Face the physical +x spout along world +y while upright. This keeps
        # the full handle-held tilt inside the arm's reachable orientation range.
        upright = Rotation.from_euler("z", np.pi / 2)
        self.move(
            phase="transport pour pivot above tray",
            position=target,
            rotation=upright * self.rotation_offset,
            seconds=5,
            grip=1,
        )
        # Rotate around the physical grasp, as a wrist pour does. The freely
        # moving vessel follows contact forces; its bottom is not held at a pivot.
        for angle in np.linspace(
            0, parameters.angle, round(parameters.tilt_seconds * self.frequency) + 1
        )[1:]:
            orientation = (
                Rotation.from_rotvec(np.array([-1.0, 0.0, 0.0]) * np.deg2rad(angle)) * upright
            )
            self.phase = "tilt pitcher"
            q = self.inverse_kinematics(
                position=target, rotation=orientation * self.rotation_offset
            )
            self.motor_step(target_q=q, grip=1)
        self.pause(phase="hold pour", seconds=parameters.hold_seconds)
        for angle in np.linspace(
            parameters.angle, 0, round(parameters.tilt_seconds * self.frequency) + 1
        )[1:]:
            orientation = (
                Rotation.from_rotvec(np.array([-1.0, 0.0, 0.0]) * np.deg2rad(angle)) * upright
            )
            self.phase = "return upright"
            q = self.inverse_kinematics(
                position=target,
                rotation=orientation * self.rotation_offset,
            )
            self.motor_step(target_q=q, grip=1)
        self.pause(phase="settle cubes in tray", seconds=2)

    def place_pitcher(self) -> None:
        # Measure the current compliant contact grasp before positioning the vessel.
        position, rotation = self.body_pose(name="pitcher")
        hand, orientation = self.hand_pose()
        self.hand_offset = rotation.inv().apply(hand - position)
        self.rotation_offset = rotation.inv() * orientation
        self.move_base(
            target=self.base_home
            + [-self.placement_base_backoff, self.placement_base_side_step, 0],
            phase="return base for placement",
            seconds=5,
        )
        self.move_pitcher(
            phase="return pitcher above home",
            position=self.home + [0, 0, self.pitcher_lift_height],
            rotation=Rotation.identity(),
            seconds=4,
        )
        self.move_pitcher(
            phase="lower pitcher to floor",
            position=self.home + [0, 0, self.placement_clearance],
            rotation=Rotation.identity(),
            seconds=4,
        )
        self.held = None
        self.pause(phase="release pitcher", seconds=2, grip=0)
        hand, orient = self.hand_pose()
        self.move(
            phase="withdraw gripper",
            position=hand + [0, 0, 0.25],
            rotation=orient,
            seconds=3,
            grip=0,
        )
        self.move_base(
            target=self.base_home + [-0.35, 0, 0], phase="clear workspace for reset", seconds=4
        )

    def ready(self) -> bool:
        return (
            self.held is None
            and self.scene.cube_locations() == ["pitcher"] * 5
            and self.scene.cubes_settled()
            and self.vessel_home(name="pitcher")
            and self.vessel_home(name="tray")
        )

    def vessel_home(self, *, name: str) -> bool:
        position, rotation = self.body_pose(name=name)
        target = self.home if name == "pitcher" else self.tray_home
        reference = Rotation.identity() if name == "pitcher" else self.tray_home_rotation
        return bool(
            np.linalg.norm(position - target) < 0.025
            and (reference.inv() * rotation).magnitude() < np.deg2rad(5)
        )

    def select_reset_cube(self) -> int:
        candidates = [
            i
            for i, location in enumerate(self.scene.cube_locations())
            if location in {"tray", "floor"}
        ]
        if not candidates:
            raise ValueError("No cubes outside pitcher to reset")
        points = {i: self.body_pose(name=f"cube_{i}")[0][:2] for i in candidates}
        centre = np.mean(list(points.values()), axis=0)
        # Clear the outside of a cluster first so fingers can reach interior cubes.
        return max(
            candidates,
            key=lambda i: (round(float(np.linalg.norm(points[i] - centre)), 5), points[i][0]),
        )

    def pick_cube(self, *, cube: int, location: str) -> None:
        name = f"cube_{cube}"
        if self.held is not None or self.scene.cube_locations()[cube] != location:
            raise ValueError("Cube pick requires an empty hand and cube in the named location")
        position, cube_rotation = self.body_pose(name=name)
        horizontal = min(cube_rotation.as_matrix().T, key=lambda axis: abs(axis[2]))
        yaw = (np.arctan2(horizontal[1], horizontal[0]) + np.pi / 4) % (np.pi / 2) - np.pi / 4
        if location == "tray":
            tray, tray_rotation = self.body_pose(name="tray")
            geometry = self.scene.get_object("tray")
            clearance = (
                np.array([geometry.length, geometry.width]) / 2
                - geometry.wall_thickness
                - np.abs(tray_rotation.inv().apply(position - tray)[:2])
            )
            candidates = [yaw + k * np.pi / 2 for k in range(-2, 3)]
            _, current_rotation = self.hand_pose()

            def grasp_score(angle: float) -> tuple[float, float]:  # noqa: PLR0917
                half = np.array([
                    0.06 * abs(np.cos(angle)) + 0.02 * abs(np.sin(angle)),
                    0.06 * abs(np.sin(angle)) + 0.02 * abs(np.cos(angle)),
                ])
                target = Rotation.from_euler("z", angle) * Rotation.from_euler("y", np.pi)
                return (
                    round(float(np.min(clearance - half)), 5),
                    -(target * current_rotation.inv()).magnitude(),
                )

            yaw = max(candidates, key=grasp_score)
        down = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi)
        # A fully open 85 mm gripper sweeps neighbouring 20 mm cubes aside.
        # Preclose above the pile, leaving clearance around the selected cube.
        pregrip = 0.7
        self.move_base(
            target=np.array([position[0] - 0.65, position[1], 0]),
            phase="position base for cube pick",
            seconds=5,
        )
        self.move(
            phase="approach cube",
            position=position + [0, 0, 0.25],
            rotation=down,
            seconds=5,
            grip=pregrip,
        )
        self.move(
            phase="descend to cube",
            position=position + [0, 0, 0.04],
            rotation=down,
            seconds=4,
            grip=pregrip,
        )
        self.pause(phase="settle cube grasp approach", seconds=1, grip=pregrip)
        self.phase = "grasp cube"
        for command in np.linspace(pregrip, 1, round(3 * self.frequency) + 1)[1:]:
            self.motor_step(target_q=self.last_q, grip=float(command))
        self.pause(phase="retain cube grasp", seconds=1, grip=1)
        if len(self.finger_contacts(name=name)) != 2:
            raise ValueError("Cube grasp lacks contact on both fingers")
        self.held = name
        self.move(
            phase="lift cube", position=position + [0, 0, 0.28], rotation=down, seconds=4, grip=1
        )
        actual, _ = self.body_pose(name=name)
        if actual[2] < position[2] + 0.20 or len(self.finger_contacts(name=name)) != 2:
            self.held = None
            raise ValueError("Physical cube grasp was not retained during lift")

    def drop_cube(self, *, cube: int) -> None:
        name = f"cube_{cube}"
        if self.held != name or not self.vessel_home(name="pitcher"):
            raise ValueError("Drop requires the named held cube and pitcher upright at home")
        pitcher, _ = self.body_pose(name="pitcher")
        _, down = self.hand_pose()
        self.move_base(
            target=np.array([pitcher[0] - 0.65, pitcher[1], 0]),
            phase="position base for cube drop",
            seconds=5,
        )
        self.move(
            phase="transport cube above pitcher",
            position=pitcher + [0, 0, 0.35],
            rotation=down,
            seconds=5,
            grip=1,
        )
        if len(self.finger_contacts(name=name)) != 2:
            self.held = None
            raise ValueError("Cube lost before release into pitcher")
        self.held = None
        self.pause(phase="release cube into pitcher", seconds=3, grip=0)
        if not self.scene.cube_in_vessel(name, "pitcher"):
            raise ValueError("Released cube did not land in pitcher")

    def run_skill(
        self, *, skill: str, parameters: PourParameters | None = None, cube: int = 0
    ) -> dict[str, Any]:
        before = self.scene.physical_snapshot()
        error = None
        try:
            if skill == "PickPitcher":
                self.pick_pitcher()
                success = True
            elif skill == "Pour":
                self.pour(parameters=parameters or PourParameters())
                success = self.scene._check_goals()
            elif skill == "PlacePitcherAtHome":
                self.place_pitcher()
                position, rotation = self.body_pose(name="pitcher")
                success = bool(
                    np.linalg.norm(position - self.home) < 0.025
                    and rotation.magnitude() < np.deg2rad(5)
                )
            elif skill in {"PickCubeFromTray", "PickCubeFromFloor"}:
                self.pick_cube(
                    cube=cube, location="tray" if skill == "PickCubeFromTray" else "floor"
                )
                success = True
            elif skill == "DropCubeIntoPitcher":
                self.drop_cube(cube=cube)
                success = self.scene.cube_in_vessel(f"cube_{cube}", "pitcher")
            elif skill == "Wait":
                self.pause(phase="wait", seconds=1)
                success = True
            else:
                raise ValueError(f"Unknown native skill {skill}")
        except ValueError as exc:
            success = False
            error = str(exc)
        after = self.scene.physical_snapshot()
        event = {
            "skill": skill,
            "cube": cube,
            "success": success,
            "error": error,
            "parameters": parameters.model_dump() if parameters else None,
            "seconds": after["time"] - before["time"],
            "before": before,
            "after": after,
        }
        self.events.append(event)
        print(
            f"{skill}: success={success}; phase={self.phase}; "
            f"locations={after['locations']}; error={error}",
            flush=True,
        )
        return event
