"""Force-assisted table strokes with fast feedback and guarded forward progress."""

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .controllers import TableSweep
from .servo import SweepServo
from .wiper import TableWiper


class ContactTableSweep(TableSweep):
    MAX_CLEARANCE = 0.005
    MAX_ORIENTATION_ERROR_DEG = 8.0

    def __init__(self, *, wiper: TableWiper, center_y: float) -> None:
        super().__init__(wiper=wiper, center_y=center_y)
        self.metrics: dict[str, float | int] = {}

    def pull(self, *, position: np.ndarray, rotation: np.ndarray, yaw: float) -> None:
        shelf, wiper = self.shelf, self.wiper
        direction = self.direction(yaw=yaw)
        length = self.stroke_length(yaw=yaw)
        base = shelf.data.qpos[shelf.ba].copy()
        arm = shelf.data.qpos[shelf.qa].copy()
        distance = 0.0
        advance = self.SPEED * 0.1
        paused_steps = 0
        consecutive_pauses = 0
        completed = False
        assert shelf.in_hand is not None
        desired_body_rotation = rotation @ shelf.in_hand[:3, :3]
        contact = TableContactLoad(wiper=wiper)
        servo = SweepServo(
            shelf=shelf,
            wiper=wiper,
            max_normal_force_n=contact.MAX_FORCE_N,
            control_update=contact.update,
        )
        contact.servo = servo
        try:
            with servo:
                for _ in range(20):
                    shelf.step(arm=arm, base=base, label="sweep_contact_seat")
                    wiper.require_held(label="sweep_contact_seat")
                    if contact.support_force_n >= 0.3 and contact.maximum_clearance() <= 0.0015:
                        break
                else:
                    raise SkillFailed("SweepTable could not establish supported blade contact")
                for _ in range(int(3 * length / advance) + 100):
                    body_rotation = wiper.wiper_pose()[:3, :3]
                    orientation_error = float(
                        Rotation.from_matrix(body_rotation @ desired_body_rotation.T).magnitude()
                        * 180
                        / np.pi
                    )
                    safe = (
                        contact.maximum_clearance() <= self.MAX_CLEARANCE
                        and orientation_error <= self.MAX_ORIENTATION_ERROR_DEG
                    )
                    waypoint = position + distance * direction
                    lag = float((waypoint - shelf.data.site_xpos[shelf.site]) @ direction)
                    if safe and lag < self.TRACK_TOLERANCE:
                        if distance >= length:
                            completed = True
                            break
                        distance = min(length, distance + advance)
                    if safe:
                        consecutive_pauses = 0
                        base_target = base + distance * direction
                    else:
                        paused_steps += 1
                        consecutive_pauses += 1
                        actual_progress = float(
                            (shelf.data.site_xpos[shelf.site] - position) @ direction
                        )
                        distance = max(0.0, min(distance, actual_progress))
                        base_target = shelf.data.qpos[shelf.ba].copy()
                        if consecutive_pauses >= 50:
                            raise SkillFailed(
                                "SweepTable contact guard could not restore blade pose"
                            )
                    waypoint = position + distance * direction
                    shelf.base_target = base_target.copy()
                    previous = arm
                    arm, _ = shelf.ik(
                        position=waypoint,
                        rotation=rotation,
                        base=base_target,
                        holding=True,
                        attempts=4,
                        seed=previous,
                    )
                    jump = np.max(np.abs((arm - previous + np.pi) % (2 * np.pi) - np.pi))
                    if not np.isfinite(arm).all() or jump > 0.6:
                        raise SkillFailed("SweepTable IK lost its continuous branch")
                    shelf.step(arm=arm, base=base_target, label="sweep_pull")
                    wiper.require_held(label="sweep_pull")
                if not completed:
                    raise SkillFailed(f"SweepTable stalled at {distance:.3f}/{length:.3f} m")
                contact.enabled = False
                servo.normal_force_n = 0
                shelf.hold(count=10, label="sweep_settle")
                wiper.line(
                    position=shelf.data.site_xpos[shelf.site] + [0, 0, 0.10],
                    rotation=rotation,
                    label="sweep_lift",
                    holding=True,
                )
                shelf.hold(count=50, label="sweep_settle")
        finally:
            self.metrics = {
                "controller_revision": 3,
                "servo_updates": servo.servo_updates,
                "clipped_joint_torque_commands": servo.clipped_torque_count,
                "guard_paused_steps": paused_steps,
                "max_blade_clearance_over_table_m": contact.max_clearance_m,
                "max_support_force_n": contact.max_support_force_n,
                "max_commanded_downforce_n": contact.max_commanded_force_n,
                "clearance_limit_m": self.MAX_CLEARANCE,
                "target_support_force_n": contact.TARGET_FORCE_N,
                "downforce_limit_n": contact.MAX_FORCE_N,
            }


class TableContactLoad:
    TARGET_FORCE_N = 2.0
    MAX_FORCE_N = 12.0

    def __init__(self, *, wiper: TableWiper) -> None:
        self.wiper = wiper
        self.shelf = wiper.shelf
        self.servo: SweepServo | None = None
        self.enabled = True
        self.support_force_n = 0.0
        self.integral_n = 0.0
        self.max_clearance_m = 0.0
        self.max_support_force_n = 0.0
        self.max_commanded_force_n = 0.0
        self._force = np.zeros(6)

    def maximum_clearance(self) -> float:
        wiper = self.wiper
        axis = wiper.data.geom_xmat[wiper.head_geom].reshape(3, 3)[:, 2]
        return float(
            wiper.data.geom_xpos[wiper.head_geom, 2]
            + wiper.model.geom_size[wiper.head_geom, 1] * abs(axis[2])
            - wiper.radius
            - wiper.top
        )

    def update(self) -> None:
        assert self.servo is not None
        if not self.enabled:
            self.servo.normal_force_n = 0
            return
        shelf, wiper = self.shelf, self.wiper
        head = shelf.data.geom_xpos[wiper.head_geom]
        axis = shelf.data.geom_xmat[wiper.head_geom].reshape(3, 3)[:, 2]
        leading_x = (
            head[0] - shelf.model.geom_size[wiper.head_geom, 1] * abs(axis[0]) - wiper.radius
        )
        scale = float(np.clip((leading_x - shelf.front_x) / 0.02, 0, 1))
        if scale <= 0:
            self.integral_n = 0
            self.servo.normal_force_n = 0
            return
        self.max_clearance_m = max(self.max_clearance_m, self.maximum_clearance())
        support = 0.0
        for index, contact in enumerate(shelf.data.contact[: shelf.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            if wiper.head_geom in pair and pair & shelf.cupboard_geoms:
                shelf.mujoco.mj_contactForce(shelf.model, shelf.data, index, self._force)
                world = np.asarray(contact.frame).reshape(3, 3).T @ self._force[:3]
                support += float(world[2] if contact.geom2 == wiper.head_geom else -world[2])
        self.max_support_force_n = max(self.max_support_force_n, support)
        timestep = float(shelf.model.opt.timestep)
        self.support_force_n += (1 - np.exp(-timestep / 0.02)) * (support - self.support_force_n)
        target = self.TARGET_FORCE_N * scale
        error = target - self.support_force_n
        command = target + 0.4 * error + self.integral_n
        if (
            0 < command < self.MAX_FORCE_N
            or command <= 0 < error
            or (command >= self.MAX_FORCE_N and error < 0)
        ):
            self.integral_n = float(
                np.clip(
                    self.integral_n + 2.5 * error * timestep, -self.MAX_FORCE_N, self.MAX_FORCE_N
                )
            )
        self.servo.normal_force_n = scale * float(np.clip(command, 0, self.MAX_FORCE_N))
        self.max_commanded_force_n = max(self.max_commanded_force_n, self.servo.normal_force_n)
