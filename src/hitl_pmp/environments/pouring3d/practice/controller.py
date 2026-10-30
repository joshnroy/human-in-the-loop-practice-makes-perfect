"""Physical recovery, collision-aware station motions, and observable motor ticks."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.feedback_controller import (
    FeedbackPourController,
    FeedbackPourParameters,
)
from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp

from .skillet import CastIronSkillet


class PracticePourController(FeedbackPourController):
    active_pour_count: int = 0
    obstacle_clearance: float = 0.015
    pour_height_drop: float = 0.08
    obstacle_pairs: list[tuple[int, int]] = Field(default_factory=list)
    pan_pairs: list[tuple[int, int]] = Field(default_factory=list)
    shrimp_recovery_active: bool = False
    recovery_approach_clearance: float = 0.10
    recovery_finger_offset: float = 0.039
    recovery_open_grip: float = 0.4
    recovery_pitch_degrees: float = 30.0
    recovery_base_x: float = 0.20
    recovery_lift_from_actual: bool = False
    recovery_floor_carry_angle: float = 30.0
    recovery_contact_close: bool = False
    payload_hold_grip: float = 1.0
    shrimp_approach_path: list[np.ndarray] = Field(default_factory=list)

    before_step: Callable[[], None] | None = None
    after_step: Callable[[], None] | None = None

    @classmethod
    def create(cls, *, env: Any) -> "PracticePourController":
        controller = cast(PracticePourController, super().create(env=env))
        if "prep_table" in controller.scene._fixtures_dict:
            robot_geoms = [
                i
                for i in range(controller.model.ngeom)
                if (
                    controller.model.body(int(controller.model.geom_bodyid[i])).name or ""
                ).startswith("robot")
                and controller.model.geom_contype[i]
            ]
            support_geoms = [
                i
                for i in range(controller.model.ngeom)
                if controller.model.body(int(controller.model.geom_bodyid[i])).name
                in {"prep_table", "stovetop"}
                and controller.model.geom_contype[i]
            ]
            controller.obstacle_pairs = [(a, b) for a in robot_geoms for b in support_geoms]
            pan_geoms = [
                i
                for i in range(controller.model.ngeom)
                if controller.model.body(int(controller.model.geom_bodyid[i])).name == "tray"
            ]
            controller.pan_pairs = [(a, b) for a in robot_geoms for b in pan_geoms]
            controller.base_home[0] = 0.0
            controller.handle_lean_degrees = 120.0
            controller.handle_contact_depth = 0.025
            controller.return_speed_deg_s = 10.0
            controller.pour_height_drop = 0.16
            controller.handle_lean_axis = "y"
            controller.pitcher_approach_clearance = 0.10
            controller.pitcher_lift_height = 0.26
            controller.pour_clearance = 0.36
            controller.pour_base_offset = 0.70
            controller.pour_posture = 1.0
            controller.posture_weight = 0.001
            controller.placement_base_backoff = 0.05
            controller.placement_clearance = 0.035
            controller.pour_yaw_degrees = 0.0
        return controller

    def active_obstacle_pairs(self) -> list[tuple[int, int]]:
        return self.obstacle_pairs + (
            self.pan_pairs if self.held == "pitcher" or self.shrimp_recovery_active else []
        )

    def obstacle_residual(self) -> np.ndarray:
        import mujoco

        pairs = self.active_obstacle_pairs()
        if not pairs:
            return np.zeros(0)
        ids = np.asarray(pairs)
        a, b = ids[:, 0], ids[:, 1]
        bound = np.linalg.norm(self.scratch.geom_xpos[a] - self.scratch.geom_xpos[b], axis=1)
        bound -= self.model.geom_rbound[a] + self.model.geom_rbound[b]
        result = np.zeros(len(pairs))
        for i in np.flatnonzero(bound < self.obstacle_clearance):
            body = self.model.body(int(self.model.geom_bodyid[a[i]])).name or ""
            clearance = self.obstacle_clearance
            if self.phase in {"seat pitcher on table", "withdraw gripper"} and body == "robot_base":
                # Robotiq's palm is named robot_base. Seating the low handle
                # requires a closer palm approach, while retaining a 5 mm gap.
                clearance = 0.005
            if "shrimp" in self.phase and body.startswith(("robot_left_", "robot_right_")):
                # Fingers must get near the supporting surface to pinch a small
                # object; the arm and wrist retain their normal clearance.
                clearance = 0.0 if body.endswith("_pad") else 0.002
            result[i] = 5.0 * max(
                0.0,
                clearance
                - mujoco.mj_geomDistance(
                    self.model, self.scratch, int(a[i]), int(b[i]), self.obstacle_clearance, None
                ),
            )
        return result

    def feedback_pour(
        self, *, parameters: FeedbackPourParameters, requested_count: int | None = None
    ) -> None:
        self.active_pour_count = (
            self.scene.requested_count if requested_count is None else requested_count
        )
        super().feedback_pour(parameters=parameters, requested_count=requested_count)

    def inverse_kinematics(self, *, position: np.ndarray, rotation: Rotation) -> np.ndarray:
        previous = self.last_q.copy()
        target = super().inverse_kinematics(position=position, rotation=rotation)
        # Pose accuracy alone cannot certify the soft collision penalty. Reject a
        # blocked target before advancing the controller's commanded branch.
        clearance_cost = self.obstacle_residual()
        if self.phase in {"descend to shrimp body", "lift shrimp"}:
            # The pinch is a contact motion: retain the finger clearance cost in
            # IK, but do not require its free-space margin at every waypoint.
            # The wrist and arm must still satisfy the hard clearance check.
            for i, (a, _) in enumerate(self.active_obstacle_pairs()):
                body = self.model.body(int(self.model.geom_bodyid[a])).name or ""
                if body.startswith(("robot_left_", "robot_right_")):
                    clearance_cost[i] = 0.0
        if np.max(clearance_cost, initial=0.0) > 0.005:
            self.last_q = previous
            a, b = self.active_obstacle_pairs()[int(np.argmax(clearance_cost))]
            body = self.model.body(int(self.model.geom_bodyid[a])).name
            obstacle = self.model.geom(b).name
            raise ValueError(f"Insufficient obstacle clearance in {self.phase}: {body}/{obstacle}")
        return target

    def tracked_step(self, *, center: np.ndarray, angle: float, offset: float = 0) -> None:
        stations = "prep_table" in self.scene._fixtures_dict
        if stations:
            center = center.copy()
            fraction = min(1.0, angle / 30.0)
            center[2] -= self.pour_height_drop * fraction * fraction * (3 - 2 * fraction)
        super().tracked_step(center=center, angle=angle, offset=offset)
        if stations and self.phase in {"feedback tilt pitcher", "feedback return upright"}:
            for _ in range(30):
                error = np.max(np.abs(self.last_q - self.data.qpos[self.arm_q]))
                if error < 0.025:
                    return
                if (
                    self.phase == "feedback tilt pitcher"
                    and self.observe_mouth() >= self.active_pour_count
                ):
                    return
                self.motor_step(target_q=self.last_q, grip=1)
                self.observe_mouth()
            raise ValueError(f"Arm did not track pour waypoint in {self.phase}: {error:.3f} rad")

    def pitcher_grasp_pose(self) -> tuple[np.ndarray, Rotation, Rotation]:
        position, upright, grasp = super().pitcher_grasp_pose()
        if "prep_table" in self.scene._fixtures_dict and self.handle_lean_degrees >= 90:
            return position, grasp, grasp
        return position, upright, grasp

    def prepare_empty_arm(self, *, position: np.ndarray, rotation: Rotation) -> None:
        """Find a reachable empty-hand posture, then execute a checked joint path."""

        if self.held is not None:
            raise ValueError("Arm preparation requires an empty hand")
        start = self.data.qpos[self.arm_q].copy()
        self.scratch.qpos[:] = self.data.qpos

        def residual(q: np.ndarray) -> np.ndarray:  # noqa: PLR0917
            self.scratch.qpos[self.arm_q] = q
            self.update_ik_scratch()
            actual = Rotation.from_matrix(self.scratch.site_xmat[self.site].reshape(3, 3))
            return np.r_[
                self.scratch.site_xpos[self.site] - position,
                0.22 * (rotation * actual.inv()).as_rotvec(),
                self.obstacle_residual(),
            ]

        rng = np.random.default_rng(0)
        candidates = []
        best_error = float("inf")
        reachable = 0
        for index in range(16):
            seed = (
                start
                if index == 0
                else rng.uniform(np.maximum(self.lower, -np.pi), np.minimum(self.upper, np.pi))
            )
            fit = least_squares(residual, seed, bounds=(self.lower, self.upper), max_nfev=200)
            fit_error = float(np.linalg.norm(residual(fit.x)))
            best_error = min(best_error, fit_error)
            if fit_error > 0.008:
                continue
            reachable += 1
            goal = fit.x.copy()
            continuous = self.lower == -12
            delta = goal[continuous] - start[continuous]
            goal[continuous] = start[continuous] + np.arctan2(np.sin(delta), np.cos(delta))
            clear = True
            for u in np.linspace(0, 1, 101):
                residual(start + u * (goal - start))
                if np.max(self.obstacle_residual(), initial=0) > 0.005:
                    clear = False
                    break
                # The empty hand may move through an arc, but not through the floor.
                if self.scratch.site_xpos[self.site, 2] < 0.25:
                    clear = False
                    break
            if clear:
                candidates.append(goal)
        if not candidates:
            raise ValueError(
                f"No collision-free empty-arm preparation path: residual {best_error:.4f}, "
                f"reachable poses {reachable}"
            )
        goal = min(candidates, key=lambda q: np.linalg.norm(q - start))
        self.phase = "prepare arm in clear aisle"
        count = round(max(8.0, float(np.max(np.abs(goal - start))) / 0.20) * self.frequency)
        for index in range(1, count + 1):
            u = index / count
            self.last_q = start + u * u * (3 - 2 * u) * (goal - start)
            self.motor_step(target_q=self.last_q, grip=0)
        self.move(position=position, rotation=rotation, seconds=2, grip=0, phase=self.phase)

    def pick_pitcher(self) -> None:
        self.shrimp_recovery_active = False
        if "prep_table" in self.scene._fixtures_dict:
            if not self.ready():
                raise ValueError("Pitcher pick requires a ready scene")
            self.move_base(
                target=np.array([0.0, self.home[1], 0]),
                phase="clear table before raising arm",
                seconds=3,
            )
            hover, upright, _ = self.pitcher_grasp_pose()
            hover += [0, 0, self.pitcher_approach_clearance]
            hover[0] -= self.base_home[0]
            if self.handle_lean_degrees >= 90:
                self.prepare_empty_arm(position=hover, rotation=upright)
            else:
                self.move(
                    position=hover,
                    rotation=upright,
                    seconds=8,
                    grip=0,
                    phase="raise arm in aisle before pitcher approach",
                )
        super().pick_pitcher()

    def move_base(self, *, target: np.ndarray, phase: str, seconds: float) -> None:
        elevated = "prep_table" in self.scene._fixtures_dict
        if elevated and self.recovery_lift_from_actual and phase == "position for shrimp pick":
            # Holding a fixed arm configuration sweeps the forearm through the
            # cabinet as the base approaches. Retract the arm during that travel.
            self.phase = phase
            start = self.data.qpos[self.base_q].copy()
            hand, rotation = self.hand_pose()
            count = round(seconds * self.frequency)
            for index in range(1, count + 11):
                u = min(1.0, index / count)
                self.base_target = start + u * u * (3 - 2 * u) * (target - start)
                shift = self.base_target - start
                position = hand + np.array([0.4 * shift[0], shift[1], 0])
                q = self.inverse_kinematics(position=position, rotation=rotation)
                self.motor_step(target_q=q, grip=self.grip)
            if np.linalg.norm(self.data.qpos[self.base_q] - target) > 0.02:
                raise ValueError(f"Base tracking failed in {phase}")
            return
        if elevated and self.held is not None:
            distance = np.linalg.norm(target[:2] - self.data.qpos[self.base_q][:2])
            seconds = max(seconds, float(distance / 0.15))
        if not elevated or self.held != "pitcher":
            super().move_base(target=target, phase=phase, seconds=seconds)
            return
        # Observe compliant handle motion during the long carry. Correct the
        # wrist through motors instead of assuming the handle remains rigidly attached.
        self.phase = phase
        start = self.data.qpos[self.base_q].copy()
        origin, orientation = self.body_pose(name="pitcher")
        count = max(1, round(seconds * self.frequency))
        for index in range(1, count + 11):
            u = min(1.0, index / count)
            self.base_target = start + u * u * (3 - 2 * u) * (target - start)
            actual, rotation = self.body_pose(name="pitcher")
            hand, hand_rotation = self.hand_pose()
            offset = rotation.inv().apply(hand - actual)
            relative = rotation.inv() * hand_rotation
            # Backing away for placement must carry the vessel with the base;
            # holding its world x fixed overextends the arm and twists the handle.
            destination = origin + np.array([
                self.base_target[0] - start[0] if phase == "return base for placement" else 0.0,
                self.base_target[1] - start[1],
                0.0,
            ])
            q = self.inverse_kinematics(
                position=destination + orientation.apply(offset), rotation=orientation * relative
            )
            self.motor_step(target_q=q, grip=1)
        if np.linalg.norm(self.data.qpos[self.base_q] - target) > 0.02:
            raise ValueError(f"Base tracking failed in {phase}")
        position, rotation = self.body_pose(name="pitcher")
        hand, orientation = self.hand_pose()
        self.hand_offset = rotation.inv().apply(hand - position)
        self.rotation_offset = rotation.inv() * orientation

    def pause(self, *, phase: str, seconds: float, grip: float | None = None) -> None:
        if "prep_table" not in self.scene._fixtures_dict or phase != "settle shrimp in tray":
            super().pause(phase=phase, seconds=seconds, grip=grip)
            return
        # A compliant handle is not a rigid attachment. Keep observing through
        # the return instead of letting the vessel sag while holding the wrist.
        center, _ = self.hand_pose()
        self.phase = "stabilize upright pitcher before transport"
        stable = 0
        for _ in range(round(15 * self.frequency)):
            self.tracked_step(center=center, angle=0)
            _, actual = self.body_pose(name="pitcher")
            stable = stable + 1 if actual.magnitude() < np.deg2rad(3) else 0
            if stable >= round(seconds * self.frequency):
                return
        raise ValueError("Pitcher did not stabilize upright after pouring")

    def move_pitcher(
        self, *, phase: str, position: np.ndarray, rotation: Rotation, seconds: float
    ) -> None:
        if "prep_table" not in self.scene._fixtures_dict or phase != "lower pitcher to floor":
            super().move_pitcher(phase=phase, position=position, rotation=rotation, seconds=seconds)
            return
        # Align while clear of the support, then lower onto it before opening.
        for target in (self.home + [0, 0, 0.04], self.home + [0, 0, 0.002]):
            self.phase = (
                "align pitcher above table"
                if target[2] > self.home[2] + 0.01
                else "seat pitcher on table"
            )
            stable = 0
            for _ in range(round(20 * self.frequency)):
                actual, actual_rotation = self.body_pose(name="pitcher")
                hand, hand_rotation = self.hand_pose()
                shift = target - actual
                shift *= min(1.0, 0.01 / max(1e-9, float(np.linalg.norm(shift))))
                turn = (rotation * actual_rotation.inv()).as_rotvec()
                turn *= min(1.0, np.deg2rad(2) / max(1e-9, float(np.linalg.norm(turn))))
                q = self.inverse_kinematics(
                    position=hand + shift,
                    rotation=Rotation.from_rotvec(turn) * hand_rotation,
                )
                self.motor_step(target_q=q, grip=1)
                actual, actual_rotation = self.body_pose(name="pitcher")
                aligned = np.linalg.norm(actual - target) < 0.006 and (
                    rotation * actual_rotation.inv()
                ).magnitude() < np.deg2rad(3)
                stable = stable + 1 if aligned else 0
                if stable >= 5:
                    break
            else:
                raise ValueError("Pitcher placement did not converge; retaining grasp")

    def motor_step(self, *, target_q: np.ndarray, grip: float) -> None:
        if self.recovery_contact_close and self.held in self.scene.cube_names and grip == 1:
            grip = self.payload_hold_grip
        if self.before_step is not None:
            self.before_step()
        before = self.steps
        try:
            super().motor_step(target_q=target_q, grip=grip)
            if self.held is not None and self.held in self.scene.cube_names:
                self.lost_contacts = (
                    0 if len(self.finger_contacts(name=self.held)) == 2 else self.lost_contacts + 1
                )
                if self.lost_contacts > 3:
                    self.held = None
                    raise ValueError("Physical shrimp grasp lost")
        finally:
            if (
                self.steps > before
                and self.shrimp_recovery_active
                and self.phase
                in {
                    "approach above raised shrimp",
                    "approach shrimp",
                    "descend to shrimp body",
                    "lift shrimp",
                }
            ):
                self.shrimp_approach_path.append(self.data.qpos[self.arm_q].copy())
            if self.steps > before and self.after_step is not None:
                self.after_step()

    def ready(self) -> bool:
        return (
            self.held is None
            and self.scene.cube_locations() == ["pitcher"] * len(self.scene.cube_names)
            and self.scene.cubes_settled()
            and self.vessel_home(name="pitcher")
            and self.vessel_home(name="tray")
        )

    def payload_grasp_geometry(self, *, name: str) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self.body_pose(name=name)
        endpoints = np.array([p for a, b, _ in CookedShrimp.segments() for p in (a, b)])
        midpoint = (endpoints.min(axis=0) + endpoints.max(axis=0)) / 2
        return position + rotation.apply(midpoint), rotation.apply([0, 1, 0])

    def pick_shrimp(self, *, index: int, location: str) -> None:
        name = self.scene.cube_names[index]
        if self.held is not None or self.scene.cube_locations()[index] != location:
            raise ValueError("Shrimp pick requires an empty hand and the named source")
        self.shrimp_recovery_active = "prep_table" in self.scene._fixtures_dict
        self.shrimp_approach_path.clear()
        position, _ = self.body_pose(name=name)
        # Enclose the entire curl before closing. Preclosing around one segment
        # brushes its neighbour and pushes the low-friction piece out of the grasp.
        point, across = self.payload_grasp_geometry(name=name)
        yaw = np.arctan2(across[1], across[0])
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if location == "tray":
            tray, orientation = self.body_pose(name="tray")
            geometry = self.scene.get_object("tray")
            clearance = np.array([geometry.length, geometry.width]) / 2 - geometry.wall_thickness
            clearance -= np.abs(orientation.inv().apply(point - tray)[:2])

            def margin(angle: float) -> float:  # noqa: PLR0917
                if isinstance(geometry, CastIronSkillet):
                    corners = np.array([[x, y, 0] for x in (-0.06, 0.06) for y in (-0.02, 0.02)])
                    world = Rotation.from_euler("z", angle).apply(corners) + point
                    local = orientation.inv().apply(world - tray)
                    normals, offsets = geometry.cavity_planes()
                    return float(np.min(offsets - local @ normals.T))
                footprint = np.array([
                    0.06 * abs(np.cos(angle)) + 0.02 * abs(np.sin(angle)),
                    0.06 * abs(np.sin(angle)) + 0.02 * abs(np.cos(angle)),
                ])
                return float(np.min(clearance - footprint))

            if margin(yaw) < 0.005:
                yaw = max((yaw, yaw + np.pi / 2, 0.0, np.pi / 2), key=margin)
        down = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi)
        elevated = "prep_table" in self.scene._fixtures_dict and location != "floor"
        if elevated:
            # Rotate the tilt with the grasp yaw, keeping the closing axis
            # horizontal. Tilting in world y after yaw sends one finger into
            # the support for rotated shrimp. Pick the equivalent yaw that
            # leaves the wrist on the aisle side of the object.
            if yaw > 0:
                yaw -= np.pi
            down = (
                Rotation.from_euler("z", yaw)
                * Rotation.from_euler("x", self.recovery_pitch_degrees, degrees=True)
                * Rotation.from_euler("y", np.pi)
            )
        grasp = (
            point - down.apply([0, 0, self.recovery_finger_offset])
            if elevated
            else point + [0, 0, self.recovery_finger_offset]
        )
        if elevated:
            # Raise the whole arm before moving under the worktop edge. A straight
            # Cartesian approach from the low rest pose catches the wrist below it.
            self.move_base(
                target=np.array([-0.10, point[1], 0]),
                phase="clear aisle for shrimp pick",
                seconds=5,
            )
            self.prepare_empty_arm(
                position=grasp + [-0.30, 0, self.recovery_approach_clearance],
                rotation=down,
            )
        self.move_base(
            target=np.array([self.recovery_base_x if elevated else point[0] - 0.65, point[1], 0]),
            phase="position for shrimp pick",
            seconds=5,
        )
        if elevated:
            self.move(
                position=grasp + [0, 0, self.recovery_approach_clearance],
                rotation=down,
                seconds=6,
                grip=self.recovery_open_grip,
                phase="approach above raised shrimp",
            )
        self.move(
            position=grasp + [0, 0, self.recovery_approach_clearance]
            if elevated
            else point + [0, 0, 0.25],
            rotation=down,
            seconds=5,
            grip=self.recovery_open_grip,
            phase="approach shrimp",
        )
        self.move(
            position=grasp,
            rotation=down,
            seconds=4,
            grip=self.recovery_open_grip,
            phase="descend to shrimp body",
        )
        self.pause(phase="settle shrimp approach", seconds=1, grip=self.recovery_open_grip)
        self.phase = "pinch shrimp"
        self.payload_hold_grip = 1.0
        for grip in np.linspace(self.recovery_open_grip, 1, round(3 * self.frequency) + 1)[1:]:
            self.motor_step(target_q=self.last_q, grip=float(grip))
            if self.recovery_contact_close and len(self.finger_contacts(name=name)) == 2:
                # Position control with a small preload, rather than driving the
                # fingers fully closed through a tapered rigid object.
                self.payload_hold_grip = min(1.0, float(grip) + 0.03)
                break
        self.pause(phase="retain shrimp", seconds=1, grip=self.payload_hold_grip)
        if len(self.finger_contacts(name=name)) != 2:
            raise ValueError("Shrimp grasp lacks both finger contacts")
        self.held, self.lost_contacts = name, 0
        lift_position = grasp + [0, 0, 0.15] if elevated else point + [0, 0, 0.28]
        lift_rotation = down
        if self.recovery_lift_from_actual:
            hand, lift_rotation = self.hand_pose()
            lift_position = hand + [0, 0, 0.15 if elevated else 0.24]
        self.move(
            position=lift_position,
            rotation=lift_rotation,
            seconds=4,
            grip=1,
            phase="lift shrimp",
        )
        if self.body_pose(name=name)[0][2] < position[2] + (0.10 if elevated else 0.20):
            self.held = None
            raise ValueError("Shrimp not retained through lift")

    def clear_failed_pick(self) -> None:
        """Open and retreat physically after a failed empty-hand pickup."""
        if self.held is not None:
            return
        # A blocked approach may leave the arm far from its commanded IK solution.
        # Recovery must start on the measured branch, not that unexecuted target.
        self.last_q = self.data.qpos[self.arm_q].copy()
        self.pause(phase="release failed shrimp pick", seconds=0.5, grip=0)
        if self.shrimp_recovery_active and self.shrimp_approach_path:
            # Reverse the physically executed approach instead of requesting a
            # new vertical IK branch from a possibly singular failure posture.
            path = list(reversed(self.shrimp_approach_path))
            self.shrimp_approach_path.clear()
            self.phase = "retreat after failed shrimp pick"
            for target in path:
                self.last_q = target.copy()
                self.motor_step(target_q=target, grip=0)
                for _ in range(20):
                    if np.max(np.abs(target - self.data.qpos[self.arm_q])) < 0.025:
                        break
                    self.motor_step(target_q=target, grip=0)
                else:
                    raise ValueError("Motor tracking failed while reversing shrimp approach")
            return
        position, rotation = self.hand_pose()
        self.move(
            position=position + [0, 0, 0.20],
            rotation=rotation,
            seconds=3,
            grip=0,
            phase="retreat after failed shrimp pick",
        )

    def drop_shrimp(self, *, index: int) -> None:
        name = self.scene.cube_names[index]
        if self.held != name or not self.vessel_home(name="pitcher"):
            raise ValueError("Drop requires held shrimp and home pitcher")
        pitcher, _ = self.body_pose(name="pitcher")
        _, down = self.hand_pose()
        elevated = "prep_table" in self.scene._fixtures_dict
        hand, down = self.hand_pose()
        if elevated and self.recovery_lift_from_actual and hand[2] < 0.7:
            # Reorient the low floor grasp before raising it to worktop height;
            # a vertically downward wrist reaches its joint limit at that height.
            down = down * Rotation.from_euler("x", self.recovery_floor_carry_angle, degrees=True)
            self.move(
                position=np.array([self.data.qpos[self.base_q][0] + 0.55, hand[1], 0.70]),
                rotation=down,
                seconds=6,
                grip=1,
                phase="reorient held shrimp in aisle",
            )
            hand, down = self.hand_pose()
        if (
            elevated
            and self.recovery_lift_from_actual
            and (hand[2] < 1.0 or hand[0] - self.data.qpos[self.base_q][0] > 0.5)
        ):
            # Floor recovery must clear the worktop before base transport. Keep
            # the retained grasp orientation and tuck the load toward the robot.
            self.move(
                position=np.array([self.data.qpos[self.base_q][0] + 0.45, hand[1], 1.03]),
                rotation=down,
                seconds=8,
                grip=1,
                phase="raise held shrimp in aisle",
            )
        self.move_base(
            target=np.array([0.20 if elevated else pitcher[0] - 0.65, pitcher[1], 0]),
            phase="position for shrimp drop",
            seconds=5,
        )
        # Center the observed payload rather than assuming its offset from the
        # wrist is constant across differently oriented curled-piece grasps.
        destination = pitcher + [0, 0, self.scene.get_object("pitcher").height + 0.07]
        for attempt in range(3):
            point, _ = self.body_pose(name=name)
            hand, _ = self.hand_pose()
            self.move(
                position=hand + destination - point,
                rotation=down,
                seconds=5 if attempt == 0 else 2,
                grip=1,
                phase="center held shrimp over pitcher",
            )
            self.pause(phase="stabilize shrimp before release", seconds=0.5, grip=1)
            if np.linalg.norm(self.body_pose(name=name)[0] - destination) < 0.004:
                break
        if len(self.finger_contacts(name=name)) != 2:
            self.held = None
            raise ValueError("Shrimp lost before release")
        if np.linalg.norm(self.body_pose(name=name)[0] - destination) >= 0.01:
            raise ValueError("Held shrimp did not converge above pitcher")
        self.held = None
        self.pause(phase="release shrimp into pitcher", seconds=3, grip=0)
        if not self.scene.cube_in_vessel(name, "pitcher"):
            raise ValueError("Dropped shrimp did not land inside pitcher")
