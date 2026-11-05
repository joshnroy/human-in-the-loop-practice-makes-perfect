"""Cold rigid-body proxy, visible water, and an explicitly simulated hob interface."""

import json
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController
from hitl_pmp.environments.pouring3d.potsticker_native import PotstickerPourEnv, PotstickerScene

from .config import KitchenGeometry
from .settling import PayloadRestWindow
from .types import ClosedLoopPourParameters


class KitchenScene(PotstickerScene):
    # Above the dumplings' priority 2 and the jug's 1, so MuJoCo takes the skillet's
    # coefficient for every skillet contact and the dumplings' own elsewhere.
    SKILLET_PRIORITY: ClassVar[int] = 3
    # KinDER's 0.005 m let the held jug creep ~0.2 deg/s about the pad-to-pad axis under
    # its own gravity torque, until two pours left it too rolled to place upright.
    HANDLE_TORSIONAL_FRICTION: ClassVar[float] = 0.02
    HANDLE_GEOMS: ClassVar[tuple[str, ...]] = ("pitcher_handle_upper", "pitcher_handle_grip")

    def __init__(self, *args: Any, skillet_mu: float = 0.4, **kwargs: Any) -> None:
        if not np.isfinite(skillet_mu) or skillet_mu < 0:
            raise ValueError("Invalid skillet friction")
        self.skillet_mu = skillet_mu
        self.payload_rest = PayloadRestWindow()
        super().__init__(*args, **kwargs)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        self.payload_rest.clear()
        return super().reset(seed=seed, options=options)

    def step(self, action: Any) -> Any:  # noqa: PLR0917 -- gymnasium protocol
        result = super().step(action)
        # Observation only: recording reads qpos and never changes the trajectory.
        self.record_payload_rest()
        return result

    def payload_positions(self) -> np.ndarray:
        # qpos, not xpos: mj_step leaves derived poses at the preceding instant.
        model, data = self._model_and_data()
        return np.array([
            data.qpos[adr : adr + 3]
            for adr in (
                int(model.jnt_qposadr[int(model.body_jntadr[model.body(name).id])])
                for name in self.cube_names
            )
        ])

    def record_payload_rest(self) -> None:
        _, data = self._model_and_data()
        self.payload_rest.record(time=float(data.time), positions=self.payload_positions())

    # `cubes_settled` stays KinDER's speed rule for construction and the human reset:
    # both start from zeroed velocities with no position history, and the controllers
    # were validated on the start state that rule produces. Outcomes, the drop's settle
    # wait and `ready()` also accept position rest (`KitchenController`): a dumpling
    # rocking on one contact point failed 5/60 drops on the speed rule alone.
    def payload_at_rest(self) -> bool:
        """Outcome rest by position (`PayloadRestWindow`), immune to rocking in place."""
        _, data = self._model_and_data()
        return self.payload_rest.settled(time=float(data.time), positions=self.payload_positions())

    def _check_goals(self) -> bool:
        locations = self.cube_locations()
        return (
            locations.count("tray") == self.requested_count
            and locations.count("pitcher") == len(self.cube_names) - self.requested_count
            and self.payload_at_rest()
        )

    def physical_snapshot(self) -> dict[str, Any]:
        result = super().physical_snapshot()
        result["settled"] = self.payload_at_rest()
        return result

    def _create_scene_xml(self) -> str:
        root = ET.fromstring(super()._create_scene_xml())
        tray = root.find(".//body[@name='tray']")
        assert tray is not None
        for geom in tray.iter("geom"):
            geom.set("priority", str(self.SKILLET_PRIORITY))
            geom.set("friction", f"{self.skillet_mu} 0.005 0.0001")
        for name in self.HANDLE_GEOMS:
            handle = root.find(f".//geom[@name='{name}']")
            assert handle is not None
            sliding, _, rolling = handle.attrib["friction"].split()
            handle.set("friction", f"{sliding} {self.HANDLE_TORSIONAL_FRICTION} {rolling}")
        size = root.find("size")
        if size is None:
            size = ET.SubElement(root, "size")
        size.set("nuserdata", "1")
        pan = self.get_object("tray")
        body = root.find(".//body[@name='tray']")
        assert body is not None
        if not pan.options["water_present"]:
            return ET.tostring(root, encoding="unicode")
        # Visual only: fluid dynamics, added water inertia, buoyancy and heat are
        # not validated. Keeping it non-colliding avoids an invisible solid lid.
        ET.SubElement(
            body,
            "geom",
            name="kitchen_water_proxy",
            type="cylinder",
            size=f"{float(pan.options['bottom_radius']) - 0.005} .0003",
            pos=f"0 0 {pan.options['water_depth_proxy_m']}",
            rgba=".25 .55 .8 .28",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        return ET.tostring(root, encoding="unicode")

    def water_present(self) -> bool:
        import mujoco

        model, _ = self._model_and_data()
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "kitchen_water_proxy") >= 0

    def set_cooking(self, *, enabled: bool) -> None:
        model, data = self._model_and_data()
        data.userdata[0] = float(enabled)
        model.geom_rgba[model.geom("stovetop_burner_0").id] = (
            [0.9, 0.15, 0.05, 1] if enabled else [0.3, 0.31, 0.32, 1]
        )
        # The pan covers the ring. Highlight its existing control too, so the
        # simulated readback remains visible without moving the pan.
        model.geom_rgba[model.geom("stovetop_knob_0").id] = (
            [0.9, 0.15, 0.05, 1] if enabled else [0.12, 0.12, 0.13, 1]
        )

    def cooking_observed(self) -> bool:
        _, data = self._model_and_data()
        return bool(data.userdata[0] == 1)


class KitchenNativeEnv(PotstickerPourEnv):
    def _create_object_centric_env(self, *args: Any, **kwargs: Any) -> KitchenScene:
        return KitchenScene(*args, **kwargs)


class KitchenController(PotstickerController):
    # Match the validated pour reach; transport posture is checked separately.
    station_pour_base_offset: ClassVar[float] = 0.60
    recovery_pitch_degrees: float = 30.0
    recovery_base_x: float = 0.30
    # Episode-initial arm configuration; set by build(). Continuous joints are
    # unwound by whole turns towards it before every empty-hand pick.
    arm_neutral_q: np.ndarray = Field(default_factory=lambda: np.zeros(7))
    # The commanded-delta motors track about 0.2 rad/s; prepare_empty_arm uses the same.
    unwind_speed: float = 0.20
    # Execution speeds of planned motions, within what the motors are measured to track.
    planned_joint_speed: float = 0.20
    planned_base_speed: float = 0.15
    planned_base_turn_speed: float = 0.30
    # A loaded motion within this of a continuous joint's IK bound fails loudly.
    held_bound_margin: ClassVar[float] = np.deg2rad(20.0)
    LEVEL_JUG: ClassVar[str] = "level jug at pour start"
    # Rounds of the closed-loop levelling at the pour start (re-measure, re-aim).
    levelling_rounds: int = 3
    # Two pours delivered the requested count but were still moving at 20 s.
    pour_settle_seconds: float = 30.0
    # The closed-loop Pour's fixed settings (Josh, 2026-11-04); only the ramp rate is
    # learned (`ClosedLoopPourParameters`).
    # Short of the ~86 deg slide onset, so the fast approach moves no dumpling.
    POUR_APPROACH_DEGREES: ClassVar[float] = 80.0
    # The approved tilt ceiling: the wrist (joint 6, about 121 deg) reaches 95 deg.
    POUR_MAX_TILT_DEGREES: ClassVar[float] = 95.0
    # Sideways wiggle during the ramp: the prototype's 40 mm, at which 1 deg/s was exact.
    POUR_WIGGLE_M: ClassVar[float] = 0.04
    # Fast return, so as little as possible leaves after the cutoff.
    POUR_RETURN_DEG_S: ClassVar[float] = 45.0

    def outcome_settled(self) -> bool:
        return bool(self.scene.payload_at_rest())

    def ready_settled(self) -> bool:
        # Position rest admits a dumpling rocking in place; the speed rule still covers
        # construction and the human reset, which have no position history yet.
        return bool(self.scene.payload_at_rest() or self.scene.cubes_settled())

    def settle_returned_payload(self) -> None:
        # By position: the speed rule never sees a dumpling rocking in place as settled.
        for _ in range(200):
            self.pause(phase="settle returned potstickers", seconds=0.1, grip=0)
            if self.scene.payload_at_rest():
                return
        raise ValueError("Returned potstickers did not settle within 20 seconds")

    def run_joint_pour(self, *, parameters: Any, requested_count: int) -> dict:
        del parameters, requested_count
        raise ValueError("The kitchen Pour is closed-loop; use run_closed_loop_pour")

    def run_closed_loop_pour(
        self, *, parameters: ClosedLoopPourParameters, requested_count: int
    ) -> dict:
        """The Pour skill: carry and level to the pour start, then the closed-loop pour."""
        if self.held != "pitcher":
            raise ValueError("Pour requires a retained handle grasp")
        task_request = self.scene.requested_count
        try:
            self.prepare_to_pour()
            result = self.closed_loop_pour(parameters=parameters, requested=requested_count)
            result.update(skill=f"Pour{requested_count}", requested_count=requested_count)
            self.events.append(result)
            return result
        finally:
            self.scene.requested_count = task_request

    def poured_count(self) -> int:
        """Dumplings that have left the jug, from true simulator state (no delay)."""
        return sum(1 for location in self.scene.cube_locations() if location != "pitcher")

    def closed_loop_pour(self, *, parameters: ClosedLoopPourParameters, requested: int) -> dict:
        """Tilt until `requested` dumplings have left the jug, then return upright.

        Approach to `POUR_APPROACH_DEGREES` at the carry tilt speed; past it, tilt at the
        learned ramp rate with a fixed sideways wiggle, up to `POUR_MAX_TILT_DEGREES`.
        The cutoff cannot take back a dumpling already on its way out, so a fast ramp
        overshoots: that is the trade-off the learned rate faces.
        """
        import mujoco

        if self.held != "pitcher":
            raise ValueError("A physical handle grasp is required")
        # mj_step leaves derived poses at the preceding integration instant.
        mujoco.mj_forward(self.model, self.data)
        self.scene.requested_count = requested
        self.cutoff_reason = None
        before = self.scene.physical_snapshot()
        center = self.body_pose(name="tray")[0] + [0, 0, self.pour_clearance]
        self.calibrate_joint_payload()
        self.joint_pour_active = True
        return_speed, self.return_speed_deg_s = self.return_speed_deg_s, self.POUR_RETURN_DEG_S
        angle = elapsed = offset = 0.0
        error = None
        try:
            while angle < self.POUR_MAX_TILT_DEGREES:
                if self.poured_count() >= requested:
                    self.cutoff_reason = f"{requested} left the jug at tilt {angle:.2f} deg"
                    break
                ramping = angle >= self.POUR_APPROACH_DEGREES
                rate = parameters.ramp_rate_deg_s if ramping else self.TILT_SPEED_DEG_S
                angle = min(self.POUR_MAX_TILT_DEGREES, angle + rate / self.frequency)
                if angle >= self.POUR_APPROACH_DEGREES:
                    elapsed += 1 / self.frequency
                    offset = self.offset(elapsed=elapsed, amplitude_m=self.POUR_WIGGLE_M)
                # Not a "feedback ..." phase: the ramp neither waits on joint tracking
                # each waypoint nor widens the IK clearance, exactly as prototyped.
                self.phase = (
                    "closed-loop tilt ramp with sideways wiggle"
                    if ramping
                    else "closed-loop tilt approach"
                )
                self.tracked_step(center=center, angle=angle, offset=offset)
            if self.cutoff_reason is None:
                self.cutoff_reason = "reached max tilt"
            self.return_pitcher_upright(center=center, angle=angle, start_offset=offset)
            # Close the loop on the held jug's measured orientation, as the timed pour does.
            for _ in range(round(15 * self.frequency)):
                actual = self.hand_pose()[1] * self.rotation_offset.inv()
                if np.degrees(np.arccos(np.clip(actual.apply([0, 0, 1])[2], -1, 1))) < 3:
                    break
                self.phase = "feedback return upright"
                self.tracked_step(center=center, angle=0)
            else:
                raise ValueError("Pitcher did not become upright after return")
            self.pause(phase="settle potstickers in pan", seconds=2)
            # Fixed observation window; food state never changes the action.
            self.pause(phase="wait for potstickers to settle", seconds=self.pour_settle_seconds - 2)
        except ValueError as exc:
            error = str(exc)
        finally:
            self.joint_pour_active = False
            self.return_speed_deg_s = return_speed
        after = self.scene.physical_snapshot()
        locations = after["locations"]
        return {
            "mode": "closed_loop",
            "parameters": parameters.model_dump(),
            "requested": requested,
            "success": error is None
            and locations.count("tray") == requested
            and locations.count("pitcher") == len(locations) - requested
            and after["settled"],
            "error": error,
            "delivered": locations.count("tray"),
            "retained": locations.count("pitcher"),
            "locations": locations,
            "settled": after["settled"],
            "cutoff_reason": self.cutoff_reason,
            "cutoff_tilt_degrees": angle,
            "before": before,
            "after": after,
            "fixed_settings": {
                "approach_degrees": self.POUR_APPROACH_DEGREES,
                "approach_speed_deg_s": self.TILT_SPEED_DEG_S,
                "max_tilt_degrees": self.POUR_MAX_TILT_DEGREES,
                "wiggle_m": self.POUR_WIGGLE_M,
                "wiggle_hz": self.FEED_HZ,
                "return_speed_deg_s": self.POUR_RETURN_DEG_S,
                "settle_seconds": self.pour_settle_seconds,
                "observation": "true simulator locations, no delay",
            },
            "controller_revision": "kitchen_closed_loop_v1",
            "bounded_ik_recoveries": self.bounded_ik_recoveries,
        }

    @classmethod
    def build(
        cls,
        *,
        geometry: KitchenGeometry,
        deployment: bool,
        requested: int,
        seed: int,
        scene_bg: bool,
        surface_mu: float,
        skillet_mu: float = 0.4,
    ) -> Any:
        # The file exists through construction/reset, and never replaces an asset
        # in the benchmark checkout. Geometry is also serialized in recordings.
        with tempfile.TemporaryDirectory(prefix="pouring-kitchen-") as folder:
            config = Path(folder) / "scene.json"
            config.write_text(json.dumps(geometry.scene_config(deployment=deployment)))
            env = KitchenNativeEnv(
                task_config_path=str(config),
                payload_count=3,
                requested_count=requested,
                surface_mu=surface_mu,
                skillet_mu=skillet_mu,
                scene_bg=scene_bg,
                render_mode="rgb_array",
            )
            try:
                env.reset(seed=seed)
                controller: Any = cls.create(env=env)
            except BaseException:
                env.close()
                raise
        # The visible water is neither contact geometry nor an IK obstacle.
        controller.pan_pairs = [
            (a, b) for a, b in controller.pan_pairs if controller.model.geom_contype[b] != 0
        ]
        controller.mode = "joint_lateral"
        controller.arm_neutral_q = controller.data.qpos[controller.arm_q].copy()
        return controller

    def continuous_turns(self) -> np.ndarray:
        """Whole turns each continuous arm joint has wound away from its neutral angle.

        Hysteresis: a joint less than a full turn from neutral counts as unwound, so
        one resting near +-180 deg (the wrist after a pour) never flips back and forth.
        """
        continuous = self.lower == -12
        offset = self.data.qpos[self.arm_q] - self.arm_neutral_q
        wound = continuous & (np.abs(offset) >= 2 * np.pi)
        turns = np.zeros(len(offset), dtype=int)
        turns[wound] = np.round(offset[wound] / (2 * np.pi)).astype(int)
        return turns

    def unwind_continuous_joints(self) -> None:
        """Rotate each wound continuous joint back by whole turns, empty hand only.

        IK resolves each continuous joint to the equivalent nearest the previous
        angle, so pick/place cycles ratchet joint 1 by roughly a turn every few
        cycles until it pins at the IK bound (+-12 rad) and strands the jug. A whole
        turn leaves the hand pose unchanged; the joint path between is collision-
        checked in scratch before any motor command, and a blocked path executes
        nothing. The other joints are untouched and there is no home posture.
        """
        if self.held is not None:
            raise ValueError("Unwinding requires an empty hand")
        turns = self.continuous_turns()
        if not turns.any():
            return
        start = self.data.qpos[self.arm_q].copy()
        goal = start - 2 * np.pi * turns
        self.phase = "unwind continuous arm joints"
        self.scratch.qpos[:] = self.data.qpos
        for u in np.linspace(0, 1, 1 + 4 * int(np.ceil(np.max(np.abs(goal - start)) / 0.05))):
            self.scratch.qpos[self.arm_q] = start + u * (goal - start)
            self.update_ik_scratch()
            if np.max(self.obstacle_residual(), initial=0) > 0.005:
                self.events.append({"unwind_blocked": turns.tolist(), "fraction": float(u)})
                return
            if self.scratch.site_xpos[self.site, 2] < 0.25:
                self.events.append({"unwind_blocked": turns.tolist(), "fraction": float(u)})
                return
        seconds = max(2.0, float(np.max(np.abs(goal - start))) / self.unwind_speed)
        count = round(seconds * self.frequency)
        for index in range(1, count + 1):
            u = index / count
            self.last_q = start + u * u * (3 - 2 * u) * (goal - start)
            self.motor_step(target_q=self.last_q, grip=0)
        self.last_q = goal.copy()
        for _ in range(round(10 * self.frequency)):
            if np.max(np.abs(goal - self.data.qpos[self.arm_q])) < 0.025:
                return
            self.motor_step(target_q=goal, grip=0)
        raise ValueError("Arm did not track the unwinding motion")

    def pick_pitcher(self) -> None:
        """Plan the base and the arm to a hover above the handle (PyBullet), follow
        both, then descend, grasp and lift as scripted contact motions."""
        from .planning.parameterized_skills import PickPitcherController
        from .planning.utils import KitchenPyBulletSim

        if self.held is not None:
            raise ValueError("Pitcher pick requires an empty hand")
        self.unwind_continuous_joints()
        if not self.ready():
            raise ValueError("Pitcher pick requires a ready scene")
        self.shrimp_recovery_active = False
        origin, _ = self.body_pose(name="pitcher")
        handle, upright, down = self.pitcher_grasp_pose()
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            planner = PickPitcherController(sim=sim)
            planner.reset(
                x=self.scene._get_object_centric_state(),
                arm=self.data.qpos[self.arm_q].copy(),
                base=self.data.qpos[self.base_q].copy(),
                base_target=self.base_home.copy(),
                hover=handle + [-self.pitcher_hover_backoff, 0, self.pitcher_approach_clearance],
                rotation=upright,
                home=self.arm_neutral_q.copy(),
            )
        finally:
            sim.close()
        if planner.TUCK in planner.plans:
            self.follow_arm_path(
                path=planner.plans[planner.TUCK], phase=planner.TUCK, grip=self.pitcher_open_grip
            )
        self.follow_base_path(path=planner.plans[planner.BASE_MOTION], phase=planner.BASE_MOTION)
        self.follow_arm_path(
            path=planner.plans[planner.MOVE_ARM_TO_HOVER],
            phase=planner.MOVE_ARM_TO_HOVER,
            grip=self.pitcher_open_grip,
        )
        self.grasp_and_lift_pitcher(handle=handle, down=down, origin=origin)

    def level_hand_rotation(self) -> Rotation:
        """The hand orientation that, with the current grasp, holds the jug upright at
        its current heading: the carry pose's orientation."""
        _, jug = self.body_pose(name="pitcher")
        _, hand = self.hand_pose()
        axis = jug.apply([1.0, 0.0, 0.0])
        level = Rotation.from_euler("z", float(np.arctan2(axis[1], axis[0])))
        return level * (jug.inv() * hand)

    def follow_base_path(self, *, path: list[np.ndarray], phase: str) -> None:
        """Drive the base through every waypoint, the arm held where it is."""
        self.phase = phase
        hold = self.data.qpos[self.arm_q].copy()
        for start, end in zip(path[:-1], path[1:], strict=True):
            delta = end - start
            delta[2] = np.arctan2(np.sin(delta[2]), np.cos(delta[2]))
            seconds = max(
                float(np.linalg.norm(delta[:2])) / self.planned_base_speed,
                abs(float(delta[2])) / self.planned_base_turn_speed,
            )
            count = max(1, int(np.ceil(seconds * self.frequency)))
            for index in range(1, count + 1):
                self.base_target = start + delta * (index / count)
                self.motor_step(target_q=hold, grip=self.grip)
        for _ in range(round(5 * self.frequency)):
            if np.linalg.norm(self.data.qpos[self.base_q][:2] - path[-1][:2]) < 0.01:
                break
            self.motor_step(target_q=hold, grip=self.grip)
        else:
            raise ValueError(f"Base did not track the planned path in {phase}")
        self.last_q = hold

    def follow_arm_path(self, *, path: list[np.ndarray], phase: str, grip: float) -> None:
        """Drive the arm along each straight joint segment of a checked plan.

        Continuous joints go the short way round, as upstream's follower does.
        """
        self.phase = phase
        self.grip = grip
        continuous = self.lower == -12
        current = self.data.qpos[self.arm_q].copy()
        for waypoint in path[1:]:
            delta = np.asarray(waypoint, dtype=float) - current
            delta[continuous] = np.arctan2(np.sin(delta[continuous]), np.cos(delta[continuous]))
            count = max(
                1, int(np.ceil(np.max(np.abs(delta)) / self.planned_joint_speed * self.frequency))
            )
            for index in range(1, count + 1):
                self.last_q = current + delta * (index / count)
                self.motor_step(target_q=self.last_q, grip=grip)
            current = current + delta
        for _ in range(round(10 * self.frequency)):
            if np.max(np.abs(current - self.data.qpos[self.arm_q])) < 0.01:
                # A planned path can carry joint 5 across half a turn; the IK that
                # follows in this skill then attracts it on the turn it is now near.
                self.choose_posture_turn()
                return
            self.motor_step(target_q=current, grip=grip)
        raise ValueError(f"Arm did not track the planned path in {phase}")

    def pick_shrimp(self, *, index: int, location: str) -> None:
        if self.held is None:
            self.unwind_continuous_joints()
        super().pick_shrimp(index=index, location=location)

    def inverse_kinematics(self, *, position: np.ndarray, rotation: Rotation) -> np.ndarray:
        previous = self.last_q.copy()
        q = super().inverse_kinematics(position=position, rotation=rotation)
        if self.held is not None:
            continuous = self.lower == -12
            room = np.minimum(q - self.lower, self.upper - q)
            near = np.flatnonzero(continuous & (room < self.held_bound_margin))
            if near.size:
                # Unwinding needs an empty hand, so a loaded arm this wound would
                # pin at the bound and strand what it carries. Say so by name.
                self.last_q = previous
                joint = int(near[0])
                raise ValueError(
                    f"Continuous joint {joint + 1} at {np.rad2deg(q[joint]):.0f} deg is within "
                    f"{np.rad2deg(self.held_bound_margin):.0f} deg of its IK bound "
                    f"({np.rad2deg(self.upper[joint]):.0f} deg) while holding {self.held}"
                )
        return q

    def can_release_pitcher_over_home(self) -> bool:
        if self.held != "pitcher":
            return False
        position, rotation = self.body_pose(name="pitcher")
        joint = self.model.body_jntadr[self.model.body("pitcher").id]
        address = self.model.jnt_dofadr[joint]
        velocity = self.data.qvel[address : address + 6]
        # Permit the observed 7 mm loaded-jug offset while retaining the native
        # post-release landing check (10 mm), support height and speed checks.
        return bool(
            0 <= position[2] - self.home[2] <= 0.02
            and np.linalg.norm(position[:2] - self.home[:2]) < 0.009
            and rotation.magnitude() < np.deg2rad(3)
            and np.linalg.norm(velocity[:3]) < 0.05
            and np.linalg.norm(velocity[3:]) < 0.1
        )

    def place_pitcher(self) -> None:
        """Plan the upright return to a hover above home (PyBullet) and follow it; seat
        and release as scripted contact; then plan and follow the withdrawal."""
        from .planning.parameterized_skills import (
            ClearWorkspaceController,
            PlacePitcherController,
        )
        from .planning.utils import KitchenPyBulletSim

        if self.held != "pitcher":
            raise ValueError("Placement requires holding pitcher")
        # Measure the current compliant grasp before positioning the vessel.
        position, rotation = self.body_pose(name="pitcher")
        hand, orientation = self.hand_pose()
        self.hand_offset = rotation.inv().apply(hand - position)
        self.rotation_offset = rotation.inv() * orientation
        locations = self.scene.cube_locations()
        riders = tuple(
            name
            for name, at in zip(self.scene.cube_names, locations, strict=True)
            if at == "pitcher"
        )
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            planner = PlacePitcherController(sim=sim)
            planner.reset(
                x=self.scene._get_object_centric_state(),
                arm=self.data.qpos[self.arm_q].copy(),
                base=self.data.qpos[self.base_q].copy(),
                base_target=self.base_home.copy(),
                hand_rotation=self.level_hand_rotation(),
                hand_offset=self.hand_offset,
                rotation_offset=self.rotation_offset,
                home=self.home.copy(),
                riders=riders,
            )
        finally:
            sim.close()
        self.follow_arm_path(path=planner.plans[planner.TUCK], phase=planner.TUCK, grip=1)
        self.follow_base_path(path=planner.plans[planner.BASE_MOTION], phase=planner.BASE_MOTION)
        self.follow_arm_path(
            path=planner.plans[planner.MOVE_ARM_TO_HOVER], phase=planner.MOVE_ARM_TO_HOVER, grip=1
        )
        try:
            self.move_pitcher(
                phase="lower pitcher to floor",
                position=self.home + [0, 0, self.placement_clearance],
                rotation=Rotation.identity(),
                seconds=4,
            )
        except ValueError as exc:
            if not self.can_release_pitcher_over_home():
                raise
            # Release only from a measured low, upright pose above support; normal
            # physics and the landing checks determine success.
            self.events.append({"event": "seat fallback", "error": str(exc)})
        self.last_q = self.data.qpos[self.arm_q].copy()
        self.base_target = self.data.qpos[self.base_q].copy()
        self.held = None
        self.pause(phase="release pitcher", seconds=2, grip=0)
        hand, orientation = self.hand_pose()
        self.move(
            phase="withdraw gripper",
            position=hand + [0, 0, 0.25],
            rotation=orientation,
            seconds=3,
            grip=0,
        )
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            clear = ClearWorkspaceController(sim=sim)
            clear.reset(
                x=self.scene._get_object_centric_state(),
                arm=self.data.qpos[self.arm_q].copy(),
                base_target=self.base_home + [-0.35, 0, 0],
            )
        finally:
            sim.close()
        self.follow_base_path(path=clear.plans[clear.BASE_MOTION], phase=clear.BASE_MOTION)

    def active_obstacle_pairs(self) -> list[tuple[int, int]]:
        pairs = super().active_obstacle_pairs()
        if self.held is None and self.phase == "withdraw empty gripper above pitcher":
            pairs = pairs + self.pitcher_pairs
        return pairs

    def obstacle_residual(self) -> np.ndarray:
        previous = self.obstacle_clearance
        if self.phase in {
            "position base for pour",
            "transport pour pivot above tray",
            "level jug at pour start",
            "feedback tilt pitcher",
            "feedback horizontal sideways wiggle",
            "feedback return upright",
            "return base for placement",
        }:
            # Motor tracking and compliant grasp dynamics can consume the
            # nominal IK margin. Keep a larger elbow/worktop gap during carry.
            self.obstacle_clearance = 0.035
        try:
            return super().obstacle_residual()
        finally:
            self.obstacle_clearance = previous

    def motor_step(self, *, target_q: np.ndarray, grip: float) -> None:
        super().motor_step(target_q=target_q, grip=grip)
        if self.held == "pitcher" and self.phase in {
            "fold and raise pitcher for transport",
            "position base for pour",
            "level jug at pour start",
            "transport pour pivot above tray",
            "return base for placement",
        }:
            # Advancing waypoints while the loaded arm lags can consume even a
            # generous geometric margin. Wait on joint feedback, charging every tick.
            for _ in range(30):
                if np.max(np.abs(target_q - self.data.qpos[self.arm_q])) < 0.015:
                    break
                super().motor_step(target_q=target_q, grip=grip)
            else:
                raise ValueError(f"Loaded arm failed to track transport waypoint: {self.phase}")

    def move(
        self, *, phase: str, position: np.ndarray, rotation: Rotation, seconds: float, grip: float
    ) -> None:
        if phase == "withdraw empty gripper above pitcher":
            # The release is already above the mouth. A further vertical rise
            # can hit the wrist limit; retract horizontally with jug collision checks.
            hand, rotation = self.hand_pose()
            position = hand + [-0.10, 0, 0]
            self.last_q = self.data.qpos[self.arm_q].copy()
        super().move(phase=phase, position=position, rotation=rotation, seconds=seconds, grip=grip)

    def prepare_to_pour(self) -> np.ndarray:
        """Plan the whole carry to the pour start upright (PyBullet), then follow it.

        Replaces the scripted fold, base move and pivot transport. Returns the pour's
        pivot, as the scripted version did.
        """
        from .planning.parameterized_skills import CarryToPourController
        from .planning.utils import KitchenPyBulletSim

        tray, _ = self.body_pose(name="tray")
        center = tray + [0, 0, self.pour_clearance]
        locations = self.scene.cube_locations()
        riders = tuple(
            name
            for name, at in zip(self.scene.cube_names, locations, strict=True)
            if at == "pitcher"
        )
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            planner = CarryToPourController(sim=sim)
            planner.reset(
                x=self.scene._get_object_centric_state(),
                arm=self.data.qpos[self.arm_q].copy(),
                base=self.data.qpos[self.base_q].copy(),
                base_target=np.array([center[0] - self.pour_base_offset, center[1], 0.0]),
                hand_rotation=self.level_hand_rotation(),
                start=center,
                start_rotation=self.pour_orientation(angle=0) * self.rotation_offset,
                riders=riders,
            )
        finally:
            sim.close()
        self.follow_arm_path(path=planner.plans[planner.TUCK], phase=planner.TUCK, grip=1)
        self.follow_base_path(path=planner.plans[planner.BASE_MOTION], phase=planner.BASE_MOTION)
        self.follow_arm_path(
            path=planner.plans[planner.MOVE_ARM_TO_POUR], phase=planner.MOVE_ARM_TO_POUR, grip=1
        )
        self.posture_goal = self.pour_posture
        self.level_jug_at_pour_start(center=center)
        return center

    def level_jug_at_pour_start(self, *, center: np.ndarray) -> None:
        """Close the loop on the jug's observed pose at the pour start, as the scripted
        carry does: re-measure the grasp, then turn the hand so the jug stands at the
        pour's upright orientation. The planner carries the jug as rigid; the compliant
        grasp leaves it hanging several degrees off, which this removes."""
        upright = self.pour_orientation(angle=0)
        for _ in range(self.levelling_rounds):
            jug, jug_rotation = self.body_pose(name="pitcher")
            hand, hand_rotation = self.hand_pose()
            if (jug_rotation * upright.inv()).magnitude() < np.deg2rad(1.0):
                return
            self.hand_offset = jug_rotation.inv().apply(hand - jug)
            self.rotation_offset = jug_rotation.inv() * hand_rotation
            self.move(
                phase=self.LEVEL_JUG,
                position=center,
                rotation=upright * self.rotation_offset,
                seconds=3,
                grip=1,
            )

    def move_base(self, *, target: np.ndarray, phase: str, seconds: float) -> None:
        """Preflight the carried arm along the base path, before issuing motors."""
        self.check_base_path(target=target, phase=phase)
        super().move_base(target=target, phase=phase, seconds=seconds)

    def check_base_path(self, *, target: np.ndarray, phase: str) -> None:
        import mujoco

        self.phase = phase
        start = self.data.qpos[self.base_q].copy()
        pairs = list(self.active_obstacle_pairs())
        jug_address = None
        if self.held == "pitcher":
            joint = self.model.body_jntadr[self.model.body("pitcher").id]
            jug_address = self.model.jnt_qposadr[joint]
            jug = [
                g
                for g in range(self.model.ngeom)
                if self.model.body(int(self.model.geom_bodyid[g])).name == "pitcher"
                and self.model.geom_contype[g]
            ]
            obstacles = [
                g
                for g in range(self.model.ngeom)
                if self.model.body(int(self.model.geom_bodyid[g])).name
                in {"prep_table", "stovetop", "tray"}
                and self.model.geom_contype[g]
            ]
            pairs.extend((a, b) for a in jug for b in obstacles)
        ids = np.asarray(pairs)
        a, b = ids[:, 0], ids[:, 1]
        previous_q, previous_base = self.last_q.copy(), self.base_target.copy()
        previous_posture = self.posture
        hand, hand_rotation = self.hand_pose()
        if self.held == "pitcher":
            origin, orientation = self.body_pose(name="pitcher")
            offset = orientation.inv().apply(hand - origin)
            relative = orientation.inv() * hand_rotation
        try:
            for u in np.linspace(0.0, 1.0, 61):
                base = start + u * (target - start)
                shift = base - start
                self.base_target = base.copy()
                self.scratch.qpos[:] = self.data.qpos
                if self.held == "pitcher":
                    # Match PracticePourController's nominal observed-vessel
                    # carry: world x is held except on the return to placement.
                    destination = origin + [
                        shift[0] if phase == "return base for placement" else 0.0,
                        shift[1],
                        0.0,
                    ]
                    q = self.inverse_kinematics(
                        position=destination + orientation.apply(offset),
                        rotation=orientation * relative,
                    )
                    assert jug_address is not None
                    self.scratch.qpos[jug_address : jug_address + 3] = destination
                    xyzw = orientation.as_quat()
                    self.scratch.qpos[jug_address + 3 : jug_address + 7] = np.r_[xyzw[3], xyzw[:3]]
                elif self.recovery_lift_from_actual and phase == "position for shrimp pick":
                    q = self.inverse_kinematics(
                        position=hand + [0.4 * shift[0], shift[1], 0.0], rotation=hand_rotation
                    )
                else:
                    q = previous_q
                self.scratch.qpos[self.arm_q] = q
                self.scratch.qpos[self.base_q] = base
                self.update_ik_scratch()
                bound = np.linalg.norm(
                    self.scratch.geom_xpos[a] - self.scratch.geom_xpos[b], axis=1
                )
                bound -= self.model.geom_rbound[a] + self.model.geom_rbound[b]
                for i in np.flatnonzero(bound < 0.005):
                    distance = mujoco.mj_geomDistance(
                        self.model, self.scratch, int(a[i]), int(b[i]), 0.005, None
                    )
                    if distance < 0.005 - 1e-9:
                        names = [
                            self.model.body(int(self.model.geom_bodyid[g])).name
                            for g in (a[i], b[i])
                        ]
                        raise ValueError(
                            f"Base path lacks 5 mm obstacle clearance: {phase}, "
                            f"{names}, {distance:.4f} m at {u:.2f}"
                        )
        finally:
            self.last_q, self.base_target = previous_q, previous_base
            self.posture = previous_posture

    def prepare_empty_arm(
        self,
        *,
        position: np.ndarray,
        rotation: Rotation,
        continuation: Callable[[np.ndarray], bool] | None = None,
    ) -> None:
        if self.shrimp_recovery_active and continuation is not None:
            # Prepare closer to the aisle; the smaller pan's far-side pieces
            # otherwise put the initial hover at the arm's reach limit.
            position = position + [-0.10, 0, 0]
        super().prepare_empty_arm(position=position, rotation=rotation, continuation=continuation)

    def payload_approach_feasible(
        self,
        *,
        q: np.ndarray,
        preparation: np.ndarray,
        grasp: np.ndarray,
        rotation: Rotation,
        base: np.ndarray,
    ) -> bool:
        if not super().payload_approach_feasible(
            q=q, preparation=preparation + [-0.10, 0, 0], grasp=grasp, rotation=rotation, base=base
        ):
            return False
        previous, old_base, phase = self.last_q.copy(), self.base_target.copy(), self.phase
        posture = self.posture
        try:
            # A reachable pinch alone is insufficient: require a continuation
            # through the lift before committing to the grasp branch.
            self.last_q = self.payload_approach_plan[-1][1].copy()
            self.base_target = base.copy()
            self.phase = "lift shrimp"
            for height in np.linspace(0.0, 0.18, 19)[1:]:
                self.inverse_kinematics(position=grasp + [0, 0, height], rotation=rotation)
            return True
        except ValueError:
            self.payload_approach_plan.clear()
            return False
        finally:
            self.last_q, self.base_target, self.phase = previous, old_base, phase
            self.posture = posture

    def move_away(self, *, distance: float) -> None:
        if self.held is not None or not self.vessel_home(name="pitcher"):
            raise ValueError("Put the jug down before moving away")
        target = self.base_home.copy()
        target[0] -= distance
        self.move_base(target=target, phase="leave kitchen work area", seconds=4)
        if np.linalg.norm(self.data.qpos[self.base_q][:2] - target[:2]) > 0.02:
            raise ValueError("Robot did not reach the observed away pose")
