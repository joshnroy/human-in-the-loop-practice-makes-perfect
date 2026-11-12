"""Cold rigid-body proxy, visible water, and an explicitly simulated hob interface."""

import json
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import Field
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hitl_pmp.core.control_steps import SimulatorDiverged
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

    # KinDER's robot MJCF sets `implicitfast`, which drops the gyroscopic term from the
    # implicit derivative: a dumpling spinning past ~100 rad/s on the skillet or floor
    # gains energy with no torque on it until MuJoCo's guard wipes the scene (Della sweep
    # 16997464). The robot XML's <option> is merged in after the scene's, so the override
    # goes on the compiled model, before the first step (Josh, 2026-11-07).
    INTEGRATOR: ClassVar[str] = "mjINT_IMPLICIT"

    def _initialize_object_poses(self) -> None:
        import mujoco

        model, _ = self._model_and_data()
        model.opt.integrator = getattr(mujoco.mjtIntegrator, self.INTEGRATOR)
        super()._initialize_object_poses()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        self.payload_rest.clear()
        result = super().reset(seed=seed, options=options)
        self.release_payloads()
        self.release_pitcher_grasp()
        self.weld_skillet()
        self.weld_pitcher_home()
        return result

    # The skillet is welded to the stove where the episode placed it (Josh, 2026-11-06),
    # so no pour, pick or spill can move it and `TrayHome` is always true. A weld
    # equality to the world rather than #726's mocap body: the skillet keeps its free
    # joint, so qpos keeps its layout and every recorded state still restores.
    SKILLET_WELD: ClassVar[str] = "skillet_weld"
    PITCHER_HOME_WELD: ClassVar[str] = "pitcher_home_weld"

    def weld_pitcher_home(self) -> None:
        """Anchor the jug at its measured parked pose, without assigning qpos.

        The controller admits this only at initial reset, after human replacement,
        or after a measured home landing. Pick-and-place support is assumed reliable;
        this constraint is disabled from verified handle grasp through pouring.
        """
        import mujoco

        model, data = self._model_and_data()
        if self.pitcher_attached():
            raise ValueError("Cannot anchor a jug that is welded to the gripper")
        mujoco.mj_forward(model, data)
        weld = model.equality(self.PITCHER_HOME_WELD).id
        jug = model.body("pitcher").id
        model.eq_data[weld, :3] = 0.0
        model.eq_data[weld, 3:6] = data.xpos[jug]
        model.eq_data[weld, 6:10] = data.xquat[jug]
        data.eq_active[weld] = 1
        mujoco.mj_forward(model, data)

    def release_pitcher_home(self) -> None:
        model, data = self._model_and_data()
        data.eq_active[model.equality(self.PITCHER_HOME_WELD).id] = 0

    def attach_pitcher(self) -> None:
        """Transfer the verified handle grasp from home support to the hand."""
        self.release_pitcher_home()
        self.attach_payload(name="pitcher")

    def pitcher_attached(self) -> bool:
        _, data = self._model_and_data()
        return bool(data.eq_active[self.grasp_weld(name="pitcher")])

    def release_pitcher_grasp(self) -> None:
        _, data = self._model_and_data()
        data.eq_active[self.grasp_weld(name="pitcher")] = 0

    def weld_skillet(self) -> None:
        import mujoco

        model, data = self._model_and_data()
        mujoco.mj_forward(model, data)
        weld = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, self.SKILLET_WELD))
        pan = model.body("tray").id
        model.eq_data[weld, :3] = 0.0
        model.eq_data[weld, 3:6] = data.xpos[pan]
        model.eq_data[weld, 6:10] = data.xquat[pan]
        data.eq_active[weld] = 1
        mujoco.mj_forward(model, data)

    # A dumpling the gripper has closed on is held rigidly until it is released: a
    # weld from the hand to the piece, frozen at the pose it was grasped in (Josh,
    # 2026-11-05). Grasp quality is not what this experiment measures, so a pick can
    # fail only by refusing to plan or by not reaching the piece, never by slipping.
    def grasp_weld(self, *, name: str) -> int:
        import mujoco

        model, _ = self._model_and_data()
        return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_weld_{name}"))

    def attach_payload(self, *, name: str) -> None:
        import mujoco

        model, data = self._model_and_data()
        hand, piece = model.body(f"{self.robot_name}_bracelet_link").id, model.body(name).id
        hand_rotation = data.xmat[hand].reshape(3, 3)
        weld = self.grasp_weld(name=name)
        model.eq_data[weld, 3:6] = hand_rotation.T @ (data.xpos[piece] - data.xpos[hand])
        relative = np.zeros(4)
        hand_inverse = np.zeros(4)
        mujoco.mju_negQuat(hand_inverse, data.xquat[hand])
        mujoco.mju_mulQuat(relative, hand_inverse, data.xquat[piece])
        model.eq_data[weld, 6:10] = relative
        data.eq_active[weld] = 1

    def attached_payload(self) -> str | None:
        _, data = self._model_and_data()
        return next(
            (name for name in self.cube_names if data.eq_active[self.grasp_weld(name=name)]),
            None,
        )

    def release_payloads(self) -> None:
        _, data = self._model_and_data()
        for name in self.cube_names:
            data.eq_active[self.grasp_weld(name=name)] = 0

    def respawn_payload_in_pitcher(self) -> None:
        self.release_payloads()
        super().respawn_payload_in_pitcher()
        # A respawn moves every piece in zero sim time: positions from before it are not
        # this trajectory's past, so rest by position starts again (Settled falls back
        # to the speed rule until the window refills).
        self.payload_rest.clear()

    def step(self, action: Any) -> Any:  # noqa: PLR0917 -- gymnasium protocol
        _, data = self._model_and_data()
        time, warnings = float(data.time), self.instability_warnings()
        result = super().step(action)
        self.scene_steps += 1
        self.check_simulator_stable(time=time, warnings=warnings)
        # Observation only: recording reads qpos and never changes the trajectory.
        self.record_payload_rest()
        return result

    # The warnings MuJoCo's guard raises just before it calls `mj_resetData`: every
    # body to `qpos0`, every equality (the skillet weld) to its default, time to 0.
    # The simulation then carries on from that wiped scene, which no skill caused.
    INSTABILITY_WARNINGS: ClassVar[tuple[str, ...]] = (
        "mjWARN_BADQPOS",
        "mjWARN_BADQVEL",
        "mjWARN_BADQACC",
    )
    scene_steps: int = 0

    def instability_warnings(self) -> tuple[int, ...]:
        import mujoco

        _, data = self._model_and_data()
        return tuple(
            int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
            for name in self.INSTABILITY_WARNINGS
        )

    def check_simulator_stable(self, *, time: float, warnings: tuple[int, ...]) -> None:
        """Raise `SimulatorDiverged` if the guard fired during the step from `time`.

        Read-only. `mj_resetData` clears the warning counters before the guard counts
        its own warning again, so a count that changed at all, or time that went
        backwards, means it fired."""
        import mujoco

        model, data = self._model_and_data()
        after = self.instability_warnings()
        if after == warnings and float(data.time) >= time:
            return
        changed = [i for i, (a, b) in enumerate(zip(after, warnings, strict=True)) if a != b]
        name = self.INSTABILITY_WARNINGS[changed[-1] if changed else -1]
        index = int(data.warning[int(getattr(mujoco.mjtWarning, name))].lastinfo)
        if name == "mjWARN_BADQPOS":
            # `lastinfo` is a qpos address; the joint is the last one starting at or before it.
            joint = int(np.searchsorted(model.jnt_qposadr, index, side="right")) - 1
            body_id = int(model.jnt_bodyid[joint]) if joint >= 0 else -1
        else:
            body_id = int(model.dof_bodyid[index]) if 0 <= index < model.nv else -1
        raise SimulatorDiverged(
            time=time,
            time_after_reset=float(data.time),
            warning=name,
            index=index,
            body=model.body(body_id).name if body_id >= 0 else None,
            scene_step=self.scene_steps,
        )

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

    def payload_resting(self) -> np.ndarray | None:
        """Per piece, whether its centre stayed within the rest threshold over the
        window; None until the recorded history spans the whole window."""
        _, data = self._model_and_data()
        time = float(data.time)
        if not self.payload_rest.covers(time=time):
            return None
        return self.payload_rest.resting(time=time, positions=self.payload_positions())

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
        equality = root.find("equality")
        if equality is None:
            equality = ET.SubElement(root, "equality")
        ET.SubElement(
            equality,
            "weld",
            name=self.SKILLET_WELD,
            body1="world",
            body2="tray",
            active="false",
            solref="0.002 1",
        )
        ET.SubElement(
            equality,
            "weld",
            name=self.PITCHER_HOME_WELD,
            body1="world",
            body2="pitcher",
            active="false",
            solref="0.002 1",
        )
        for name in (*self.cube_names, "pitcher"):
            ET.SubElement(
                equality,
                "weld",
                name=f"grasp_weld_{name}",
                body1=f"{self.robot_name}_bracelet_link",
                body2=name,
                active="false",
            )
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
    # Episode-initial jug and skillet free-joint poses and open finger joints; set by
    # build(). The human reset stands the vessels back at the former two and releases
    # the hand to the latter.
    jug_home_qpos: np.ndarray = Field(default_factory=lambda: np.zeros(7))
    skillet_home_qpos: np.ndarray = Field(default_factory=lambda: np.zeros(7))
    open_finger_qpos: np.ndarray = Field(default_factory=lambda: np.zeros(0))
    # The commanded-delta motors track about 0.2 rad/s; prepare_empty_arm uses the same.
    unwind_speed: float = 0.20
    # Execution speeds of planned motions, within what the motors are measured to track
    # (`planned_joint_speed` is the arm's, on PotstickerController).
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
        # The pour moves joint 5's posture attractor to `pour_posture`. Left there, it
        # drags the wrist in every scripted move after the pour: the place's lift off the
        # handle turned the open fingers and spun a released empty jug 17.8 deg off home.
        posture = (self.posture, self.posture_goal)
        try:
            self.prepare_to_pour()
            result = self.closed_loop_pour(parameters=parameters, requested=requested_count)
            result.update(skill=f"Pour{requested_count}", requested_count=requested_count)
            self.events.append(result)
            return result
        finally:
            self.scene.requested_count = task_request
            self.posture, self.posture_goal = posture

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
        controller.jug_home_qpos = controller.data.qpos[controller.free_qpos(name="pitcher")].copy()
        controller.skillet_home_qpos = controller.data.qpos[
            controller.free_qpos(name="tray")
        ].copy()
        controller.open_finger_qpos = controller.data.qpos[controller.finger_qpos()].copy()
        return controller

    def free_qpos(self, *, name: str) -> slice:
        """A free body's 7 qpos entries (position, quaternion)."""
        address = int(self.model.jnt_qposadr[self.model.body_jntadr[self.model.body(name).id]])
        return slice(address, address + 7)

    def free_dofs(self, *, name: str) -> slice:
        address = int(self.model.jnt_dofadr[self.model.body_jntadr[self.model.body(name).id]])
        return slice(address, address + 6)

    def human_stands_vessel_home(self, *, name: str) -> None:
        """Stand a vessel ("pitcher" or "tray") at its episode-initial pose, at rest."""
        import mujoco

        home = self.jug_home_qpos if name == "pitcher" else self.skillet_home_qpos
        self.data.qpos[self.free_qpos(name=name)] = home
        self.data.qvel[self.free_dofs(name=name)] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def finger_joints(self) -> list[int]:
        """The gripper's linkage joints: every robot hinge that is not the base or arm."""
        robot = self.scene.robot_name
        taken = {*np.atleast_1d(self.base_q).tolist(), *np.atleast_1d(self.arm_q).tolist()}
        return [
            j
            for j in range(self.model.njnt)
            if self.model.joint(j).name.startswith(f"{robot}_")
            and int(self.model.jnt_qposadr[j]) not in taken
        ]

    def finger_qpos(self) -> np.ndarray:
        return np.array([int(self.model.jnt_qposadr[j]) for j in self.finger_joints()], dtype=int)

    def human_takes_jug_home(self) -> None:
        """The human reset's hand-off: take whatever the hand holds, open the hand, and
        stand the jug upright at its episode-initial pose, at rest. The robot's base and
        arm are not moved; the food is respawned in the jug afterwards (`respawn_...`)."""
        self.scene.release_payloads()
        self.scene.release_pitcher_grasp()
        self.held = None
        self.grip = 0.0
        self.data.qpos[self.finger_qpos()] = self.open_finger_qpos
        self.data.qvel[[int(self.model.jnt_dofadr[j]) for j in self.finger_joints()]] = 0.0
        self.human_stands_vessel_home(name="pitcher")
        self.scene.weld_pitcher_home()

    def continuous_turns(self, *, threshold: float = 2 * np.pi) -> np.ndarray:
        """Whole turns each continuous arm joint has wound away from its neutral angle.

        Hysteresis: by default a joint less than a full turn from neutral counts as
        unwound, so one resting near +-180 deg (the wrist after a pour) never flips back
        and forth. PickPitcher passes half a turn (`pick_pitcher`).
        """
        continuous = self.lower == -12
        offset = self.data.qpos[self.arm_q] - self.arm_neutral_q
        wound = continuous & (np.abs(offset) >= threshold)
        turns = np.zeros(len(offset), dtype=int)
        turns[wound] = np.round(offset[wound] / (2 * np.pi)).astype(int)
        return turns

    def unwind_continuous_joints(self, *, threshold: float = 2 * np.pi) -> None:
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
        turns = self.continuous_turns(threshold=threshold)
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
        # The pick, carry and pour each move the arm before the next unwind is possible,
        # so they start within half a turn of neutral, not merely within a full one.
        self.unwind_continuous_joints(threshold=np.pi)
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

    def pitcher_grasp_acquired(self) -> None:
        """Retain a verified handle grasp throughout lifting, carrying and pouring."""
        self.scene.attach_pitcher()

    def grasp_and_lift_pitcher(
        self, *, handle: np.ndarray, down: Rotation, origin: np.ndarray
    ) -> None:
        try:
            super().grasp_and_lift_pitcher(handle=handle, down=down, origin=origin)
        finally:
            if self.held != "pitcher":
                self.scene.release_pitcher_grasp()
                position, _ = self.body_pose(name="pitcher")
                if self.vessel_home(name="pitcher") and abs(position[2] - self.home[2]) <= 0.003:
                    self.anchor_parked_pitcher()

    def anchor_parked_pitcher(self) -> None:
        """Anchor a landed, home-aligned jug before the empty gripper withdraws."""
        position, _ = self.body_pose(name="pitcher")
        if (
            self.held == "pitcher"
            or not self.vessel_home(name="pitcher")
            or abs(position[2] - self.home[2]) > 0.003
        ):
            raise ValueError("Cannot anchor a pitcher that has not landed at home")
        self.scene.weld_pitcher_home()

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

    # A pick from the pan that plans no approach refuses with this, before any motion.
    PAN_PICK_REFUSAL: ClassVar[str] = "No planned pick from the pan"
    # The hover above the grasp, and how finely the vertical descent and lift are solved.
    PAN_PICK_HOVER_M: ClassVar[float] = 0.16
    PAN_PICK_STEP_M: ClassVar[float] = 0.01
    # A straight rise that stops short of the hover (the wrist at its limit) is kept if it
    # clears the dumpling by this much; the planner takes the lift from there.
    PAN_PICK_MIN_STRAIGHT_M: ClassVar[float] = 0.04
    # The planned remainder of the lift: the grasp's own orientation, then tilted away
    # from the base by these.
    PAN_LIFT_TILTS_DEG: ClassVar[tuple[float, ...]] = (0.0, 20.0, 40.0)
    # Grasp yaws, as turns of the yaw across the dumpling's long axis: that yaw and its
    # 180 deg equivalent first, then turned 45 and 90 deg (pinching more along the piece).
    PAN_PICK_YAW_TURNS_DEG: ClassVar[tuple[float, ...]] = (0.0, 180.0, 45.0, -45.0, 90.0, 225.0)
    # Multi-start IK for the grasp configuration, as `prepare_empty_arm` searches.
    PAN_PICK_IK_STARTS: ClassVar[int] = 16
    # A pick from the floor that plans no approach refuses with this, before any motion.
    FLOOR_PICK_REFUSAL: ClassVar[str] = "No planned pick from the floor"
    # Base poses offered for a floor pick: the base centre this far from the dumpling,
    # facing it from each heading in turn. The scripted pick's pose, from -x, comes first.
    FLOOR_PICK_REACH_M: ClassVar[float] = 0.65
    FLOOR_PICK_HEADINGS_DEG: ClassVar[tuple[float, ...]] = (
        0.0,
        45.0,
        -45.0,
        90.0,
        -90.0,
        135.0,
        -135.0,
        180.0,
    )

    def pick_payload_with_retry(self, *, index: int, location: str) -> None:
        """From the pan and from the floor the pick is planned (`pick_from_pan`,
        `pick_from_floor`); elsewhere it is unchanged."""
        if location == "tray":
            self.pick_from_pan(index=index)
        elif location == "floor":
            self.pick_from_floor(index=index)
        else:
            super().pick_payload_with_retry(index=index, location=location)

    def pan_pick_rotations(self, *, name: str, heading: float = 0.0) -> list[tuple[str, Rotation]]:
        """Grasp orientations in the order tried: each yaw of `PAN_PICK_YAW_TURNS_DEG` at
        the configured, the top-down and the steep pitch (the retry's three pitches).

        This replaces the pan footprint rule (`PracticePourController.pick_shrimp`), which
        near the wall chose the least-negative footprint margin, yaw + 90 deg in 3/3
        stranded runs, a pose with no IK solution at any pitch. Here each orientation is
        checked by IK and planning instead, with the pads allowed to touch the pan.

        `heading` is the base's yaw at the pick (the pan pick's is 0): yaws are taken
        relative to it, so the wrist starts on the robot's side of the dumpling.
        """
        _, across = self.payload_grasp_geometry(name=name)
        natural = float(np.arctan2(across[1], across[0])) - heading
        natural = (natural + np.pi / 2) % np.pi - np.pi / 2
        if natural > 0:
            # The wrist on the aisle side of the dumpling, as the scripted pick takes it.
            natural -= np.pi
        natural += heading
        pitch = self.recovery_pitch_degrees
        # The configured tilt first: the drop's planned carry finds no carry pose for a
        # top-down grasp (measured at the pan centre, 2026-11-09), and the scripted pick
        # reached the tilt there too, after its top-down candidate failed.
        pitches = (pitch, 0.0, 30.0 if pitch > 45 else 60.0)
        return [
            (
                f"yaw {turn:+.0f} pitch {tilt:.0f}",
                Rotation.from_euler("z", natural + np.deg2rad(turn))
                * Rotation.from_euler("x", tilt, degrees=True)
                * Rotation.from_euler("y", np.pi),
            )
            for turn in self.PAN_PICK_YAW_TURNS_DEG
            for tilt in pitches
        ]

    def pan_pick_chain(
        self, *, grasp: np.ndarray, rotation: Rotation, base: np.ndarray
    ) -> list[np.ndarray] | str:
        """Arm configurations from the grasp straight up to the hover, with the base at
        `base`, or why there are none. Solved in scratch upwards from a multi-start grasp
        solution with the descent's own IK and clearance (pads may touch the pan, the arm
        and wrist keep their margin), so the descent replayed downwards reaches the grasp
        on the branch it was checked on, and the lift replays it upwards.

        When no grasp solution rises straight all the way, the longest straight rise that
        clears `PAN_PICK_MIN_STRAIGHT_M` is returned short of the hover, for the planner
        to complete (`plan_pick_from_pan`)."""
        saved = (self.last_q.copy(), self.base_target.copy(), self.phase, self.posture)
        steps = round(self.PAN_PICK_HOVER_M / self.PAN_PICK_STEP_M)
        heights = self.PAN_PICK_STEP_M * np.arange(1, steps + 1, dtype=float)
        try:
            self.phase = "descend to shrimp body"
            self.base_target = np.asarray(base, dtype=float).copy()

            def residual(q: np.ndarray) -> np.ndarray:  # noqa: PLR0917 -- scipy callback
                self.scratch.qpos[self.arm_q] = q
                self.update_ik_scratch()
                actual = Rotation.from_matrix(self.scratch.site_xmat[self.site].reshape(3, 3))
                return np.r_[
                    self.scratch.site_xpos[self.site] - grasp,
                    0.22 * (rotation * actual.inv()).as_rotvec(),
                    self.obstacle_residual(),
                ]

            rng = np.random.default_rng(0)
            neutral = self.arm_neutral_q
            continuous = self.lower == -12
            best, reason = float("inf"), ""
            chains: list[list[np.ndarray]] = []
            partial: list[list[np.ndarray]] = []
            for start in range(self.PAN_PICK_IK_STARTS):
                self.scratch.qpos[:] = self.data.qpos
                self.scratch.qpos[self.base_q] = self.base_target
                if start == 0:
                    seed = neutral.copy()
                elif start == 1:
                    seed = self.data.qpos[self.arm_q].copy()
                else:
                    seed = rng.uniform(
                        np.maximum(self.lower, -np.pi), np.minimum(self.upper, np.pi)
                    )
                fit = least_squares(
                    residual,
                    np.clip(seed, self.lower + 1e-6, self.upper - 1e-6),
                    bounds=(self.lower, self.upper),
                    max_nfev=200,
                )
                error = float(np.linalg.norm(residual(fit.x)))
                best = min(best, error)
                if error > self.IK_TOLERANCE_M:
                    continue
                q = fit.x.copy()
                turn = q[continuous] - neutral[continuous]
                q[continuous] = neutral[continuous] + np.arctan2(np.sin(turn), np.cos(turn))
                if any(np.linalg.norm(q - chain[0]) < 0.05 for chain in chains):
                    continue
                self.last_q = q.copy()
                rise: list[np.ndarray] = []
                try:
                    for h in heights:
                        rise.append(
                            self.inverse_kinematics(position=grasp + [0, 0, h], rotation=rotation)
                        )
                except ValueError as exc:
                    reason = f"; no straight rise from a grasp solution ({exc})"
                    if len(rise) * self.PAN_PICK_STEP_M >= self.PAN_PICK_MIN_STRAIGHT_M - 1e-9:
                        partial.append([q, *rise])
                    continue
                chains.append([q, *rise])
            # The branch nearest the episode-initial arm: the one the drop's carry and
            # the other skills start from. A flipped elbow or wrist reaches the same
            # grasp, but the welded carry and the in-place release fail from it.
            if chains:
                return min(chains, key=lambda chain: float(np.linalg.norm(chain[0] - neutral)))
            if partial:
                return min(
                    partial,
                    key=lambda chain: (-len(chain), float(np.linalg.norm(chain[0] - neutral))),
                )
            return f"no grasp IK within {self.IK_TOLERANCE_M * 1000:.0f} mm (best {best:.4f})" + (
                reason
            )
        finally:
            self.last_q, self.base_target, self.phase, self.posture = saved

    def plan_pick_from_pan(
        self, *, name: str, base_target: np.ndarray
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray], np.ndarray, str]:
        """Plan the whole pick before anything moves: the planned free-space motions,
        the grasp-to-hover chain, the grasp position and the orientation's label."""
        plans, chain, grasp, label, _ = self.plan_payload_pick(
            name=name, base_targets=[base_target], refusal=self.PAN_PICK_REFUSAL
        )
        return plans, chain, grasp, label

    def plan_payload_pick(
        self, *, name: str, base_targets: list[np.ndarray], refusal: str
    ) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray], np.ndarray, str, np.ndarray]:
        """`plan_pick_from_pan` from the first of `base_targets` (x, y, yaw) that plans,
        each with the grasp orientations of `pan_pick_rotations` at its heading; also
        returns the base target. A base that plans no path refuses the whole pick when it
        is the only one offered, as the pan pick always has; otherwise the next is tried."""
        from .planning.parameterized_skills import PickShrimpController, PlanningRefused
        from .planning.utils import KitchenPyBulletSim

        point, _ = self.payload_grasp_geometry(name=name)
        refusals: list[str] = []
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            planner = PickShrimpController(sim=sim)
            for base_target in base_targets:
                where = (
                    ""
                    if len(base_targets) == 1
                    else (
                        f"base ({base_target[0]:.2f}, {base_target[1]:.2f}, "
                        f"{np.rad2deg(base_target[2]):.0f} deg) "
                    )
                )
                based = False
                for label, rotation in self.pan_pick_rotations(
                    name=name, heading=float(base_target[2])
                ):
                    grasp = point - rotation.apply([0, 0, self.recovery_finger_offset])
                    chain = self.pan_pick_chain(grasp=grasp, rotation=rotation, base=base_target)
                    if isinstance(chain, str):
                        refusals.append(f"{where}{label}: {chain}")
                        continue
                    if len(chain) - 1 < round(self.PAN_PICK_HOVER_M / self.PAN_PICK_STEP_M):
                        try:
                            chain = chain + self.planned_pan_lift(
                                sim=sim,
                                chain=chain,
                                grasp=grasp,
                                rotation=rotation,
                                base_target=base_target,
                                name=name,
                            )
                        except PlanningRefused as exc:
                            straight = (len(chain) - 1) * self.PAN_PICK_STEP_M * 100
                            refusals.append(
                                f"{where}{label}: straight rise {straight:.0f} cm, then {exc}"
                            )
                            continue
                    if not based:
                        # The tuck, when one is needed, first tries this grasp's orientation.
                        try:
                            planner.plan_base(
                                x=self.scene._get_object_centric_state(),
                                arm=self.data.qpos[self.arm_q].copy(),
                                base=self.data.qpos[self.base_q].copy(),
                                base_target=base_target,
                                rotations=[rotation],
                                home=self.arm_neutral_q.copy(),
                            )
                        except PlanningRefused as exc:
                            if len(base_targets) == 1:
                                raise PlanningRefused(f"{refusal}: {exc}") from exc
                            refusals.append(f"{where}{exc}")
                            break
                        based = True
                    try:
                        planner.plan_hover(hover=chain[-1], base_target=base_target)
                    except PlanningRefused as exc:
                        refusals.append(f"{where}{label}: {exc}")
                        continue
                    return planner.plans, chain, grasp, label, base_target
        finally:
            sim.close()
        raise PlanningRefused(f"{refusal}: " + "; ".join(refusals))

    def floor_pick_base_targets(self, *, name: str) -> list[np.ndarray]:
        """Base poses facing the dumpling from each of `FLOOR_PICK_HEADINGS_DEG`, less those
        whose 2-D footprint overlaps an obstacle's: no path could end there."""
        from .planning.parameterized_skills import KitchenMotionPlanner

        point, _ = self.payload_grasp_geometry(name=name)
        state = self.scene._get_object_centric_state()
        targets = []
        for heading in np.deg2rad(self.FLOOR_PICK_HEADINGS_DEG):
            direction = np.array([np.cos(heading), np.sin(heading)])
            target = np.r_[point[:2] - self.FLOOR_PICK_REACH_M * direction, heading]
            if not KitchenMotionPlanner.footprint_blocked(state=state, target=target):
                targets.append(target)
        return targets

    def pick_from_floor(self, *, index: int) -> None:
        """Pick a dumpling from the floor as `pick_from_pan` picks one from the pan, the
        base facing it from the first heading that plans. The scripted pick drove the base
        straight to a pose in front of the dumpling, through the stove when it lay past it
        (pouring-threefix-20261109, EES seed 2 at p = 0.1, a dead end for the run)."""
        name = self.scene.cube_names[index]
        if self.held is not None or self.scene.cube_locations()[index] != "floor":
            raise ValueError("Shrimp pick requires an empty hand and the named source")
        self.shrimp_recovery_active = True
        self.shrimp_approach_path.clear()
        self.payload_approach_plan.clear()
        targets = self.floor_pick_base_targets(name=name)
        if not targets:
            raise ValueError(
                f"{self.FLOOR_PICK_REFUSAL}: every base pose facing the dumpling overlaps "
                "an obstacle"
            )
        start = self.body_pose(name=name)[0]
        plans, chain, grasp, label, base = self.plan_payload_pick(
            name=name, base_targets=targets, refusal=self.FLOOR_PICK_REFUSAL
        )
        self.events.append({
            "event": "planned pick from the floor",
            "payload": name,
            "grasp": label,
            "base": [float(v) for v in base],
        })
        self.follow_planned_pick(name=name, plans=plans, chain=chain, grasp=grasp, start=start)

    def planned_pan_lift(
        self,
        *,
        sim: Any,
        chain: list[np.ndarray],
        grasp: np.ndarray,
        rotation: Rotation,
        base_target: np.ndarray,
        name: str,
    ) -> list[np.ndarray]:
        """The rest of a lift whose straight rise stopped short of the hover: planned
        from the top of the straight rise to the hover, the dumpling carried as it will
        be once welded. The descent replays it downwards before the grasp, inside the
        same swept volume."""
        from spatialmath import SE2

        from .planning.parameterized_skills import (
            HeldObject,
            KitchenMotionPlanner,
            PlanningRefused,
        )

        sim.set_state(self.scene._get_object_centric_state())
        sim.set_robot_base(SE2(*(float(v) for v in base_target)))
        held = HeldObject.grasped(sim=sim, arm=chain[0], name=name, riders=(), keep_upright=False)
        # The hover can lie at the edge of the grasp orientation's reach; the welded
        # dumpling allows the hand to tilt away from the base, which extends it.
        away = np.array([grasp[0] - base_target[0], grasp[1] - base_target[1], 0.0])
        axis = np.cross(away / np.linalg.norm(away), [0.0, 0.0, 1.0])
        refusals: list[str] = []
        for tilt in self.PAN_LIFT_TILTS_DEG:
            try:
                path = KitchenMotionPlanner.plan_arm(
                    sim=sim,
                    start=chain[-1],
                    position=grasp + [0, 0, self.PAN_PICK_HOVER_M],
                    rotation=Rotation.from_rotvec(np.deg2rad(tilt) * axis) * rotation,
                    what="the lift above the pan",
                    held=held,
                )
            except PlanningRefused as exc:
                refusals.append(f"tilted {tilt:.0f} deg: {exc}")
                continue
            return [np.asarray(q, dtype=float) for q in path[1:]]
        raise PlanningRefused(", ".join(refusals))

    def pick_from_pan(self, *, index: int) -> None:
        """Pick a dumpling from the pan with planned free-space motion, as the jug's
        skills move: everything is planned before the robot moves (a refusal executes
        nothing). Then the planned tuck, base and arm motions to a hover above the
        dumpling; the checked vertical descent; the pinch, which welds the piece to the
        hand; and the lift, the descent replayed upwards."""
        name = self.scene.cube_names[index]
        if self.held is not None or self.scene.cube_locations()[index] != "tray":
            raise ValueError("Shrimp pick requires an empty hand and the named source")
        self.shrimp_recovery_active = True
        self.shrimp_approach_path.clear()
        self.payload_approach_plan.clear()
        start = self.body_pose(name=name)[0]
        point, _ = self.payload_grasp_geometry(name=name)
        base_target = np.array([self.recovery_base_x, point[1], 0.0])
        plans, chain, grasp, label = self.plan_pick_from_pan(name=name, base_target=base_target)
        self.events.append({"event": "planned pick from the pan", "payload": name, "grasp": label})
        self.follow_planned_pick(name=name, plans=plans, chain=chain, grasp=grasp, start=start)

    def follow_planned_pick(
        self,
        *,
        name: str,
        plans: dict[str, list[np.ndarray]],
        chain: list[np.ndarray],
        grasp: np.ndarray,
        start: np.ndarray,
    ) -> None:
        """Execute a planned pick: the planned tuck, base and arm motions to the hover,
        the checked descent, the pinch and the lift; `start` is where the dumpling lay."""
        from .planning.parameterized_skills import PickShrimpController

        grip = self.recovery_open_grip
        self.grip = grip
        before = self.scene.cube_locations()
        if PickShrimpController.TUCK in plans:
            self.follow_arm_path(
                path=plans[PickShrimpController.TUCK], phase=PickShrimpController.TUCK, grip=grip
            )
        self.follow_base_path(
            path=plans[PickShrimpController.BASE_MOTION], phase=PickShrimpController.BASE_MOTION
        )
        self.follow_arm_path(
            path=plans[PickShrimpController.MOVE_ARM_TO_HOVER],
            phase=PickShrimpController.MOVE_ARM_TO_HOVER,
            grip=grip,
        )
        # The planned path ends at the hover modulo whole turns of the continuous joints;
        # put the checked chain on the turn the arm is on.
        continuous = self.lower == -12
        arm = self.data.qpos[self.arm_q]
        turns = np.zeros(7)
        turns[continuous] = np.round((arm - chain[-1])[continuous] / (2 * np.pi))
        chain = [q + 2 * np.pi * turns for q in chain]
        self.base_target = self.data.qpos[self.base_q].copy()
        self.last_q = arm.copy()
        descent = chain[::-1]
        self.follow_contact_path(path=descent, phase="descend to shrimp body", grip=grip)
        # A failed pick retraces the executed descent (`clear_failed_pick`).
        self.shrimp_approach_path = [(self.base_target.copy(), q.copy()) for q in descent]
        if np.linalg.norm(self.hand_pose()[0] - grasp) > 0.03:
            raise ValueError("Motor tracking failed in descend to shrimp body")
        self.pinch_shrimp(name=name)
        self.follow_contact_path(path=chain, phase="lift shrimp", grip=1)
        if self.body_pose(name=name)[0][2] < start[2] + 0.10:
            self.held = None
            raise ValueError("Shrimp not retained through lift")
        extra = [
            other
            for other, was, now in zip(
                self.scene.cube_names, before, self.scene.cube_locations(), strict=True
            )
            if other != name and was != "air" and now == "air"
        ]
        if extra:
            self.return_lifted_dumplings(name=name, descent=descent)
            raise ValueError(f"{self.SECOND_DUMPLING_LIFTED}: {', '.join(extra)}")

    # A pick whose lift carried another dumpling up with the named one fails with this.
    SECOND_DUMPLING_LIFTED: ClassVar[str] = "Pick lifted a second dumpling"

    def return_lifted_dumplings(self, *, name: str, descent: list[np.ndarray]) -> None:
        """Lower the hand back down the checked descent and open it there.

        A second dumpling pinched between a pad and the held one hangs by friction, off
        every surface: it has no location, so no robot skill can name it. Released where
        they were taken, both lie on a surface again. The caller's `clear_failed_pick`
        then retraces the descent upwards, as after any failed pick."""
        self.follow_contact_path(
            path=descent, phase="lower shrimp after lifting a second dumpling", grip=1
        )
        self.held = None
        self.events.append({"event": "released a pick that lifted a second dumpling"})
        self.pause(phase="release shrimp after lifting a second dumpling", seconds=1, grip=0)
        for _ in range(round(10 * self.frequency)):
            if self.scene.cubes_settled():
                break
            self.motor_step(target_q=self.last_q, grip=0)

    # The scripted descent and lift each took 4 s; their checked replays take no less.
    CONTACT_PATH_SECONDS: ClassVar[float] = 4.0

    def follow_contact_path(self, *, path: list[np.ndarray], phase: str, grip: float) -> None:
        """Drive the arm through a checked vertical path that ends in contact (the pads on
        the dumpling, or on the pan beside it), as the scripted `move` did: at least
        `CONTACT_PATH_SECONDS`, then a short settle. Unlike `follow_arm_path` it does not
        demand joint convergence at the end, which contact can prevent; the caller judges
        where the hand ended, as the scripted pick did."""
        self.phase = phase
        self.grip = grip
        start = self.data.qpos[self.arm_q].copy()
        points = [start, *(np.asarray(q, dtype=float) for q in path)]
        spans = [float(np.max(np.abs(b - a))) for a, b in zip(points[:-1], points[1:], strict=True)]
        total = max(sum(spans), 1e-9)
        seconds = max(self.CONTACT_PATH_SECONDS, total / self.planned_joint_speed)
        for (a, b), span in zip(zip(points[:-1], points[1:], strict=True), spans, strict=True):
            count = max(1, int(np.ceil(seconds * self.frequency * span / total)))
            for index in range(1, count + 1):
                self.last_q = a + (b - a) * (index / count)
                self.motor_step(target_q=self.last_q, grip=grip)
        for _ in range(round(2 * self.frequency)):
            if np.max(np.abs(self.last_q - self.data.qpos[self.arm_q])) < 0.025:
                break
            self.motor_step(target_q=self.last_q, grip=grip)
        self.events.append({
            "event": "contact path followed",
            "phase": phase,
            "joint_error_rad": float(np.max(np.abs(self.last_q - self.data.qpos[self.arm_q]))),
        })

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
                neutral=self.arm_neutral_q.copy(),
            )
        finally:
            sim.close()
        if not np.array_equal(planner.base_target, self.base_home):
            self.events.append({
                "event": "placement base pose occupied; placing from a nearby pose",
                "nominal": self.base_home.tolist(),
                "base_target": planner.base_target.tolist(),
            })
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
        self.scene.release_pitcher_grasp()
        self.held = None
        self.pause(phase="release pitcher", seconds=2, grip=0)
        self.anchor_parked_pitcher()
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

    # Where the base stands to drop a dumpling into the jug, as the scripted carry did.
    SHRIMP_DROP_BASE_X: ClassVar[float] = 0.20

    def carry_held_shrimp_to_pitcher(self, *, name: str, pitcher: np.ndarray) -> Rotation:
        """Plan the held dumpling's carry to a hover above the jug (PyBullet) and follow
        it, as the jug's own carries are planned. The scripted carry kept the grasp
        orientation and tracked it with local IK, which pins the wrist at its limit; the
        weld makes any orientation safe, so the planner may turn the hand."""
        from .planning.parameterized_skills import DropShrimpController
        from .planning.utils import KitchenPyBulletSim

        hand, rotation = self.hand_pose()
        point, _ = self.body_pose(name=name)
        hover = pitcher + [0, 0, self.scene.get_object("pitcher").height + 0.07]
        sim = KitchenPyBulletSim.from_controller(controller=self)
        try:
            planner = DropShrimpController(sim=sim)
            planner.reset(
                x=self.scene._get_object_centric_state(),
                arm=self.data.qpos[self.arm_q].copy(),
                base=self.data.qpos[self.base_q].copy(),
                base_target=np.array([self.SHRIMP_DROP_BASE_X, pitcher[1], 0.0]),
                name=name,
                hand_rotation=rotation,
                payload_in_hand=rotation.inv().apply(point - hand),
                hover=hover,
            )
        finally:
            sim.close()
        self.follow_arm_path(path=planner.plans[planner.TUCK], phase=planner.TUCK, grip=1)
        self.follow_base_path(path=planner.plans[planner.BASE_MOTION], phase=planner.BASE_MOTION)
        self.follow_arm_path(
            path=planner.plans[planner.MOVE_ARM_TO_HOVER], phase=planner.MOVE_ARM_TO_HOVER, grip=1
        )
        self.last_q = self.data.qpos[self.arm_q].copy()
        self.base_target = self.data.qpos[self.base_q].copy()
        return self.hand_pose()[1]

    def active_obstacle_pairs(self) -> list[tuple[int, int]]:
        pairs = super().active_obstacle_pairs()
        # After a drop the unwind starts beside the jug: a whole turn of the wrist can
        # sweep the fingers through it.
        if self.held is None and self.phase in {
            "withdraw empty gripper above pitcher",
            "unwind continuous arm joints",
        }:
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

    # Phases in which a closed gripper takes a dumpling: the weld engages here once
    # the piece is between the pads, touching one pad or, after the full close, none.
    PAYLOAD_GRASP_PHASES: ClassVar[frozenset[str]] = frozenset({"pinch shrimp", "retain shrimp"})
    # A piece touching neither pad still counts as taken after the full close if its
    # centre of mass is this close to the pinch point (the pads span about 85 mm).
    PAYLOAD_BETWEEN_PADS_M: ClassVar[float] = 0.05

    def payload_between_pads(self, *, name: str) -> bool:
        centre = self.data.xipos[self.model.body(name).id]
        return bool(
            np.linalg.norm(centre - self.data.site_xpos[self.site]) < self.PAYLOAD_BETWEEN_PADS_M
        )

    def finger_contacts(self, *, name: str, handle_only: bool = False) -> set[str]:
        if name == "pitcher" and self.scene.pitcher_attached():
            return {"left", "right"}
        if name in self.scene.cube_names:
            if self.scene.attached_payload() == name:
                return {"left", "right"}
            contacts = super().finger_contacts(name=name, handle_only=handle_only)
            if self.phase in self.PAYLOAD_GRASP_PHASES and (
                contacts or (self.phase == "retain shrimp" and self.payload_between_pads(name=name))
            ):
                self.scene.attach_payload(name=name)
                self.events.append({"event": "dumpling attached", "payload": name})
                return {"left", "right"}
            return contacts
        return super().finger_contacts(name=name, handle_only=handle_only)

    def motor_step(self, *, target_q: np.ndarray, grip: float) -> None:
        attached = self.scene.attached_payload()
        if (
            attached is not None
            and self.held != attached
            and self.phase not in self.PAYLOAD_GRASP_PHASES
        ):
            # Every release path clears `held` before it opens the fingers.
            self.scene.release_payloads()
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
                neutral=self.arm_neutral_q.copy(),
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
