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


class KitchenScene(PotstickerScene):
    # Above the dumplings' priority 2 and the jug's 1, so MuJoCo takes the skillet's
    # coefficient for every skillet contact and the dumplings' own elsewhere.
    SKILLET_PRIORITY: ClassVar[int] = 3

    def __init__(self, *args: Any, skillet_mu: float = 0.4, **kwargs: Any) -> None:
        if not np.isfinite(skillet_mu) or skillet_mu < 0:
            raise ValueError("Invalid skillet friction")
        self.skillet_mu = skillet_mu
        super().__init__(*args, **kwargs)

    def _create_scene_xml(self) -> str:
        root = ET.fromstring(super()._create_scene_xml())
        tray = root.find(".//body[@name='tray']")
        assert tray is not None
        for geom in tray.iter("geom"):
            geom.set("priority", str(self.SKILLET_PRIORITY))
            geom.set("friction", f"{self.skillet_mu} 0.005 0.0001")
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
    # A loaded motion within this of a continuous joint's IK bound fails loudly.
    held_bound_margin: ClassVar[float] = np.deg2rad(20.0)

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
        """Whole turns each continuous arm joint has wound away from its neutral angle."""
        continuous = self.lower == -12
        q = self.data.qpos[self.arm_q]
        turns = np.zeros(len(q), dtype=int)
        turns[continuous] = np.round(
            (q[continuous] - self.arm_neutral_q[continuous]) / (2 * np.pi)
        ).astype(int)
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
        if self.held is None:
            self.unwind_continuous_joints()
        super().pick_pitcher()

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
        try:
            super().place_pitcher()
        except ValueError as exc:
            if (
                not str(exc).startswith("IK infeasible in seat pitcher on table")
                or not self.can_release_pitcher_over_home()
            ):
                raise
            # Release only from a measured low, upright pose above support;
            # normal physics and the inherited landing checks determine success.
            self.release_pitcher_over_home()

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
        if np.linalg.norm(self.data.qpos[self.base_q][:2] - self.base_home[:2]) < 0.3:
            hand, rotation = self.hand_pose()
            # Fold the elbow up in the clear aisle before approaching the hob.
            # A base-offset-only change leaves the carried elbow below its edge.
            self.move(
                position=hand + [-0.15, 0, 0.25],
                rotation=rotation,
                phase="fold and raise pitcher for transport",
                seconds=6,
                grip=1,
            )
        return super().prepare_to_pour()

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
