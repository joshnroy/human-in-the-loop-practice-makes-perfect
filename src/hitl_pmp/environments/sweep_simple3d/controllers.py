"""New floor skills using existing collision-aware planning and physical execution."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from pybullet_helpers.geometry import Pose
from pydantic import BaseModel, Field, PrivateAttr

from hitl_pmp.environments.sweep_simple3d.physical.motion import ExecutionError, Motion
from hitl_pmp.environments.sweep_simple3d.physical.planning_scene import ArmMath, PlanningScene
from hitl_pmp.environments.sweep_simple3d.physical.primitives import Primitives


class ContactTravelBudget(BaseModel):
    """Bound extra clearance approach separately from observed loaded travel."""

    length: float = Field(gt=0, allow_inf_nan=False)
    stroke: float = Field(gt=0, allow_inf_nan=False)
    step: float = Field(gt=0, allow_inf_nan=False)
    behind: float = Field(ge=0, allow_inf_nan=False)
    onset: float | None = None

    def targets(self) -> np.ndarray:
        # Preserve the original zero-behind command sequence exactly.
        if self.behind == 0:
            return np.arange(self.step, min(self.length + 0.10, self.stroke + self.step), self.step)
        # Remaining cube travel excludes the clearance offset behind its anchor.
        bound = self.behind + min(self.length + 0.10, self.stroke)
        return np.minimum(np.arange(self.step, bound + self.step, self.step), bound)

    def observe(self, *, projection: float, loaded: bool) -> bool:
        if not np.isfinite(projection):
            raise ValueError("Nonfinite observed contact travel")
        if loaded and self.onset is None:
            self.onset = projection
        return self.onset is not None and projection >= self.onset + self.stroke

    def target(self, *, proposed: float) -> float:
        return proposed if self.onset is None else min(proposed, self.onset + self.stroke)


class FloorApproachPreference(BaseModel):
    """Prefer native endpoint reserve without discarding a valid first fallback."""

    candidate: Any = None
    margin: float = -np.inf
    preferred: bool = False

    @staticmethod
    def native_margin(*, arm: Any, limits: np.ndarray) -> float:
        q = np.asarray(arm[:7], dtype=float)
        distances = np.column_stack((q - limits[:, 0], limits[:, 1] - q))
        finite = np.isfinite(limits)
        return float(distances[finite].min()) if finite.any() else float("inf")

    def consider(self, *, candidate: Any, margin: float) -> bool:
        if np.isnan(margin) or margin < 0:
            raise ValueError("Only native-valid checked candidates can be ranked")
        if self.candidate is None:
            self.candidate, self.margin = candidate, margin
        if margin >= 0.01:
            self.candidate, self.margin, self.preferred = candidate, margin, True
        return self.preferred


class ContactTravelLimit(Exception):
    """Internal motion stop: unload normally after reaching the loaded travel cap."""


class ContactTiltLimit(ContactTravelLimit):
    """Internal stop before loaded tilt exhausts the checked retreat allowance."""


class ContactJointReserveLimit(ContactTravelLimit):
    """Stop loaded contact while a valid arm still has room for checked retreat."""


class FloorPrimitives(Primitives):
    """Floor pickup uses the tested generic handle grasp with a learned base stance."""

    scene: "FloorPlanningScene"
    narrow_contact: bool = False
    stand_ahead: bool = True
    native_contact_guard: bool = True
    contact_stroke_length: float = 0.10
    contact_step: float = 0.003
    floor_clearance: float = Field(default=0.005, ge=0.001, le=0.01)
    distance: float = Field(default=0.7, ge=0.55, le=0.85)
    heading_offset: float = Field(default=0.0, ge=-np.pi / 12, le=np.pi / 12)
    _ground_clearance_hold: np.ndarray | None = PrivateAttr(default=None)
    _exhausted_stow_key: tuple[Any, ...] | None = PrivateAttr(default=None)

    def guard_loaded_tool_tilt(self, *, phase: str) -> None:
        """Unload loaded contact with 0.10rad reserve; keep the planning limit intact."""
        from scipy.spatial.transform import Rotation

        if not self.wiper_loaded_by_cube():
            return
        rotation = Rotation.from_quat(self.session.quaternion(name="wiper_0"))
        tilt = float(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))
        limit = self.scene.max_tool_tilt
        unload_tilt = max(0.0, limit - 0.10)
        if tilt >= unload_tilt:
            self.session._write(
                record={
                    "kind": "contact_stroke_ended",
                    "t": self.session.ticks,
                    "reason": "loaded tool tilt reserve exhausted; checked unload required",
                    "phase": phase,
                    "tilt": tilt,
                    "unload_tilt": unload_tilt,
                    "planning_tilt_limit": limit,
                    "already_over_planning_limit": tilt > limit,
                }
            )
            # An already-invalid starting pose still faces the unchanged retreat
            # checker and remains a failure if that checked route is unavailable.
            raise ContactTiltLimit

    def guard_loaded_joint_reserve(self, *, phase: str) -> None:
        """Stop loaded motion at the existing 0.01rad native approach reserve."""
        if not self.wiper_loaded_by_cube():
            return
        arm = self.session.arm().copy()
        margin = FloorApproachPreference.native_margin(arm=arm, limits=self.scene._arm_limits)
        if margin <= 0.01:
            self.session._write(
                record={
                    "kind": "contact_stroke_ended",
                    "t": self.session.ticks,
                    "reason": "loaded native joint reserve exhausted; checked unload required",
                    "phase": phase,
                    "native_joint_margin": margin,
                    "required_reserve": 0.01,
                    "observed_arm": arm.tolist(),
                    "already_outside_native_limits": margin < 0.0,
                }
            )
            raise ContactJointReserveLimit

    def wiper_pickup_descent_clear(self, *, start: np.ndarray, path: list[np.ndarray]) -> bool:
        """Allow pad contact only; a fallen blade must not collide with finger links."""
        from hitl_pmp.environments.sweep_simple3d.native_palm import NativePalmClearance

        if self.scene._native_palm is None:
            self.scene._native_palm = NativePalmClearance(
                model=self.session.mj_model, live_data=self.session.mj_data
            )
        tool = Pose(
            tuple(self.session.position(name="wiper_0")), self.session.quaternion(name="wiper_0")
        )
        previous = np.asarray(start)
        for waypoint in path:
            waypoint = previous + ArmMath.wrap(delta=np.asarray(waypoint) - previous)
            steps = max(1, int(np.ceil(np.max(np.abs(waypoint - previous)) / 0.05)))
            for fraction in np.linspace(0.0, 1.0, steps + 1):
                arm = previous + fraction * (waypoint - previous)
                contacts = self.scene._native_palm.nonpad_tool_contacts(
                    joints=self.scene.planning_fingers(arm=arm, state=0.0),
                    tool_pose=tool,
                    base=self.scene._planning_base,
                )
                if contacts:
                    self.session._write(
                        record={
                            "kind": "pickup_nonpad_path_rejection",
                            "t": self.session.ticks,
                            "fraction": float(fraction),
                            "arm": arm.tolist(),
                            "base": None
                            if self.scene._planning_base is None
                            else list(self.scene._planning_base),
                            "contacts": contacts,
                        }
                    )
                    return False
            previous = waypoint
        return True

    def wiper_grasp_offsets(self) -> tuple[float, ...]:
        # A low cross-handle grasp shortens the contact-force lever arm.
        return (-0.12,)

    def wiper_approach_angles(self) -> tuple[float, ...]:
        return np.pi / 2, 1.2, 1.8

    def wiper_grasp_orientations(self, *, axis: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Keep nominal candidates first; approach a tipped handle from above last."""
        orientations = super().wiper_grasp_orientations(axis=axis)
        if abs(float(axis[2])) >= np.cos(np.pi / 4):
            return orientations
        handle, handle_axis = self.wiper_handle_geometry()
        axes = np.asarray(self.session.mj_data.geom_xmat[handle]).reshape(3, 3)
        face = max(
            (axes[:, i] for i in range(3) if i != handle_axis),
            key=lambda a: float(np.linalg.norm(a[:2])),
        )
        yaw = float(np.arctan2(face[1], face[0]))
        for azimuth in (yaw, yaw + np.pi):
            closing = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
            toward = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
            for angle in (0.0, 0.2, 0.4):
                approach = np.sin(angle) * toward + np.array([0.0, 0.0, -np.cos(angle)])
                orientations.append((closing, approach))
        return orientations

    diagnostic_grasp_standoff: float = Field(default=0.020, ge=0.0, le=0.035)
    retain_pickup_carry_pose: bool = True

    def wiper_pickup_carry_goal(self) -> np.ndarray:
        """Optionally retain a verified raised grasp only at pickup completion."""
        from pybullet_helpers.geometry import multiply_poses

        if self.retain_pickup_carry_pose:
            self.require_handle(phase="pickup carry verification")
            if self.blade_minimum_height() > self.floor_clearance:
                self.scene.sync()
                arm = self.session.arm().copy()
                held_tf = multiply_poses(
                    self.scene.ee_now().invert(),
                    Pose(
                        tuple(self.session.position(name="wiper_0")),
                        self.session.quaternion(name="wiper_0"),
                    ),
                )
                if self.scene.held_path_clear(
                    path=[arm],
                    start=arm,
                    bodies=self.scene.bodies(),
                    held=self.scene.wiper_body,
                    held_tf=held_tf,
                    allowed_tilt=self.scene.max_tool_tilt,
                ):
                    return arm
        return self.wiper_stow_goal()

    def wiper_grasp_standoff(self) -> float:
        """Use the floor handle insertion; allow explicit diagnostic trials."""
        if not 0.0 <= self.diagnostic_grasp_standoff <= 0.035:
            raise ValueError("Diagnostic grasp standoff must lie in [0, 0.035] meters")
        return self.diagnostic_grasp_standoff

    def wiper_grasp_yaw(self, *, axis: np.ndarray) -> float:
        """Align the grasp with the native handle face, independent of base stance."""
        del axis
        handle, _ = self.wiper_handle_geometry()
        axes = self.session.mj_data.geom_xmat[handle].reshape(3, 3)
        return float(np.arctan2(axes[1, 0], axes[0, 0]))

    def wiper_pick_targets_after_navigation(
        self, *, hover: Pose, target: Pose, along: float, approach: np.ndarray
    ) -> tuple[Pose, Pose]:
        """Refresh a floor tool displaced by empty-hand navigation before descent."""
        del hover
        handle, handle_axis = self.wiper_handle_geometry()
        data = self.session.mj_data
        center = np.asarray(data.geom_xpos[handle])
        axis = np.asarray(data.geom_xmat[handle]).reshape(3, 3)[:, handle_axis]
        point = self.wiper_grasp_point(center=center, axis=axis, along=along)
        refreshed = Pose(tuple(point - approach * self.wiper_grasp_standoff()), target.orientation)
        hover = Pose(tuple(np.asarray(refreshed.position) - approach * 0.08), refreshed.orientation)
        self.session._write(
            record={
                "kind": "pickup_target_refreshed",
                "t": self.session.ticks,
                "previous_target": list(target.position),
                "target": list(refreshed.position),
                "observed_handle_center": center.tolist(),
                "observed_handle_axis": axis.tolist(),
            }
        )
        # The shared caller still resolves IK and collision-checks approach and
        # descent at the actual base. This hook neither moves nor grasps anything.
        return hover, refreshed

    def wiper_stow_goal(self) -> np.ndarray:
        from hashlib import sha256

        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        # IKFast sorts relative to the planning robot's current joints. Restore
        # the observed state before consulting an exact-state failure cache.
        self.scene.sync()
        base = self.session.base()
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        from scipy.spatial.transform import Rotation

        from hitl_pmp.environments.sweep_simple3d.physical.types import KitchenScene

        model, data = self.session.mj_model, self.session.mj_data
        geometry = sha256()
        # Include mutable native geometry, not just model identity: diagnostic
        # restores and human actions must never reuse a different scene's failure.
        for name in (
            "geom_type",
            "geom_size",
            "geom_pos",
            "geom_quat",
            "geom_bodyid",
            "geom_contype",
            "geom_conaffinity",
            "geom_margin",
            "geom_gap",
            "body_pos",
            "body_quat",
            "jnt_range",
            "jnt_pos",
            "jnt_axis",
            "mesh_vert",
            "mesh_face",
            "hfield_data",
            "hfield_size",
            "geom_dataid",
            "exclude_signature",
            "pair_geom1",
            "pair_geom2",
        ):
            geometry.update(np.asarray(getattr(model, name)).tobytes())
        cache_key = (
            id(model),
            id(self.scene),
            geometry.digest(),
            model.opt.enableflags,
            model.opt.disableflags,
            data.qpos.tobytes(),
            data.qvel.tobytes(),
            data.mocap_pos.tobytes(),
            data.mocap_quat.tobytes(),
            data.geom_xpos.tobytes(),
            data.geom_xmat.tobytes(),
            tuple(held_tf.position),
            tuple(held_tf.orientation),
            self.scene.max_tool_tilt,
            tuple(sorted(self.scene.bodies())),
        )
        if self._exhausted_stow_key == cache_key:
            raise ExecutionError("No collision-free compact tool transport pose")
        self._exhausted_stow_key = None

        def carry_endpoint_clear(*, arm: np.ndarray) -> bool:
            tool = multiply_poses(self.scene.fk(arm=arm), held_tf)
            tilt = float(
                np.arccos(
                    np.clip(Rotation.from_quat(tool.orientation).as_matrix()[2, 2], -1.0, 1.0)
                )
            )
            return tilt <= self.scene.max_tool_tilt

        home = np.asarray(KitchenScene.HOME)
        if (
            carry_endpoint_clear(arm=home)
            and self.scene.plan_arm(
                goal=home,
                bodies=self.scene.bodies(),
                held=self.scene.wiper_body,
                held_tf=held_tf,
                allow_joint_fallback=False,
            )
            is not None
        ):
            return home
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
                            if not carry_endpoint_clear(arm=np.asarray(solution[:7])):
                                continue
                            path = self.scene.plan_arm(
                                goal=solution[:7],
                                bodies=self.scene.bodies(),
                                held=self.scene.wiper_body,
                                held_tf=held_tf,
                                allow_joint_fallback=True,
                            )
                            if path is not None:
                                return np.asarray(solution[:7])
        # Only complete exhaustion is reusable; exceptions and partial searches
        # leave no cached result. The caller still records each failed action.
        self._exhausted_stow_key = cache_key
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

    def handle_nonpad_gripper_contacts(self) -> list[str]:
        """Observe handle wedging against robot parts outside the two intended pads."""
        import mujoco

        model, data = self.session.mj_model, self.session.mj_data
        handle, _ = self.wiper_handle_geometry()
        bodies = set()
        for contact in data.contact[: data.ncon]:
            if contact.geom1 == handle:
                other = contact.geom2
            elif contact.geom2 == handle:
                other = contact.geom1
            else:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[other])
            if (
                name
                and name.startswith("robot_")
                and name not in {"robot_left_pad", "robot_right_pad"}
            ):
                bodies.add(name)
        return sorted(bodies)

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

    @staticmethod
    def broad_blade_center(*, projections: list[float], target: float) -> float:
        """Center on the selected cube so neighboring cubes cannot move its anchor."""
        del projections
        return target

    @staticmethod
    def contact_behind_offset(
        *, relative_positions: list[tuple[float, float]], narrow: bool
    ) -> float:
        """Clear nearby cubes without anchoring every stroke to distant cubes."""
        stand_off = 0.20 if narrow else 0.05
        blade_half_depth = 0.15 if narrow else 0.01
        # Native cubes have 1 cm half-width; keep another 5 mm clearance.
        rear_reach = stand_off + blade_half_depth + 0.015
        half_width = 0.025 if narrow else 0.16
        return max(
            [0.0]
            + [
                -along
                for along, across in relative_positions
                if -rear_reach <= along < 0.0 and abs(across) <= half_width
            ]
        )

    def fixed_reverse_stance(
        self,
        *,
        region: str,
        distance: float,
        wiper_start: np.ndarray,
        bearing: float,
        original: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Keep learned stances unchanged; retry the fixed .70 reset at checked .65."""
        if region != "blocks_init_region" or distance != 0.70:
            return original
        if self.scene.plan_base(target=original) is not None:
            return original
        alternative = (
            float(wiper_start[0] + 0.65 * np.cos(bearing)),
            float(wiper_start[1] + 0.65 * np.sin(bearing)),
            original[2],
        )
        if self.scene.plan_base(target=alternative) is None:
            raise ExecutionError("No native base route to either fixed reverse stance")
        self.session._write(
            record={
                "kind": "fixed_reverse_stance_fallback",
                "t": self.session.ticks,
                "original": original,
                "selected": alternative,
                "reason": "Original .70 fixed reverse base route rejected; checked .65 route",
            }
        )
        return alternative

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
        horizontal_leg = abs(direction[0]) > abs(direction[1])
        westward_goal_leg = region == "sweep_region" and horizontal_leg
        # The same blade-end geometry serves westward goal and eastward reset
        # strokes. A broad eastward blade requires an unreachable 90-degree
        # floor wrist orientation with the observed post-goal grasp.
        narrow_contact = self.narrow_contact or horizontal_leg
        transverse = np.array([-direction[1], direction[0]])
        blade_anchor = initial[:2].copy()
        if not narrow_contact:
            projections = [
                float(self.session.position(name=f"cube_{i}")[:2] @ transverse) for i in range(5)
            ]
            midpoint = self.broad_blade_center(
                projections=projections, target=float(blade_anchor @ transverse)
            )
            blade_anchor += (midpoint - float(blade_anchor @ transverse)) * transverse
        relative_positions = []
        for other in (f"cube_{i}" for i in range(5)):
            relative = self.session.position(name=other)[:2] - blade_anchor
            relative_positions.append((float(relative @ direction), float(relative @ transverse)))
        behind = self.contact_behind_offset(
            relative_positions=relative_positions, narrow=narrow_contact
        )
        # Leave room for physical tracking error during descent. A 25 mm
        # center gap left only 5 mm beyond blade/cube half-widths, and the
        # loaded tool could land on the cubes before reaching floor height.
        wiper_start = blade_anchor - (behind + (0.20 if narrow_contact else 0.05)) * direction
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
            nominal_bearing = (
                angle if self.stand_ahead and region == "sweep_region" else angle + np.pi
            )
        stance_bearing = nominal_bearing + heading_offset
        stance_angle = float((stance_bearing + 2 * np.pi) % (2 * np.pi) - np.pi)
        stance = (
            float(wiper_start[0] + distance * np.cos(stance_bearing)),
            float(wiper_start[1] + distance * np.sin(stance_bearing)),
            stance_angle,
        )
        stance = self.fixed_reverse_stance(
            region=region,
            distance=distance,
            wiper_start=wiper_start,
            bearing=stance_bearing,
            original=stance,
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

        preference = FloorApproachPreference()
        attempts: list[dict[str, Any]] = []
        tool_yaws = (
            (angle + np.pi, angle) if narrow_contact else (angle - np.pi / 2, angle + np.pi / 2)
        )
        for tool_yaw, preserve_tilt, nearby_tilt, blade_axis in self.contact_orientation_candidates(
            tool_yaws=tool_yaws, region=region, narrow_contact=narrow_contact
        ):
            floor_pose = self.floor_tool_pose(
                xy=wiper_start,
                yaw=tool_yaw,
                preserve_tilt=preserve_tilt,
                tilt=nearby_tilt,
                blade_axis=blade_axis,
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
                    "nearby_tilt": nearby_tilt,
                    "native_blade_axis": blade_axis,
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
                    margin = preference.native_margin(
                        arm=candidate_lower[-1], limits=self.scene._arm_limits
                    )
                    attempts[-1]["best_endpoint_native_margin"] = max(
                        margin, attempts[-1].get("best_endpoint_native_margin", -np.inf)
                    )
                    if preference.consider(
                        candidate=(
                            candidate,
                            candidate_lower,
                            tool_yaw,
                            preserve_tilt,
                            nearby_tilt,
                            floor_pose,
                            blade_axis,
                        ),
                        margin=margin,
                    ):
                        break
                if preference.preferred:
                    break
            if preference.preferred:
                break
        if preference.candidate is None:
            raise ExecutionError(f"No collision-free floor sweep approach: {attempts}")
        approach, candidate_lower, tool_yaw, preserve_tilt, nearby_tilt, floor_pose, blade_axis = (
            preference.candidate
        )
        self.session._write(
            record={
                "kind": "floor_approach_selected",
                "t": self.session.ticks,
                "endpoint_native_joint_margin": preference.margin,
                "preferred_joint_margin": preference.preferred,
                "tool_yaw": tool_yaw,
                "preserve_tilt": preserve_tilt,
                "nearby_tilt": nearby_tilt,
                "native_blade_axis": blade_axis,
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
            self.session._write(
                record={
                    "kind": "floor_descent_failed",
                    "t": self.session.ticks,
                    "goal_arm": np.asarray(lower[-1]).tolist(),
                    "actual_arm": self.session.arm().tolist(),
                    "goal_tool_position": floor_pose.position,
                    "actual_tool_position": self.session.position(name="wiper_0").tolist(),
                    "native_qpos": self.session.mj_data.qpos.tolist(),
                }
            )
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
                max_ticks=180,
                final_tol=0.0005,
                tick_guard=lambda: self.require_handle(phase="initial blade-height correction"),
            ):
                if self.level_blade(bodies=bodies):
                    continue
                break
            corrected_height = self.blade_bottom_height(cube=cube, narrow=narrow_contact)
            if corrected_height > center_height and height - corrected_height < 0.0001:
                # Joint convergence alone does not establish physical overlap:
                # a submillimeter descent can finish inside the arm tolerance.
                # Use the existing checked leveling motion when it made no
                # measurable contact-height progress; never relax the floor.
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
        contact_grasp = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        contact_ended = False
        tilt_unload = False
        travel = ContactTravelBudget(
            length=length, stroke=self.contact_stroke_length, step=self.contact_step, behind=behind
        )
        self.session._write(
            record={
                "kind": "contact_travel_budget",
                "t": self.session.ticks,
                "free_lead_in": behind,
                "loaded_limit": self.contact_stroke_length,
                "command_projection_limit": float(travel.targets()[-1]),
            }
        )

        def observe_contact_travel() -> None:
            self.guard_loaded_joint_reserve(phase="contact drive")
            self.guard_loaded_tool_tilt(phase="contact drive")
            projection = float((np.asarray(self.session.base()[:2]) - base_origin) @ direction)
            before = travel.onset
            reached = travel.observe(projection=projection, loaded=self.wiper_loaded_by_cube())
            if before is None and travel.onset is not None:
                self.session._write(
                    record={
                        "kind": "contact_load_onset",
                        "t": self.session.ticks,
                        "base_projection": travel.onset,
                        "free_lead_in": behind,
                        "loaded_projection_limit": travel.onset + travel.stroke,
                    }
                )
            if reached:
                self.session._write(
                    record={
                        "kind": "contact_loaded_limit",
                        "t": self.session.ticks,
                        "base_projection": projection,
                        "contact_onset": travel.onset,
                        "loaded_limit": travel.stroke,
                    }
                )
                raise ContactTravelLimit

        for proposed in travel.targets():
            try:
                observe_contact_travel()
            except ContactTravelLimit as stop:
                tilt_unload = isinstance(stop, ContactTiltLimit)
                break
            progress = travel.target(proposed=float(proposed))
            ground_corrected = False
            current = self.session.position(name=cube)
            if (
                core._ground_fixture.check_in_region(current, region, core._robot_env)
                or np.linalg.norm(current[:2] - target) < 0.025
            ):
                break
            next_xy = base_origin + progress * direction
            target_base = (float(next_xy[0]), float(next_xy[1]), stance_angle)
            contact_arm = self.valid_contact_hold(cached=contact_arm)
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
            try:
                driven = self.motion.drive(
                    path=base_path,
                    grip=1.0,
                    max_ticks=30,
                    arm=contact_arm,
                    tol=0.0005,
                    tick_guard=observe_contact_travel,
                )
            except ContactTravelLimit as stop:
                tilt_unload = isinstance(stop, ContactTiltLimit)
                self.require_handle(phase="loaded contact travel limit")
                break
            if not driven:
                raise ExecutionError("Contact sweep base did not converge")
            self.require_handle(phase="contact base step")
            from scipy.spatial.transform import Rotation

            observed_grasp = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )
            grasp_translation = float(
                np.linalg.norm(np.asarray(observed_grasp.position) - contact_grasp.position)
            )
            grasp_rotation = float(
                (
                    Rotation.from_quat(observed_grasp.orientation)
                    * Rotation.from_quat(contact_grasp.orientation).inv()
                ).magnitude()
            )
            unexpected_contacts = self.handle_nonpad_gripper_contacts()
            unload = (
                bool(unexpected_contacts)
                if self.native_contact_guard
                else grasp_translation > 0.01 or grasp_rotation > 0.06
            )
            if unload:
                self.session._write(
                    record={
                        "kind": "contact_stroke_ended",
                        "t": self.session.ticks,
                        "reason": (
                            "native handle contact outside finger pads; unload"
                            if self.native_contact_guard
                            else "loaded grasp drift; unload before further correction"
                        ),
                        "unexpected_handle_contacts": unexpected_contacts,
                        "grasp_translation_m": grasp_translation,
                        "grasp_rotation_rad": grasp_rotation,
                    }
                )
                contact_ended = True
                break
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
                # Forward strokes retain their low target; reset strokes keep the
                # same native cube overlap used to accept their floor placement.
                contact_ceiling = self.contact_control_ceiling(cube=cube, region=region)
                if blade_bottom <= contact_ceiling:
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
                                blade_bottom - contact_ceiling + 0.002,
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

                def correction_guard() -> None:
                    self.require_handle(phase="contact correction")
                    self.guard_loaded_joint_reserve(phase="contact correction")
                    self.guard_loaded_tool_tilt(phase="contact correction")

                try:
                    correction_guard()
                    corrected = path is not None and self.motion.follow(
                        path=path,
                        grip=1.0,
                        tol=0.03,
                        final_tol=0.0005,
                        max_ticks=180,
                        tick_guard=correction_guard,
                        stop_condition=lambda: (
                            self.blade_bottom_height(cube=cube, narrow=narrow_contact)
                            <= self.contact_control_ceiling(cube=cube, region=region)
                        ),
                    )
                except (ContactTiltLimit, ContactJointReserveLimit) as stop:
                    tilt_unload = isinstance(stop, ContactTiltLimit)
                    contact_ended = True
                    break
                if not corrected:
                    self.require_handle(phase="contact correction")
                    self.session._write(
                        record={
                            "kind": "contact_stroke_ended",
                            "t": self.session.ticks,
                            "path_found": path is not None,
                            "height": wiper_now.position[2],
                            "tilt": upright_error,
                            "blade_bottom": blade_bottom,
                            "actual_blade_bottom": self.blade_bottom_height(
                                cube=cube, narrow=narrow_contact
                            ),
                            "goal_arm": np.asarray(path[-1]).tolist() if path else None,
                            "actual_arm": self.session.arm().tolist(),
                        }
                    )
                    contact_ended = True
                    break
                attained_height = self.blade_bottom_height(cube=cube, narrow=narrow_contact)
                if attained_height > contact_ceiling:
                    self.session._write(
                        record={
                            "kind": "contact_stroke_ended",
                            "t": self.session.ticks,
                            "reason": "joint convergence did not attain physical contact height",
                            "blade_bottom": attained_height,
                            "control_contact_ceiling": contact_ceiling,
                        }
                    )
                    contact_ended = True
                    break
                self.session._write(
                    record={
                        "kind": "contact_overlap_corrected",
                        "t": self.session.ticks,
                        "blade_bottom": self.blade_bottom_height(cube=cube, narrow=narrow_contact),
                        "cube_contact_ceiling": self.cube_contact_ceiling(cube=cube),
                        "control_contact_ceiling": contact_ceiling,
                    }
                )
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
        if tilt_unload:
            self.restore_unloaded_tilt_margin(bodies=bodies)
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
                "joint_target_reached": path is not None
                and bool(path)
                and float(
                    np.max(np.abs(ArmMath.wrap(delta=np.asarray(path[-1]) - self.session.arm())))
                )
                < 0.005,
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

    def restore_unloaded_tilt_margin(self, *, bodies: set[int]) -> None:
        """After a tilt stop, require measured recovery before lifting/reapproaching."""
        from scipy.spatial.transform import Rotation

        def observed_tilt() -> float:
            rotation = Rotation.from_quat(self.session.quaternion(name="wiper_0"))
            return float(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))

        target = min(0.95, self.scene.max_tool_tilt - 0.15)
        for attempt in range(4):
            before = observed_tilt()
            if self.wiper_loaded_by_cube():
                raise ExecutionError("Tilt recovery requires an unloaded blade")
            if before < target:
                return
            improved = self.level_blade(bodies=bodies)
            after = observed_tilt()
            self.session._write(
                record={
                    "kind": "unloaded_tilt_recovery",
                    "t": self.session.ticks,
                    "attempt": attempt + 1,
                    "before_tilt": before,
                    "after_tilt": after,
                    "target_tilt": target,
                    "measured_progress": before - after,
                    "leveling_success": improved,
                }
            )
            if not improved or after >= before - 0.002:
                raise ExecutionError("Checked unloaded leveling made no measured tilt progress")
        if self.wiper_loaded_by_cube():
            raise ExecutionError("Tilt recovery reloaded the blade")
        if observed_tilt() >= target:
            raise ExecutionError(
                "Checked unloaded leveling exhausted four-turn tilt recovery budget"
            )

    def valid_contact_hold(self, *, cached: np.ndarray) -> np.ndarray:
        """Replace only an invalid stale hold with an exactly valid observed arm."""
        if self.scene.within_arm_limits(arm=cached):
            return cached
        observed = self.session.arm().copy()
        valid = self.scene.within_arm_limits(arm=observed)
        self.session._write(
            record={
                "kind": "contact_hold_limit_refresh",
                "t": self.session.ticks,
                "cached_arm": cached.tolist(),
                "observed_arm": observed.tolist(),
                "observed_within_native_limits": bool(valid),
                "reason": "cached_hold_outside_native_joint_limits",
            }
        )
        if not valid:
            raise ExecutionError("Cached and observed contact arms violate native joint limits")
        return observed

    @staticmethod
    def leveling_tick_budget(*, path_waypoints: int) -> int:
        """Allow checked branch changes to settle while bounding leveling time."""
        return max(180, min(1800, 30 * path_waypoints))

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
        # Preserve support-pivot attempts first. An unloaded low grasp can
        # require a small lift to keep the elbow inside its native joint limit.
        for turn, lift in ((0.08, 0.0), (0.04, 0.0), (0.02, 0.0), (0.04, 0.02), (0.02, 0.01)):
            rotation = Rotation.from_rotvec(delta * min(1.0, turn / angle)) * observed
            position = world[support] - rotation.apply(local[support])
            corners = rotation.apply(local) + position
            position[2] += max(0.0, 0.001 - float(corners[:, 2].min())) + lift
            target = multiply_poses(
                Pose(tuple(position), tuple(rotation.as_quat())), held_tf.invert()
            )
            path = self.scene.floor_descent(
                start=self.session.arm(), target=target, bodies=bodies, held_tf=held_tf
            )
            endpoint_margin = (
                FloorApproachPreference.native_margin(arm=path[-1], limits=self.scene._arm_limits)
                if path
                else None
            )
            reserve_ok = endpoint_margin is not None and endpoint_margin >= 0.01
            before_tilt = float(np.arccos(np.clip(observed.as_matrix()[2, 2], -1.0, 1.0)))
            self.session._write(
                record={
                    "kind": "blade_leveling_candidate",
                    "t": self.session.ticks,
                    "turn": turn,
                    "lift": lift,
                    "before_tilt": before_tilt,
                    "path_found": path is not None,
                    "endpoint_native_joint_margin": endpoint_margin,
                    "required_joint_reserve": 0.01,
                    "joint_reserve_ok": reserve_ok,
                    "path_waypoints": None if path is None else len(path),
                    "goal_arm": None if not path else np.asarray(path[-1]).tolist(),
                    "target_tool_position": position.tolist(),
                    "target_tool_orientation": rotation.as_quat().tolist(),
                }
            )
            if not path or not reserve_ok:
                continue
            converged = None
            execution_error = None
            try:
                converged = self.motion.follow(
                    path=path,
                    grip=1.0,
                    tol=0.005,
                    final_tol=0.005,
                    max_ticks=self.leveling_tick_budget(path_waypoints=len(path)),
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

    def contact_control_ceiling(self, *, cube: str, region: str) -> float:
        """Reset strokes use the same native overlap height accepted at placement."""
        ceiling = self.cube_contact_ceiling(cube=cube)
        if region == "blocks_init_region":
            return ceiling
        return min(ceiling, self.floor_clearance + 0.003)

    def contact_orientation_candidates(
        self,
        *,
        tool_yaws: tuple[float, ...],
        region: str,
        narrow_contact: bool,
    ) -> list[tuple[float, bool, float | None, bool]]:
        """Try broad reset contact about the blade long axis before observed lean."""
        candidates: list[tuple[float, bool, float | None, bool]] = []
        if region == "blocks_init_region" and not narrow_contact:
            candidates.extend((yaw, True, 0.2, True) for yaw in tool_yaws)
        candidates.extend(
            (yaw, preserve, tilt, False)
            for yaw, preserve, tilt in self.floor_orientation_candidates(tool_yaws=tool_yaws)
        )
        return candidates

    def floor_orientation_candidates(
        self, *, tool_yaws: tuple[float, ...]
    ) -> list[tuple[float, bool, float | None]]:
        """Try existing and nearby poses before broader tilts within the controller reserve."""
        from scipy.spatial.transform import Rotation

        candidates: list[tuple[float, bool, float | None]] = [
            (yaw, preserve, None) for preserve in (False, True) for yaw in tool_yaws
        ]
        rotation = Rotation.from_quat(self.session.quaternion(name="wiper_0"))
        observed = float(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1)))
        ceiling = min(0.95, self.scene.max_tool_tilt - 0.15)
        nearby: list[float] = []
        for delta in (0.02, 0.04, -0.02, -0.04):
            tilt = observed + delta
            if delta > 0 and observed < ceiling:
                tilt = min(tilt, ceiling)
            if (
                0.0 < tilt <= ceiling
                and not np.isclose(tilt, observed, rtol=0, atol=1e-12)
                and not any(np.isclose(tilt, prior, rtol=0, atol=1e-12) for prior in nearby)
            ):
                nearby.append(tilt)
                candidates.extend((yaw, True, tilt) for yaw in tool_yaws)
        # A slipped overhead grasp can need .99 rad to keep the elbow away
        # from its native limit. This last fallback retains the .10-rad
        # loaded-unload reserve; it does not change either route/guard limit.
        for tilt in (0.2, 0.4, 0.6, 0.8, 0.95, 0.99):
            candidate_ceiling = self.scene.max_tool_tilt - 0.10 if tilt == 0.99 else ceiling
            if (
                tilt <= candidate_ceiling
                and not np.isclose(tilt, observed, rtol=0, atol=1e-12)
                and not any(np.isclose(tilt, prior, rtol=0, atol=1e-12) for prior in nearby)
            ):
                nearby.append(tilt)
                candidates.extend((yaw, True, tilt) for yaw in tool_yaws)
        return candidates

    def floor_tool_pose(
        self,
        *,
        xy: np.ndarray,
        yaw: float,
        preserve_tilt: bool = True,
        tilt: float | None = None,
        blade_axis: bool = False,
    ) -> Pose:
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
        observed_tilt = float(np.arccos(np.clip(target_rotation[2, 2], -1.0, 1.0)))
        if not preserve_tilt or observed_tilt > self.scene.max_tool_tilt:
            target_rotation = Rotation.from_euler("z", yaw).as_matrix()
        if tilt is not None:
            if not np.isfinite(tilt) or not 0.0 <= tilt <= self.scene.max_tool_tilt:
                raise ValueError("Floor tilt candidate must satisfy the existing planning limit")
            axis = np.cross([0.0, 0.0, 1.0], target_rotation[:, 2])
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 1e-9 else np.array([np.cos(yaw), np.sin(yaw), 0.0])
            current_tilt = float(np.arccos(np.clip(target_rotation[2, 2], -1.0, 1.0)))
            target_rotation = (
                Rotation.from_rotvec(axis * (tilt - current_tilt)).as_matrix() @ target_rotation
            )
            candidate_yaw = float(np.arctan2(target_rotation[1, 0], target_rotation[0, 0]))
            target_rotation = (
                Rotation.from_euler("z", yaw - candidate_yaw).as_matrix() @ target_rotation
            )
        blade_rotation = data.geom_xmat[blade].reshape(3, 3)
        if blade_axis:
            if tilt is None:
                raise ValueError("Native blade-axis candidate requires an explicit tilt")
            local_blade_rotation = observed.T @ blade_rotation
            target_rotation = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("x", tilt).as_matrix()
                @ local_blade_rotation.T
            )
        world_corners = np.array([
            data.geom_xpos[blade] + blade_rotation @ (model.geom_size[blade] * signs)
            for signs in product((-1.0, 1.0), repeat=3)
        ])
        body_corners = (world_corners - data.xpos[body]) @ observed
        bottom = float((body_corners @ target_rotation.T)[:, 2].min())
        return Pose(
            (float(xy[0]), float(xy[1]), self.floor_clearance - bottom),
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

    @staticmethod
    def transport_route_burden(*, path: list[tuple[float, float, float]]) -> float:
        """Use the native base planner's translation-plus-wrapped-yaw metric."""
        delta = np.diff(np.asarray(path), axis=0)
        return float(
            np.linalg.norm(delta[:, :2], axis=1).sum()
            + np.abs((delta[:, 2] + np.pi) % (2 * np.pi) - np.pi).sum()
        )

    def transport_base_candidates(
        self, *, target: tuple[float, float, float]
    ) -> list[tuple[str, list[tuple[float, float, float]]]]:
        """Retain native fallback and try the aisle above the native goal region."""
        from kinder_models.dynamic3d.utils import (
            MujocoTidyBotRobotObjectType,
            get_bounding_box,
        )

        result = []
        direct = self.scene.plan_base(target=target)
        if direct is not None:
            result.append(("native", direct))
        start = self.session.base()
        core = self.session.env.unwrapped._object_centric_env
        ranges = core.task_config["regions"]["sweep_region"]["ranges"]
        (robot,) = self.session.state.get_objects(MujocoTidyBotRobotObjectType)
        width, depth, _ = get_bounding_box(self.session.state, robot)
        # A chassis half-diagonal clears every heading above this region. The
        # extra 20 mm is the existing base planning clearance, not a goal change.
        aisle_y = max(
            start[1], target[1], max(box[3] for box in ranges) + np.hypot(width, depth) / 2 + 0.02
        )
        waypoints = [(start[0], aisle_y, start[2]), (target[0], aisle_y, target[2]), target]
        path = [start]
        for waypoint in waypoints:
            leg = self.scene.plan_base(target=waypoint, start=path[-1])
            if leg is None:
                break
            path.extend(leg[1:])
        else:
            result.append(("upper_aisle", path))
        return result

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
            candidates = self.transport_base_candidates(target=target)
            safe_paths = []
            for label, path in candidates:
                clear = True
                for base in path:
                    self.scene.sync(base=base)
                    self.scene.capture_path_rejections = True
                    blocked = self.scene.in_collision(
                        joints=self.scene.planning_fingers(arm=arm, state=0.5),
                        bodies=self.scene.bodies(),
                        held=self.scene.wiper_body,
                        held_tf=held_tf,
                    )
                    self.scene.capture_path_rejections = False
                    if blocked:
                        self.session._write(
                            record={
                                "kind": "transport_rejected",
                                "stowed": stow,
                                "reason": "carried_collision",
                                "base": base,
                                "target": target,
                                "t": self.session.ticks,
                                "checker": self.scene._last_collision_rejection,
                                "held_position": list(held_tf.position),
                                "held_orientation": list(held_tf.orientation),
                                "native_qpos": self.session.mj_data.qpos.tolist(),
                            }
                        )
                        clear = False
                        break
                if clear:
                    safe_paths.append((self.transport_route_burden(path=path), label, path))
            self.scene.sync()
            clear = bool(safe_paths)
            if clear:
                burden, label, path = min(safe_paths, key=lambda item: item[0])
                self.session._write(
                    record={
                        "kind": "transport_route_selected",
                        "t": self.session.ticks,
                        "stowed": stow,
                        "route": label,
                        "burden": burden,
                        "path": path,
                        "target": target,
                        "held_position": list(held_tf.position),
                        "held_orientation": list(held_tf.orientation),
                    }
                )
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
        """Track stow with live grip checks and bounded grasp-frame replanning."""
        from pybullet_helpers.geometry import multiply_poses
        from scipy.spatial.transform import Rotation

        self.require_handle(phase="before transport")
        goal = self.wiper_stow_goal()
        if np.max(np.abs(self.session.arm() - goal)) < 0.03:
            return
        for _ in range(8):
            start = self.session.arm().copy()
            if np.max(np.abs(ArmMath.wrap(delta=start - goal))) < 0.025:
                return
            held_tf = multiply_poses(
                self.scene.ee_now().invert(),
                Pose(
                    tuple(self.session.position(name="wiper_0")),
                    self.session.quaternion(name="wiper_0"),
                ),
            )

            def grasp_drifted(*, reference: Pose = held_tf) -> bool:
                observed = multiply_poses(
                    self.scene.ee_now().invert(),
                    Pose(
                        tuple(self.session.position(name="wiper_0")),
                        self.session.quaternion(name="wiper_0"),
                    ),
                )
                translation = float(
                    np.linalg.norm(np.asarray(observed.position) - np.asarray(reference.position))
                )
                rotation = float(
                    (
                        Rotation.from_quat(observed.orientation)
                        * Rotation.from_quat(reference.orientation).inv()
                    ).magnitude()
                )
                return translation > 0.01 or rotation > 0.06

            path = self.scene.plan_arm(
                goal=goal, bodies=self.scene.bodies(), held=self.scene.wiper_body, held_tf=held_tf
            )
            if path is None or not self.motion.follow(
                path=path,
                grip=1.0,
                final_tol=0.025,
                tick_guard=lambda: self.require_handle(phase="upright transport"),
                stop_condition=grasp_drifted,
            ):
                raise ExecutionError("No collision-free stow for the physically held wiper")
            self.require_handle(phase="upright transport")
            actual = self.session.arm()
            # Cartesian planning may return another valid IK branch for the goal pose.
            planned_goal = np.asarray(path[-1][:7]) if path else start
            if np.max(np.abs(ArmMath.wrap(delta=actual - planned_goal))) < 0.025:
                return
            # follow(True) also means an explicit drift stop, not arrival.
            if not grasp_drifted():
                raise ExecutionError("Stow stopped without attaining its physical arm target")
            if np.max(np.abs(ArmMath.wrap(delta=actual - start))) < 1e-4:
                raise ExecutionError("Stow grasp replanning made no arm progress")
        raise ExecutionError("Stow exhausted eight grasp replanning attempts")

    def place_transport_stance(self, *, position: np.ndarray) -> tuple[float, float, float]:
        """Try the original bearing, then the north aisle, with native carried-route checks."""
        from pybullet_helpers.geometry import multiply_poses

        candidates = list(self.stances(target=position, where="floor"))
        north = (float(position[0]), float(position[1] + self.distance), -np.pi / 2)
        if not any(np.allclose(candidate, north, rtol=0.0, atol=1e-9) for candidate in candidates):
            candidates.append(north)
        arm = self.session.arm().copy()
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        try:
            for target in candidates:
                self.scene.sync()
                for _, path in self.transport_base_candidates(target=target):
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
                    if clear:
                        return target
        finally:
            self.scene.sync()
        raise ExecutionError("No native carried base route to a wiper placement stance")

    def place_target_orientation(self) -> tuple[float, float, float, float]:
        """Place upright at the center of a declared native start yaw interval."""
        from scipy.spatial.transform import Rotation

        core = self.session.env.unwrapped._object_centric_env
        region = next(
            region for _, name, region in core.task_config["initial_state"] if name == "wiper_0"
        )
        ranges = core.task_config["regions"][region].get("yaw_ranges", [[0.0, 360.0]])
        if not ranges:
            raise ExecutionError("Wiper start region has no yaw interval")
        low, high = ranges[0]
        if not np.isfinite((low, high)).all() or low > high:
            raise ExecutionError("Unsupported native wiper start yaw interval")
        quaternion = Rotation.from_euler("z", np.radians((low + high) / 2)).as_quat()
        return (
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )

    def place_wiper_at_start(self) -> str:
        from pybullet_helpers.geometry import Pose, multiply_poses
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        self.stow_wiper()
        position, _ = self.session.initial_pose(name="wiper_0")
        orientation = self.place_target_orientation()
        stance = self.place_transport_stance(position=position)
        self.transport_wiper(target=stance)
        self.require_handle(phase="base transport")
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        path = None
        bodies = self.scene.bodies()
        for height in (0.001, 0.30):
            body_target = Pose((float(position[0]), float(position[1]), height), orientation)
            ee_target = multiply_poses(body_target, held_tf.invert())
            hover = Pose(
                tuple(np.asarray(ee_target.position) + [0.0, 0.0, 0.15]), ee_target.orientation
            )
            self.scene.sync()
            solutions = [
                solution
                for solution in ikfast_closest_inverse_kinematics(
                    self.scene.robot, world_from_target=hover
                )
                if self.scene.within_arm_limits(arm=solution[:7])
            ]
            for solution in solutions[:12]:
                candidate = self.scene.plan_arm(
                    goal=solution[:7],
                    bodies=bodies,
                    held=self.scene.wiper_body,
                    held_tf=held_tf,
                )
                if candidate is None:
                    continue
                descent = self.scene.floor_descent(
                    start=np.asarray(candidate[-1]),
                    target=ee_target,
                    bodies=bodies,
                    held_tf=held_tf,
                )
                if descent is not None:
                    path = candidate
                    break
            if path is not None:
                break
        if path is None or not self.motion.follow(path=path, grip=1.0):
            raise ExecutionError("No collision-free wiper placement approach")
        self.require_handle(phase="wiper placement approach")
        held_tf = multiply_poses(
            self.scene.ee_now().invert(),
            Pose(
                tuple(self.session.position(name="wiper_0")),
                self.session.quaternion(name="wiper_0"),
            ),
        )
        ee_target = multiply_poses(body_target, held_tf.invert())
        hover = Pose(
            tuple(np.asarray(ee_target.position) + [0.0, 0.0, 0.15]), ee_target.orientation
        )
        path = self.scene.floor_descent(
            start=self.session.arm(),
            target=ee_target,
            bodies=bodies,
            held_tf=held_tf,
        )
        if path is None or not self.motion.follow(path=path, grip=1.0, final_tol=0.005):
            raise ExecutionError("No collision-free wiper placement descent")
        self.motion.set_gripper(command=0.0)
        retreat = self.scene.linear_path(
            start=self.session.arm(), target=hover, bodies=bodies, max_jump=0.6
        )
        if retreat is None or not self.motion.follow(path=retreat, grip=0.0):
            raise ExecutionError("No collision-free retreat from released wiper")
        from typing import cast

        from .regions import SimpleRegions
        from .session import SweepSimpleSession

        checks = SimpleRegions.validate(session=cast(SweepSimpleSession, self.session)).checks
        if not all(value for key, value in checks.items() if key.startswith("wiper_0:")):
            raise ExecutionError("Released wiper did not settle in its approved start contract")
        return "Wiper physically released and observed settled in its approved start region"

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
    _native_palm: Any = PrivateAttr(default=None)
    _native_base_clearance: Any = PrivateAttr(default=None)
    _planning_base: tuple[float, float, float] | None = PrivateAttr(default=None)

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
            # The shared attachment checker permits gripper/tool proximity.
            # A 5-mm palm margin stranded a physically separated loaded grasp
            # at a 4.794-mm gap. Retain penetration rejection at the palm and
            # the existing positive clearance for the actual arm links.
            clearance = 0.0 if contact[4] == 10 else 0.005
            if contact[4] <= 10 and contact[8] < clearance:
                if (
                    contact[4] == 10
                    and held == self.wiper_body
                    and held_tf is not None
                    and len(joints) == 13
                ):
                    from hitl_pmp.environments.sweep_simple3d.native_palm import NativePalmClearance

                    if self._native_palm is None:
                        self._native_palm = NativePalmClearance(
                            model=self.session.mj_model, live_data=self.session.mj_data
                        )
                    # Refine only the zero-margin palm proxy rejection against
                    # the native candidate geometry, never finger contacts or
                    # positive arm-clearance requirements. Ambiguous zero stays blocked.
                    native_clearance = self._native_palm.distance(
                        joints=joints, tool_pose=tool_pose, base=self._planning_base
                    )
                    if native_clearance > 0.0:
                        continue
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
            *(
                f"{side}_{part}"
                for side in ("left", "right")
                for part in (
                    "outer_knuckle",
                    "outer_finger",
                    "inner_finger",
                    "inner_finger_pad",
                    "inner_knuckle",
                )
            ),
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
        clearance = min(
            float(mujoco.mj_geomDistance(model, data, geom, chassis_geom, distance_cap, None))
            for geom in geoms
        )
        if clearance == 0.0 and distance_threshold == 0.0:
            # Pinned MuJoCo can also return an ambiguous zero at a tiny cap.
            # Resolve only the zero-margin boolean with the native contact
            # pipeline at this exact candidate, on private data. Do not infer
            # positive safety margins from absence of native contact.
            mujoco.mj_forward(model, data)
            distances = [
                float(contact.dist)
                for contact in data.contact
                if (contact.geom1 == chassis_geom and contact.geom2 in geoms)
                or (contact.geom2 == chassis_geom and contact.geom1 in geoms)
            ]
            return min(distances) if distances else distance_cap
        return clearance

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
        self._planning_base = (bx, by, yaw) if base is None else (base[0], base[1], base[2])
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
        self,
        *,
        target: tuple[float, float, float],
        margin: float = 0.02,
        start: tuple[float, float, float] | None = None,
    ) -> list[tuple[float, float, float]] | None:
        del margin
        from kinder_models.dynamic3d.utils import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            run_base_motion_planning,
        )
        from spatialmath import SE2

        from .native_chassis import NativeChassisClearance

        if self._native_base_clearance is None:
            self._native_base_clearance = NativeChassisClearance(
                model=self.session.mj_model, live_data=self.session.mj_data
            )
        held_wiper = self.session.gripper() > 0.2 and FloorGrip.has_bilateral_contact(
            session=self.session
        )
        rejection = self._native_base_clearance.first_route_rejection(
            path=[target], held_wiper=held_wiper
        )
        if rejection is not None:
            self.session._write(
                record={
                    "kind": "native_chassis_route_rejected",
                    "t": self.session.ticks,
                    "detail": rejection,
                }
            )
            return None

        state = self.session.state
        if start is not None:
            from kinder_models.dynamic3d.utils import MujocoTidyBotRobotObjectType

            state = state.copy()
            (robot,) = state.get_objects(MujocoTidyBotRobotObjectType)
            for feature, value in zip(
                ("pos_base_x", "pos_base_y", "pos_base_rot"), start, strict=True
            ):
                state.set(robot, feature, value)
        path = run_base_motion_planning(
            state=state,
            target_base_pose=SE2(*target),
            x_bounds=WORLD_X_BOUNDS,
            y_bounds=WORLD_Y_BOUNDS,
            seed=0,
            disable_collision_objects=(["wiper_0"] if held_wiper else []),
        )
        if path is None:
            return None
        result = [(float(p.x), float(p.y), float(p.theta())) for p in path]
        rejection = self._native_base_clearance.first_route_rejection(
            path=result, held_wiper=held_wiper
        )
        if rejection is not None:
            self.session._write(
                record={
                    "kind": "native_chassis_route_rejected",
                    "t": self.session.ticks,
                    "detail": rejection,
                }
            )
            return None
        return result


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
