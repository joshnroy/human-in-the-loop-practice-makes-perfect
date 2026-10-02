"""Checked physical wiper pickup shared by Simple floor controllers."""

from typing import Any

import numpy as np
from pybullet_helpers.geometry import Pose
from pydantic import BaseModel, ConfigDict

from .motion import ExecutionError, Motion
from .planning_scene import Orientations, PlanningScene
from .session import SweepPhysicalSession
from .types import GripperGeometry, KitchenScene

S = KitchenScene
G = GripperGeometry


class WiperHold:
    """Whether the wiper is in the hand."""

    # Held, the wiper's origin is 4 to 5 cm from the gripper (measured on five seeds).
    REACH = 0.15

    @staticmethod
    def in_hand(*, gripper: np.ndarray, wiper: np.ndarray) -> bool:
        return float(np.linalg.norm(np.asarray(gripper) - np.asarray(wiper))) < WiperHold.REACH


class Primitives(BaseModel):
    """Collision-checked wiper recovery for one physical session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    session: SweepPhysicalSession
    scene: PlanningScene
    motion: Motion

    def stances(self, *, target: np.ndarray, where: str) -> list[tuple[float, float, float]]:
        """Generic floor stances; the Simple controller supplies its own stance policy."""
        del where
        return [
            (
                float(target[0] - dist * np.cos(ang)),
                float(target[1] - dist * np.sin(ang)),
                float((ang + np.pi) % (2 * np.pi) - np.pi),
            )
            for dist in (0.55, 0.5, 0.6, 0.65)
            for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ]

    def resolve_at_actual_base(
        self,
        *,
        hover: Any,
        target: Any,
        bodies: set[int],
        descent_bodies: set[int],
        finger_state: float = 0.0,
        margin: float = 0.0,
        held: int | None = None,
        held_tf: Any = None,
        step: float = 0.01,
        ik_seed: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """Re-solve hover IK, the arm plan and the straight descent at the base pose the
        robot actually reached (the base stops within millimetres, not exactly)."""
        self.scene.sync()
        q_h = self.scene.ik(pose=hover, seed=S.HOME if ik_seed is None else ik_seed)
        if q_h is None:
            return None
        plan = self.scene.plan_arm(goal=q_h, bodies=bodies, held=held, held_tf=held_tf)
        if plan is None:
            return None
        down = self.scene.linear_path(
            start=q_h,
            target=target,
            step=step,
            bodies=descent_bodies,
            finger_state=finger_state,
            margin=margin,
        )
        return None if down is None else (plan, down)

    def pad_centers(self) -> list[np.ndarray]:
        import mujoco

        m, d = self.session.mj_model, self.session.mj_data
        out = []
        for n in ("robot_left_pad1", "robot_right_pad1"):
            out.append(
                np.array(
                    d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)], dtype=float
                )
            )
        return out

    def wiper_handle_geometry(self) -> tuple[int, int]:
        import mujoco

        m = self.session.mj_model
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, S.WIPER)
        handle = max(
            (g for g in range(m.ngeom) if m.geom_bodyid[g] == body),
            key=lambda g: float(m.geom_size[g][0] / m.geom_size[g][1]),
        )
        return handle, 0

    def wiper_in_hand(self, *, gripper: np.ndarray, wiper: np.ndarray) -> bool:
        return WiperHold.in_hand(gripper=gripper, wiper=wiper)

    def wiper_grasp_point(
        self, *, center: np.ndarray, axis: np.ndarray, along: float
    ) -> np.ndarray:
        return center + along * axis

    def wiper_grasp_standoff(self) -> float:
        """Distance behind the grasp point along negative end-effector approach."""
        return 0.035

    def wiper_grasp_offsets(self) -> tuple[float, ...]:
        return 0.0, 0.03, 0.06

    def wiper_approach_angles(self) -> tuple[float, ...]:
        return 0.0, 0.4, 0.7, 1.0, 1.57, 1.9

    def wiper_stow_goal(self) -> np.ndarray:
        return np.asarray(S.HOME)

    def wiper_pickup_carry_goal(self) -> np.ndarray:
        """Keep shared pickup completion at its existing stow posture."""
        return self.wiper_stow_goal()

    def wiper_grasp_yaw(self, *, axis: np.ndarray) -> float:
        return float(np.arctan2(axis[1], axis[0]) + np.pi / 2)

    def wiper_pick_targets_after_navigation(
        self, *, hover: Pose, target: Pose, along: float, approach: np.ndarray
    ) -> tuple[Pose, Pose]:
        """Shared default preserves the already checked native pickup targets."""
        del along, approach
        return hover, target

    def wiper_pickup_descent_clear(self, *, start: np.ndarray, path: list[np.ndarray]) -> bool:
        """Environment-specific pickup geometry validation; shared behavior unchanged."""
        return True

    def wiper_grasp_orientations(self, *, axis: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Ordered grasp frames; the shared lateral/overhead search is unchanged."""
        yaw = self.wiper_grasp_yaw(axis=axis)
        orientations = []
        for azimuth in np.linspace(yaw, yaw + 2 * np.pi, 8, endpoint=False):
            for angle in self.wiper_approach_angles():
                closing = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
                toward = np.array([np.sin(azimuth), -np.cos(azimuth), 0.0])
                approach = np.sin(angle) * toward + np.array([0.0, 0.0, -np.cos(angle)])
                orientations.append((closing, approach))
        return orientations

    def recover_wiper(self) -> str:
        """Grasp the observed handle and verify that it follows a physical lift."""
        from pybullet_helpers.geometry import Pose, multiply_poses, set_pose
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        data = self.session.mj_data
        # The long, narrow handle is distinguishable from the wide blade by local x.
        handle, handle_axis = self.wiper_handle_geometry()
        center = np.array(data.geom_xpos[handle])
        handle_axes = np.array(data.geom_xmat[handle]).reshape(3, 3)
        axis = handle_axes[:, handle_axis]
        before = self.session.position(name=S.WIPER).copy()
        wiper = Pose(tuple(before), self.session.quaternion(name=S.WIPER))
        set_pose(self.scene.wiper_body, wiper, self.scene.cid)
        rejected: dict[str, int] = {}
        orientations = self.wiper_grasp_orientations(axis=axis)
        for along in self.wiper_grasp_offsets():
            grasp_point = self.wiper_grasp_point(center=center, axis=axis, along=along)
            for closing, approach in orientations:
                goal = Pose(
                    tuple(grasp_point - approach * self.wiper_grasp_standoff()),
                    Orientations.from_axes(closing=closing, approach=approach),
                )
                hover = Pose(tuple(np.asarray(goal.position) - approach * 0.08), goal.orientation)
                for stance in self.stances(target=grasp_point, where="floor"):
                    path = self.scene.plan_base(target=stance)
                    if path is None:
                        rejected["base"] = rejected.get("base", 0) + 1
                        continue
                    base = path[-1]
                    self.scene.sync(base=base)
                    bodies = self.scene.bodies()
                    self.scene.robot.set_joints(self.scene.planning_fingers(arm=S.HOME))
                    solutions = ikfast_closest_inverse_kinematics(
                        self.scene.robot,
                        world_from_target=hover,
                    )
                    q_h, down, reach = None, None, None
                    for solution in solutions[:4]:
                        q_h = np.asarray(solution[:7], dtype=float)
                        if self.scene.in_collision(
                            joints=self.scene.planning_fingers(arm=q_h),
                            bodies=bodies,
                        ):
                            continue
                        down = self.scene.linear_path(
                            start=q_h,
                            target=goal,
                            bodies=bodies,
                            max_jump=0.6,
                        )
                        if down is None or not self.wiper_pickup_descent_clear(
                            start=q_h, path=down
                        ):
                            continue
                        reach = self.scene.plan_arm(
                            goal=q_h,
                            bodies=bodies | {self.scene.wiper_body},
                            start=S.HOME,
                            base=base,
                        )
                        if reach is not None:
                            break
                    if reach is None:
                        why = "hover IK" if q_h is None else "descent" if down is None else "reach"
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    self.motion.go_home(grip=0.0)
                    self.motion.set_gripper(command=0.0)
                    if not self.motion.drive(path=path, grip=0.0):
                        raise ExecutionError("wiper pickup base did not converge")
                    hover, goal = self.wiper_pick_targets_after_navigation(
                        hover=hover, target=goal, along=along, approach=approach
                    )
                    redo = self.resolve_at_actual_base(
                        hover=hover,
                        target=goal,
                        bodies=bodies | {self.scene.wiper_body},
                        descent_bodies=bodies,
                        ik_seed=q_h,
                    )
                    if redo is None:
                        raise ExecutionError("wiper pickup unavailable at actual base pose")
                    reach, down = redo
                    if not self.wiper_pickup_descent_clear(
                        start=np.asarray(reach[-1]) if reach else self.session.arm(), path=down
                    ):
                        raise ExecutionError("Native non-pad/tool collision blocks pickup descent")
                    if not self.motion.follow(path=reach, grip=0.0):
                        raise ExecutionError("wiper pickup approach did not converge")
                    if not self.motion.follow(path=down, grip=0.0, final_tol=0.006):
                        raise ExecutionError("wiper pickup descent did not converge")
                    self.motion.set_gripper(command=1.0)
                    ee = self.scene.ee_now()
                    grasped = Pose(
                        tuple(self.session.position(name=S.WIPER)),
                        self.session.quaternion(name=S.WIPER),
                    )
                    held_tf = multiply_poses(ee.invert(), grasped)
                    up = None
                    # A blade beneath a cabinet overhang cannot rise vertically.
                    # Retreat along the observed approach before lifting, checking
                    # both the arm and the carried blade at every waypoint.
                    for retreat in (0.0, 0.05, 0.10, 0.16, 0.22):
                        back_pos = np.asarray(ee.position) - retreat * approach
                        targets = [Pose(tuple(back_pos), ee.orientation)] if retreat else []
                        targets.append(Pose(tuple(back_pos + [0.0, 0.0, 0.16]), ee.orientation))
                        candidate: list[np.ndarray] = []
                        start = self.session.arm()
                        for target in targets:
                            segment = self.scene.linear_path(
                                start=start,
                                target=target,
                                bodies=bodies,
                                finger_state=G.CLOSED_PB,
                                max_jump=0.6,
                            )
                            if segment is None or any(
                                self.scene.in_collision(
                                    joints=self.scene.planning_fingers(arm=q, state=G.CLOSED_PB),
                                    bodies=bodies,
                                    held=self.scene.wiper_body,
                                    held_tf=held_tf,
                                )
                                for q in segment
                            ):
                                candidate = []
                                break
                            candidate.extend(segment)
                            start = segment[-1]
                        if candidate:
                            up = candidate
                            break
                    if up is None:
                        raise ExecutionError("no collision-free lift for the grasped wiper")
                    if not self.motion.follow(path=up, grip=1.0):
                        raise ExecutionError("wiper pickup lift did not converge")
                    after = self.session.position(name=S.WIPER)
                    ee = np.asarray(self.scene.ee_now().position)
                    if after[2] < before[2] + 0.05 or not self.wiper_in_hand(
                        gripper=ee, wiper=after
                    ):
                        raise ExecutionError(
                            f"wiper did not follow the lift: rise {after[2] - before[2]:.3f} m;"
                            f" gripper distance {np.linalg.norm(ee - after):.3f} m"
                        )
                    # Base navigation assumes the arm is stowed. A floor grasp
                    # leaves it extended below the counter, so stow while checking
                    # the actual carried blade before any navigation begins.
                    held_now = multiply_poses(
                        self.scene.ee_now().invert(),
                        Pose(tuple(after), self.session.quaternion(name=S.WIPER)),
                    )
                    stow = self.scene.plan_arm(
                        goal=self.wiper_pickup_carry_goal(),
                        bodies=self.scene.bodies(),
                        held=self.scene.wiper_body,
                        held_tf=held_now,
                    )
                    if stow is None:
                        raise ExecutionError("no collision-free stow for the recovered wiper")
                    # Long floor-to-home paths can exhaust the default tracker
                    # while still converging; retain its exact endpoint tolerance.
                    if not self.motion.follow(path=stow, grip=1.0, max_ticks=1200):
                        raise ExecutionError("recovered wiper stow did not converge")
                    if not self.wiper_in_hand(
                        gripper=np.asarray(self.scene.ee_now().position),
                        wiper=self.session.position(name=S.WIPER),
                    ):
                        raise ExecutionError("wiper lost during stow")
                    return f"observed wiper rise {after[2] - before[2]:.3f} m; rejected {rejected}"
        raise ExecutionError(f"no collision-free wiper grasp; rejected {rejected}")
