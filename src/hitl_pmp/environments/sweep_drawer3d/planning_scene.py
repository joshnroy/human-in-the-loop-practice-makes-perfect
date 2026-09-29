"""The PyBullet planning model, with the three things kinder-models' PyBulletSim lacks.

kinder-models plans arm motions in a PyBulletSim built from the state's static colliders
and cubes. For this scene that model omits (1) the drawers -- drawers carry only a slide
`pos`, so no box of a drawer exists to collide with, and arm paths planned there pass
straight through its walls; (2) the mobile base, so an arm reaching low near the robot
can pass through its own chassis; and (3) the wiper, so a carried wiper is never checked.
All three are added here from the compiled MuJoCo model, which is the scene's own CAD:
the drawers' boxes are re-posed from the live simulator on every `sync`.

The island has six drawers, two rows of three, and all six are modelled, not only the one
the task opens. The lower row's faces and handles stand where a floor cube by the island
lies: with only the task's drawer in the model, a plan to reach such a cube is "collision
free" straight through a drawer face, and the arm is stopped by it on execution. The island slab
(`collider:kitchen_island:7`) stays in: it is faithful, and the grasps below are placed so
they do not need it removed.
"""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from scipy.spatial.transform import Rotation, Slerp

from .session import SweepDrawerSession
from .types import SweepDrawerScene

Joints = list[float]
Quat = tuple[float, float, float, float]


class PlanningScene(BaseModel):
    """PyBulletSim + drawer boxes + chassis box + wiper compound, synced from a session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: SweepDrawerSession
    # False leaves the island's five other drawers out of every collision check, as the
    # scene was before they were modelled: for measuring what modelling them is worth.
    other_drawers: bool = True

    _sim: Any = PrivateAttr(default=None)
    _p: Any = PrivateAttr(default=None)
    _drawer: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _other_drawers: list[tuple[int, int]] = PrivateAttr(default_factory=list)
    _chassis: int = PrivateAttr(default=-1)
    _wiper: int = PrivateAttr(default=-1)
    _arm_limits: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 2)))

    def model_post_init(self, __context: Any) -> None:  # noqa: PLR0917
        import pybullet
        from kinder_models.dynamic3d.utils import PyBulletSim
        from pybullet_helpers.utils import create_pybullet_block

        self._p = pybullet
        self._sim = PyBulletSim(self.session.state)
        cid = self._sim.physics_client_id
        m = self.session.mj_model
        import mujoco

        limits = []
        for number in range(1, 8):
            joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot_joint_{number}")
            limits.append(m.jnt_range[joint] if m.jnt_limited[joint] else (-np.inf, np.inf))
        self._arm_limits = np.asarray(limits, dtype=float)

        task = (SweepDrawerScene.DRAWER, SweepDrawerScene.DRAWER + "_handle")
        for g in range(m.ngeom):
            body_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            collides = m.geom_contype[g] + m.geom_conaffinity[g] > 0
            if (
                body_name.startswith(SweepDrawerScene.ISLAND_DRAWERS)
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
                and collides
            ):
                half = (
                    float(m.geom_size[g][0]),
                    float(m.geom_size[g][1]),
                    float(m.geom_size[g][2]),
                )
                block = create_pybullet_block((0.4, 0.3, 0.2, 1.0), half, cid)
                (self._drawer if body_name in task else self._other_drawers).append((g, block))
        hx, hy = SweepDrawerScene.CHASSIS_HALF
        self._chassis = create_pybullet_block((0.2, 0.2, 0.2, 1.0), (hx, hy, 0.15), cid)
        w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, SweepDrawerScene.WIPER)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == w]
        shape = pybullet.createCollisionShapeArray(
            shapeTypes=[pybullet.GEOM_BOX] * len(geoms),
            halfExtents=[[float(v) for v in m.geom_size[g]] for g in geoms],
            collisionFramePositions=[[float(v) for v in m.geom_pos[g]] for g in geoms],
            collisionFrameOrientations=[
                [float(m.geom_quat[g][k]) for k in (1, 2, 3, 0)] for g in geoms
            ],
            physicsClientId=cid,
        )
        self._wiper = pybullet.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=shape, basePosition=(0, 0, -5), physicsClientId=cid
        )
        self.sync()

    # ------------------------------------------------------------------ model
    @property
    def cid(self) -> int:
        return int(self._sim.physics_client_id)

    @property
    def robot(self) -> Any:
        return self._sim.robot

    @property
    def drawer_bodies(self) -> list[int]:
        """The boxes of the drawer the task opens, handle included."""
        return [b for _, b in self._drawer]

    @property
    def other_drawer_bodies(self) -> list[int]:
        """The boxes of the island's five other drawers, handles included."""
        return [b for _, b in self._other_drawers]

    @property
    def chassis_body(self) -> int:
        return self._chassis

    @property
    def wiper_body(self) -> int:
        return self._wiper

    def cube_body(self, *, cube: str) -> int:
        return int(self._sim._cubes[cube])

    def collider_body(self, *, name: str) -> int | None:
        body = self._sim._static_colliders.get(name)
        return None if body is None else int(body)

    def sync(self, *, base: tuple[float, float, float] | None = None) -> None:
        """Pose everything from the session's current state (optionally at another base)."""
        from pybullet_helpers.geometry import Pose, set_pose

        x = self.session.state.copy()
        robot = x.get_object_from_name(SweepDrawerScene.ROBOT)
        if base is not None:
            for k, v in zip(("pos_base_x", "pos_base_y", "pos_base_rot"), base, strict=True):
                x.set(robot, k, v)
        self._sim.set_state(x)
        d = self.session.mj_data
        import mujoco

        for g, body in self._drawer + self._other_drawers:
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            set_pose(body, Pose(tuple(d.geom_xpos[g]), (q[1], q[2], q[3], q[0])), self.cid)
        bx, by, bt = (float(x.get(robot, k)) for k in ("pos_base_x", "pos_base_y", "pos_base_rot"))
        chassis = Pose((bx, by, 0.233), (0.0, 0.0, float(np.sin(bt / 2)), float(np.cos(bt / 2))))
        set_pose(self._chassis, chassis, self.cid)
        set_pose(
            self._wiper,
            Pose(
                tuple(self.session.position(name=SweepDrawerScene.WIPER)),
                self.session.quaternion(name=SweepDrawerScene.WIPER),
            ),
            self.cid,
        )

    def bodies(
        self,
        *,
        without_cubes: Iterable[str] = (),
        without_drawer: bool = False,
        without: Iterable[int] = (),
    ) -> set[int]:
        out = set(self._sim.get_collision_bodies())
        out |= set(self.drawer_bodies)
        if self.other_drawers:
            out |= set(self.other_drawer_bodies)
        out.add(self._chassis)
        for c in without_cubes:
            out.discard(self.cube_body(cube=c))
        if without_drawer:
            out -= set(self.drawer_bodies)
        out -= set(without)
        return out

    @staticmethod
    def fingers(*, arm: Sequence[float] | np.ndarray, state: float = 0.0) -> Joints:
        """A 13-joint PyBullet configuration: 7 arm joints + the 6 mimic finger joints."""
        return [float(v) for v in arm[:7]] + [state, state, state, state, -state, -state]

    def planning_fingers(self, *, arm: Sequence[float] | np.ndarray, state: float = 0.0) -> Joints:
        """Planning articulation; environments may bind measured physical gripper joints."""
        return self.fingers(arm=arm, state=state)

    def within_arm_limits(self, *, arm: Sequence[float] | np.ndarray) -> bool:
        """The planning URDF allows wider bends than this compiled physical robot."""
        q = np.asarray(arm[:7], dtype=float)
        return bool(np.all(q >= self._arm_limits[:, 0]) and np.all(q <= self._arm_limits[:, 1]))

    def in_collision(
        self,
        *,
        joints: Joints,
        bodies: set[int],
        held: int | None = None,
        held_tf: Any = None,
        margin: float = 0.0,
    ) -> bool:
        from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object

        if not self.within_arm_limits(arm=joints):
            return True
        return bool(
            check_collisions_with_held_object(
                self.robot, bodies, self.cid, held, held_tf, joints, distance_threshold=margin
            )
        )

    def ik(self, *, pose: Any, seed: Sequence[float] | np.ndarray) -> np.ndarray | None:
        """IKFast solution closest to `seed` (7 arm joints), or None if unreachable."""
        from pybullet_helpers.inverse_kinematics import InverseKinematicsError, inverse_kinematics

        self.robot.set_joints(self.planning_fingers(arm=seed))
        try:
            sol = inverse_kinematics(self.robot, pose, set_joints=False)
        except InverseKinematicsError:
            return None
        q = np.asarray(sol[:7], dtype=float)
        if self.within_arm_limits(arm=q):
            return q
        from pybullet_helpers.ikfast.utils import ikfast_closest_inverse_kinematics

        for candidate in ikfast_closest_inverse_kinematics(self.robot, world_from_target=pose):
            q = np.asarray(candidate[:7], dtype=float)
            if self.within_arm_limits(arm=q):
                return q
        return None

    def fk(self, *, arm: Sequence[float] | np.ndarray) -> Any:
        self.robot.set_joints(self.planning_fingers(arm=arm))
        return self.robot.get_end_effector_pose()

    def ee_now(self) -> Any:
        self.sync()
        return self.fk(arm=self.session.arm())

    # ------------------------------------------------------------------ planning
    def plan_arm(
        self,
        *,
        goal: Sequence[float] | np.ndarray | None,
        bodies: set[int],
        start: Sequence[float] | np.ndarray | None = None,
        base: tuple[float, float, float] | None = None,
        held: int | None = None,
        held_tf: Any = None,
    ) -> list[np.ndarray] | None:
        """BiRRT in joint space (kinder-models' planner) against this scene."""
        from pybullet_helpers.motion_planning import run_motion_planning

        if goal is None:
            return None
        self.sync(base=base)
        q0 = self.session.arm() if start is None else np.asarray(start)
        plan = run_motion_planning(
            self.robot,
            self.planning_fingers(arm=q0),
            self.planning_fingers(arm=goal),
            collision_bodies=bodies,
            seed=0,
            physics_client_id=self.cid,
            held_object=held,
            base_link_to_held_obj=held_tf,
        )
        if plan is None or any(not self.within_arm_limits(arm=q) for q in plan):
            return None
        return [np.asarray(q[:7], dtype=float) for q in plan]

    def linear_path(
        self,
        *,
        start: Sequence[float] | np.ndarray,
        target: Any,
        step: float = 0.01,
        bodies: set[int] | None = None,
        finger_state: float = 0.0,
        margin: float = 0.0,
        max_jump: float = 0.35,
    ) -> list[np.ndarray] | None:
        """A straight end-effector line from FK(start) to `target`, solved by IK at every
        `step` metres from the previous solution. None if any waypoint is unreachable,
        jumps (an IK branch switch), or collides with `bodies`."""
        from pybullet_helpers.geometry import Pose

        p0 = self.fk(arm=start)
        a, b = np.array(p0.position), np.array(target.position)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
        slerp = Slerp([0.0, 1.0], Rotation.from_quat([p0.orientation, target.orientation]))
        q = np.asarray(start, dtype=float)
        out = []
        for i in range(1, n + 1):
            t = i / n
            pose = Pose(tuple(a + t * (b - a)), tuple(slerp(t).as_quat()))
            sol = self.ik(pose=pose, seed=q)
            if sol is None:
                return None
            sol = q + ArmMath.wrap(delta=sol - q)
            if not self.within_arm_limits(arm=sol) or np.max(np.abs(sol - q)) > max_jump:
                return None
            if bodies is not None and self.in_collision(
                joints=self.planning_fingers(arm=sol, state=finger_state),
                bodies=bodies,
                margin=margin,
            ):
                return None
            out.append(sol)
            q = sol
        return out

    def plan_base(
        self, *, target: tuple[float, float, float], margin: float = 0.02
    ) -> list[tuple[float, float, float]] | None:
        """Base BiRRT over the floor footprint. Movables on the counter or in the drawer sit
        above the chassis and are not obstacles; the open drawer is (kinder-models' base
        planner skips drawers). The chassis is inflated by `margin`."""
        from kinder.envs.dynamic3d.object_types import MujocoObjectType
        from kinder_models.dynamic3d.utils import (
            WORLD_X_BOUNDS,
            WORLD_Y_BOUNDS,
            get_overhead_kinematic2ds,
        )
        from prpl_utils.motion_planning import BiRRT
        from prpl_utils.utils import get_signed_angle_distance, wrap_angle
        from spatialmath import SE2
        from tomsgeoms2d.structs import Rectangle
        from tomsgeoms2d.utils import geom2ds_intersect

        state = self.session.state
        skip = set(self.session.elevated_movables()) | {SweepDrawerScene.ROBOT}
        movable_or_fixture = {o.name for o in state.get_objects(MujocoObjectType)}
        obstacles = [
            g
            for n, g in get_overhead_kinematic2ds(state).items()
            if n not in skip and n in movable_or_fixture
        ]
        dp = self.session.drawer_pos()
        if dp > 0.005:
            s = SweepDrawerScene
            front = s.DRAWER_HANDLE_FRONT_X + dp
            obstacles.append(
                Rectangle(
                    s.COUNTER_EDGE_X - 0.005,
                    -s.DRAWER_HALF_WIDTH,
                    front - s.COUNTER_EDGE_X + 0.005,
                    2 * s.DRAWER_HALF_WIDTH,
                    0.0,
                )
            )
        hx, hy = SweepDrawerScene.CHASSIS_HALF
        dims = [2 * hx + 2 * margin, 2 * hy + 2 * margin]
        rng = np.random.default_rng(0)

        def collides(pt: Any) -> bool:  # noqa: PLR0917
            rect = Rectangle.from_center(
                pt.x, pt.y, dims[0], dims[1], rotation_about_center=pt.theta()
            )
            return any(geom2ds_intersect(rect, o) for o in obstacles)

        def sample(_: Any) -> Any:  # noqa: PLR0917
            return SE2(
                rng.uniform(*WORLD_X_BOUNDS),
                rng.uniform(*WORLD_Y_BOUNDS),
                rng.uniform(-np.pi, np.pi),
            )

        def extend(p1: Any, p2: Any) -> Any:  # noqa: PLR0917
            dx, dy = p2.x - p1.x, p2.y - p1.y
            dth = get_signed_angle_distance(p2.theta(), p1.theta())
            n = max(
                int(abs(dx) / 0.025) + 1, int(abs(dy) / 0.025) + 1, int(abs(dth) / (np.pi / 8)) + 1
            )
            x, y, th = p1.x, p1.y, p1.theta()
            yield SE2(x, y, th)
            for _ in range(n):
                x, y, th = x + dx / n, y + dy / n, wrap_angle(th + dth / n)
                yield SE2(x, y, th)

        def distance(p1: Any, p2: Any) -> float:  # noqa: PLR0917
            return float(
                np.hypot(p2.x - p1.x, p2.y - p1.y)
                + abs(get_signed_angle_distance(p2.theta(), p1.theta()))
            )

        goal = SE2(target[0], target[1], float((target[2] + np.pi) % (2 * np.pi) - np.pi))
        if collides(goal):
            return None
        start = SE2(*self.session.base())
        prefix: list[Any] = []
        if collides(start):
            dims[0], dims[1] = 2 * hx, 2 * hy
            if collides(start):
                # Parked against the drawer handle (the stock Sweep ends there): back
                # straight away from the island first, then plan from there.
                for back in (0.06, 0.12, 0.2):
                    moved = SE2(start.x + back, start.y, start.theta())
                    if not collides(moved):
                        prefix, start = [SE2(*self.session.base())], moved
                        break
        path = BiRRT(sample, extend, collides, distance, rng, 10, 100, 50).query(start, goal)
        if path is None:
            return None
        return [(float(p.x), float(p.y), float(p.theta())) for p in prefix + list(path)]


class ArmMath:
    """Joint-space helpers for the Gen3's continuous joints (1, 3, 5, 7)."""

    @staticmethod
    def wrap(*, delta: np.ndarray) -> np.ndarray:
        """Route a difference the short way round each continuous joint."""
        d = np.array(delta, dtype=float)
        for i in (0, 2, 4, 6):
            if abs(d[i]) > np.pi:
                d[i] = (d[i] + np.pi) % (2 * np.pi) - np.pi
        return d


class Orientations:
    """End-effector orientations (x = closing axis, z = approach) as (x, y, z, w)."""

    @staticmethod
    def from_axes(*, closing: np.ndarray, approach: np.ndarray) -> Quat:
        y = np.cross(approach, closing)
        q = Rotation.from_matrix(np.stack([closing, y, approach], axis=1)).as_quat()
        return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @staticmethod
    def top_down(*, yaw: float) -> Quat:
        return Orientations.from_axes(
            closing=np.array([np.cos(yaw), np.sin(yaw), 0.0]), approach=np.array([0.0, 0.0, -1.0])
        )

    @staticmethod
    def tilted(*, yaw: float, lean: float, alpha: float) -> tuple[Quat, np.ndarray]:
        """Approach tilted by `alpha` about the closing axis so the palm leans toward
        `lean` * v (v = closing axis turned +90 deg); returns (quaternion, approach)."""
        u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        v = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        a = np.array([0.0, 0.0, -np.cos(alpha)]) - np.sin(alpha) * lean * v
        return Orientations.from_axes(closing=u, approach=a), a
