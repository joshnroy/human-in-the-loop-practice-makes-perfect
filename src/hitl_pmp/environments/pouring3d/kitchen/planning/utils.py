"""The kitchen's PyBullet planning scene, mirroring `kinder_models.dynamic3d.utils`.

`kinder_models.dynamic3d.utils.PyBulletSim` builds its scene from an
`ObjectCentricState`: KINDER's box-only static colliders, blocks for objects named
"cube", and bins or boxes added by hand. The kitchen's jug, skillet and dumplings are
none of those (the probe of 2026-11-02 found all five absent), and the tables' legs,
the stove's knobs and the skillet's handle are not boxes. So `KitchenPyBulletSim`
mirrors MuJoCo's own collision geometry instead: every contact geom of each named body
subtree becomes a link of one compound PyBullet body, with the same primitive (box,
cylinder, capsule, sphere) or the convex hull of the same mesh vertices, which is what
MuJoCo collides with. Poses still come from the `ObjectCentricState`, as upstream.

The robot is upstream's PyBullet Kinova Gen3 with its Robotiq gripper, not MuJoCo's
TidyBot meshes. That gap is deliberate (plan in one model, execute in the other) and is
bounded by `test_kitchen_pybullet_scene.py`: with the planning margin, a configuration
PyBullet clears is never penetrating in MuJoCo.
"""

from typing import Any, ClassVar

import numpy as np
import pybullet as p
from kinder_models.dynamic3d.utils import PyBulletSim
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.inverse_kinematics import set_robot_joints_with_held_object

# MuJoCo bodies mirrored into the planning scene. Fixtures never move after reset.
MIRRORED_FIXTURES = ("prep_table", "stovetop")
MIRRORED_MOVABLES = ("pitcher", "tray")
# A planned configuration keeps at least this far (m) from every obstacle, which
# covers the PyBullet/MuJoCo robot-model gap measured in the parity test.
PLANNING_MARGIN_M = 0.015
# A held jug of loose dumplings stays within this of vertical through every planned
# configuration. Provisional until the spill angle with three dumplings is measured.
UPRIGHT_TILT_LIMIT_DEG = 10.0


class KitchenPyBulletSim(PyBulletSim):
    """Upstream's planning scene plus mirrored MuJoCo geometry.

    A subclass names its own scene's bodies: `fixtures` never move after reset;
    `movables`, and every state object whose name starts with `movable_prefix`, take
    their pose from the state on each `set_state`.
    """

    fixtures: ClassVar[tuple[str, ...]] = MIRRORED_FIXTURES
    movables: ClassVar[tuple[str, ...]] = MIRRORED_MOVABLES
    movable_prefix: ClassVar[str] = "potsticker"

    def __init__(self, initial_state: Any, *, model: Any, data: Any, base_q: Any) -> None:  # noqa: PLR0917 -- upstream signature
        super().__init__(initial_state)
        # Upstream hard-codes the arm mount (0.12, 0, 0.4); the kitchen's TidyBot model
        # mounts it 5 mm differently. Read it from MuJoCo so every link agrees.
        self._base_to_arm_pose = self.arm_mount(model=model, data=data, base_q=base_q)
        self._mirrored: dict[str, int] = {}
        self._mirrored_counts: dict[str, int] = {}
        names = initial_state.get_object_names()
        prefix = self.movable_prefix
        movables = [*self.movables, *sorted(n for n in names if n.startswith(prefix))]
        for name in (*self.fixtures, *movables):
            self._mirror_body(name=name, model=model, data=data)
        for name in self.fixtures:
            body = model.body(name).id
            self.place(name=name, position=data.xpos[body], wxyz=data.xquat[body])
        self._movables = tuple(movables)

    @classmethod
    def from_controller(cls, *, controller: Any) -> "KitchenPyBulletSim":
        c = controller
        state = c.scene._get_object_centric_state()
        sim = cls(state, model=c.model, data=c.data, base_q=c.base_q)
        sim.set_state(state)
        return sim

    @staticmethod
    def arm_mount(
        *, model: Any, data: Any, base_q: Any, body: str = "robot_gen3/base_link"
    ) -> Pose:
        """The arm base's (or another robot `body`'s) pose in the mobile base's
        (x, y, yaw) frame."""
        from scipy.spatial.transform import Rotation

        x, y, yaw = (float(v) for v in data.qpos[base_q])
        arm = model.body(body).id
        to_base = Rotation.from_euler("z", yaw).inv()
        position = to_base.apply(data.xpos[arm] - np.array([x, y, 0.0]))
        rotation = to_base * Rotation.from_quat(data.xquat[arm][[1, 2, 3, 0]])
        return Pose(tuple(position.tolist()), tuple(rotation.as_quat().tolist()))

    def _mirror_body(self, *, name: str, model: Any, data: Any, subtree: bool = True) -> None:
        """Mirror the contact geoms of body `name` and, unless not `subtree`, of every
        body below it."""
        import mujoco

        root = model.body(name).id
        bodies = {root}
        for body in range(model.nbody if subtree else 0):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                bodies.add(body)
        root_position = data.xpos[root]
        root_rotation = data.xmat[root].reshape(3, 3)
        shapes, positions, orientations = [], [], []
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) not in bodies:
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            shapes.append(self._shape(model=model, geom=geom, mujoco=mujoco))
            # The geom's frame relative to the root body, from the reset kinematics.
            relative = root_rotation.T @ data.geom_xmat[geom].reshape(3, 3)
            positions.append(root_rotation.T @ (data.geom_xpos[geom] - root_position))
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, relative.flatten())
            orientations.append((quat[1], quat[2], quat[3], quat[0]))
        if not shapes:
            raise ValueError(f"MuJoCo body {name} has no collision geometry")
        client = self.physics_client_id
        count = len(shapes)
        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            linkMasses=[0.0] * count,
            linkCollisionShapeIndices=shapes,
            linkVisualShapeIndices=[-1] * count,
            linkPositions=[tuple(x) for x in positions],
            linkOrientations=orientations,
            linkInertialFramePositions=[(0, 0, 0)] * count,
            linkInertialFrameOrientations=[(0, 0, 0, 1)] * count,
            linkParentIndices=[0] * count,
            linkJointTypes=[p.JOINT_FIXED] * count,
            linkJointAxis=[(0, 0, 1)] * count,
            physicsClientId=client,
        )
        self._mirrored[name] = body
        self._mirrored_counts[name] = count

    def _shape(self, *, model: Any, geom: int, mujoco: Any) -> int:
        client = self.physics_client_id
        kind = int(model.geom_type[geom])
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return p.createCollisionShape(
                p.GEOM_BOX, halfExtents=size.tolist(), physicsClientId=client
            )
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return p.createCollisionShape(
                p.GEOM_SPHERE, radius=float(size[0]), physicsClientId=client
            )
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=float(size[0]),
                height=float(2 * size[1]),
                physicsClientId=client,
            )
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=float(size[0]),
                height=float(2 * size[1]),
                physicsClientId=client,
            )
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            # MuJoCo collides with the convex hull of these vertices, in the geom frame;
            # PyBullet builds the same hull from the same points.
            vertices = model.mesh_vert[start : start + count].tolist()
            return p.createCollisionShape(p.GEOM_MESH, vertices=vertices, physicsClientId=client)
        if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            # PyBullet has no ellipsoid: the hull of points on a sphere scaled to the
            # semi-axes, pushed out so the hull's facets circumscribe it.
            rings, segments = 9, 16
            polar = np.linspace(0, np.pi, rings)
            azimuth = np.linspace(0, 2 * np.pi, segments, endpoint=False)
            unit = np.array([
                (np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a))
                for a in polar
                for b in azimuth
            ])
            scale = 1.0 / (np.cos(np.pi / (rings - 1)) * np.cos(np.pi / segments))
            vertices = (unit * size[:3] * scale).tolist()
            return p.createCollisionShape(p.GEOM_MESH, vertices=vertices, physicsClientId=client)
        raise ValueError(f"Unsupported MuJoCo geom type {kind} for {model.geom(geom).name}")

    def mirrored_geom_count(self, *, name: str) -> int:
        return self._mirrored_counts[name]

    def place(self, *, name: str, position: Any, wxyz: Any) -> None:
        w, x, y, z = (float(v) for v in wxyz)
        px, py, pz = (float(v) for v in position)
        set_pose(
            self._mirrored[name],
            Pose((px, py, pz), (x, y, z, w)),
            self.physics_client_id,
        )

    def object_pose(self, *, name: str) -> tuple[np.ndarray, np.ndarray]:
        position, xyzw = p.getBasePositionAndOrientation(
            self._mirrored[name], physicsClientId=self.physics_client_id
        )
        return np.array(position), np.array(xyzw)

    def set_state(self, x: Any, held_object: Any = None) -> None:  # noqa: PLR0917 -- upstream signature
        """Upstream's update, then the mirrored movables; a held mirrored object rides
        with the hand by `base_link_to_held_obj`."""
        super().set_state(x, held_object=None)
        for name in self._movables:
            obj = x.get_object_from_name(name)
            position = tuple(x.get(obj, k) for k in ("x", "y", "z"))
            xyzw = tuple(x.get(obj, k) for k in ("qx", "qy", "qz", "qw"))
            set_pose(self._mirrored[name], Pose(position, xyzw), self.physics_client_id)
        if held_object is not None:
            set_robot_joints_with_held_object(
                self.robot,
                self.physics_client_id,
                self._mirrored[held_object.name],
                self.base_link_to_held_obj,
                self.robot.get_joint_positions(),
            )

    def mirrored_id(self, *, name: str) -> int:
        return self._mirrored[name]

    def named_collision_bodies(self) -> dict[str, int]:
        return {**super().named_collision_bodies(), **self._mirrored}

    def get_collision_bodies(self, held_object: int | None = None) -> set[int]:  # noqa: PLR0917 -- upstream signature
        bodies = super().get_collision_bodies(held_object=held_object)
        bodies.update(self._mirrored.values())
        if held_object is not None:
            bodies.discard(held_object)
        return bodies

    def distance_between(self, *, a: str, b: tuple[str, ...], cap: float) -> float:
        best = cap
        for name in b:
            for point in p.getClosestPoints(
                self._mirrored[a],
                self._mirrored[name],
                distance=cap,
                physicsClientId=self.physics_client_id,
            ):
                best = min(best, float(point[8]))
        return best

    def robot_clearance(self, *, base: Any, arm: Any, cap: float) -> float:
        """The robot's distance to every collision body, with its base at `base`
        (x, y, yaw) and its arm at `arm`, fingers open; capped at `cap`."""
        from spatialmath import SE2

        self.set_robot_base(SE2(float(base[0]), float(base[1]), float(base[2])))
        joints = list(self.robot.get_joint_positions())
        joints[:7] = [float(v) for v in arm]
        self.robot.set_joints(joints)
        best = cap
        for body in self.get_collision_bodies():
            for point in p.getClosestPoints(
                self.robot.robot_id, body, distance=cap, physicsClientId=self.physics_client_id
            ):
                best = min(best, float(point[8]))
        return best
