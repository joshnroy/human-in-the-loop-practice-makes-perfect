"""The kitchen's PyBullet planning scene, and how far it can be trusted against MuJoCo."""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_pybullet_scene")

PLANNING_MARGIN_M = 0.015


@pytest.fixture(scope="module")
def kitchen():
    from hitl_pmp.environments.pouring3d.kitchen.planning.utils import KitchenPyBulletSim

    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    sim = KitchenPyBulletSim.from_controller(controller=c)
    yield env, c, sim
    sim.close()


class Mujoco:
    """Signed distances measured in a MuJoCo copy of the live model."""

    @staticmethod
    def subtree(*, model, root: int) -> set[int]:
        out = set()
        for body in range(model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                out.add(body)
        return out

    @staticmethod
    def geoms(*, model, names) -> list[int]:
        bodies = set().union(*(Mujoco.subtree(model=model, root=model.body(n).id) for n in names))
        return [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in bodies
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    @staticmethod
    def arm_geoms(*, model) -> list[int]:
        # The arm and gripper; the mobile base is planned as a 2-D footprint.
        return [
            g
            for g in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[g])).name or "").startswith("robot_")
            and model.body(int(model.geom_bodyid[g])).name != "robot_base_link"
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]

    @staticmethod
    def distance(*, model, data, a: list[int], b: list[int], cap: float) -> float:
        best = cap
        for ga in a:
            for gb in b:
                gap = np.linalg.norm(data.geom_xpos[ga] - data.geom_xpos[gb])
                if gap - model.geom_rbound[ga] - model.geom_rbound[gb] >= best:
                    continue
                best = min(best, Mujoco.pair(model=model, data=data, ga=ga, gb=gb, cap=best))
        return float(best)

    @staticmethod
    def pair(*, model, data, ga: int, gb: int, cap: float) -> float:
        """`mj_geomDistance`, checked against its own witness points.

        On MuJoCo 3.3.7 it sometimes returns 0 for a mesh pair several centimetres
        apart, with witness points nowhere near either geom (measured 2026-11-02:
        2293 such pair queries in 1500 sampled arm configurations). Those are
        recomputed from the vertices instead.
        """
        fromto = np.zeros(6)
        value = mujoco.mj_geomDistance(model, data, ga, gb, cap, fromto)
        if value >= cap or abs(np.linalg.norm(fromto[3:] - fromto[:3]) - abs(value)) <= 1e-4:
            return float(value)
        pa, pb = (
            Mujoco.points(model=model, data=data, g=ga),
            Mujoco.points(model=model, data=data, g=gb),
        )
        if pb is None or (
            pa is not None and int(model.geom_type[gb]) != mujoco.mjtGeom.mjGEOM_MESH
        ):
            return min(cap, Mujoco.to_primitive(model=model, data=data, g=gb, points=pa))
        if pa is None or int(model.geom_type[ga]) != mujoco.mjtGeom.mjGEOM_MESH:
            return min(cap, Mujoco.to_primitive(model=model, data=data, g=ga, points=pb))
        from scipy.spatial import cKDTree

        # Two meshes: the closest vertex pair, an upper bound within vertex spacing.
        return min(cap, float(cKDTree(pb).query(pa)[0].min()))

    @staticmethod
    def points(*, model, data, g: int) -> np.ndarray | None:
        rotation, origin = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[g])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            local = model.mesh_vert[start : start + count]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            local = signs * model.geom_size[g]
        else:
            return None
        return local @ rotation.T + origin

    @staticmethod
    def to_primitive(*, model, data, g: int, points: np.ndarray) -> float:
        """Exact distance from points to a box, cylinder, capsule or sphere (0 inside)."""
        local = (points - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            outside = np.maximum(np.abs(local) - size, 0)
            return float(np.linalg.norm(outside, axis=1).min())
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radial = np.maximum(np.linalg.norm(local[:, :2], axis=1) - size[0], 0)
            axial = np.maximum(np.abs(local[:, 2]) - size[1], 0)
            return float(np.hypot(radial, axial).min())
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axial = np.clip(local[:, 2], -size[1], size[1])
            offset = local - np.c_[np.zeros((len(local), 2)), axial]
            return float(np.maximum(np.linalg.norm(offset, axis=1) - size[0], 0).min())
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(np.maximum(np.linalg.norm(local, axis=1) - size[0], 0).min())
        raise ValueError(f"No fallback distance for geom type {kind}")


OBSTACLES = ("prep_table", "stovetop", "pitcher", "tray", "potsticker_0")


def test_every_collision_geom_of_the_scene_is_mirrored(*, kitchen):
    _, c, sim = kitchen
    for name in OBSTACLES:
        assert sim.mirrored_geom_count(name=name) == len(Mujoco.geoms(model=c.model, names=[name]))


def test_the_planning_robot_has_mujocos_kinematics(*, kitchen):
    """Same joint angles, same link frames: the arm mount is read from MuJoCo."""
    import pybullet as p
    from spatialmath import SE2

    _, c, sim = kitchen
    m = c.model
    d = mujoco.MjData(m)
    d.qpos[:] = c.data.qpos
    client = sim.physics_client_id
    joints = p.getNumJoints(sim.robot.robot_id, physicsClientId=client)
    link = {
        p.getJointInfo(sim.robot.robot_id, j, physicsClientId=client)[12].decode(): j
        for j in range(joints)
    }
    rng = np.random.default_rng(0)
    for _ in range(20):
        base = (*rng.uniform([-0.5, -1.2], [0.5, 1.0]), rng.uniform(-np.pi, np.pi))
        q = rng.uniform(np.maximum(c.lower, -np.pi), np.minimum(c.upper, np.pi))
        d.qpos[c.base_q], d.qpos[c.arm_q] = base, q
        mujoco.mj_kinematics(m, d)
        sim.set_robot_base(SE2(*base))
        sim.robot.set_joints([*q, *sim.robot.get_joint_positions()[7:]])
        for name in ("shoulder_link", "forearm_link", "bracelet_link"):
            pb = p.getLinkState(sim.robot.robot_id, link[name], physicsClientId=client)[4]
            np.testing.assert_allclose(pb, d.xpos[m.body(f"robot_{name}").id], atol=1e-4)
        np.testing.assert_allclose(sim.get_ee_pose().position, d.site_xpos[c.site], atol=1e-4)


def test_mirrored_objects_take_their_mujoco_poses(*, kitchen):
    _, c, sim = kitchen
    sim.set_state(c.scene._get_object_centric_state())
    for name in ("pitcher", "tray", "potsticker_0", "potsticker_1", "potsticker_2"):
        position, quat = sim.object_pose(name=name)
        body = c.model.body(name).id
        np.testing.assert_allclose(position, c.data.xpos[body], atol=1e-6)
        mujoco_xyzw = c.data.xquat[body][[1, 2, 3, 0]]
        assert abs(abs(float(np.dot(quat, mujoco_xyzw))) - 1) < 1e-6


def test_the_mirror_reproduces_mujoco_distances_between_objects(*, kitchen):
    """Jug moved to random poses near the counter and stove: jug-to-fixture distances
    agree to a millimetre, so the mirrored geometry (meshes, frames) is the same."""
    _, c, sim = kitchen
    m = c.model
    d = mujoco.MjData(m)
    d.qpos[:] = c.data.qpos
    rng = np.random.default_rng(0)
    jug = Mujoco.geoms(model=m, names=["pitcher"])
    fixtures = Mujoco.geoms(model=m, names=["prep_table", "stovetop"])
    joint = int(m.jnt_qposadr[m.body("pitcher").jntadr[0]])
    compared = 0
    for _ in range(300):
        anchor = np.array([[0.62, -1.05, 0.80], [0.62, 0.85, 0.80]][int(rng.integers(2))])
        d.qpos[joint : joint + 3] = anchor + rng.uniform([-0.12, -0.3, -0.04], [0.06, 0.3, 0.12])
        quat = rng.normal(size=4)
        d.qpos[joint + 3 : joint + 7] = quat / np.linalg.norm(quat)
        mujoco.mj_kinematics(m, d)
        expected = Mujoco.distance(model=m, data=d, a=jug, b=fixtures, cap=0.1)
        if expected >= 0.1 or expected <= 0:
            continue
        sim.place(
            name="pitcher",
            position=d.xpos[m.body("pitcher").id],
            wxyz=d.xquat[m.body("pitcher").id],
        )
        actual = sim.distance_between(a="pitcher", b=("prep_table", "stovetop"), cap=0.1)
        # Bullet keeps a small collision margin on convex hulls; about a millimetre.
        assert actual == pytest.approx(expected, abs=2e-3)
        compared += 1
    assert compared >= 20


def test_planning_with_the_margin_never_lets_mujoco_penetrate(*, kitchen):
    """Robot arm and gripper near every obstacle, at sampled base poses and arm
    configurations: wherever PyBullet reports at least the planning margin, MuJoCo
    reports no penetration; and out of contact, PyBullet is never more optimistic than
    MuJoCo by the margin. (Penetration depths are not compared: both models call those
    collisions, and they differ by centimetres where the two bracelet/coupling meshes
    differ, measured 2026-11-02.)"""
    _, c, sim = kitchen
    m = c.model
    d = mujoco.MjData(m)
    d.qpos[:] = c.data.qpos
    mujoco.mj_kinematics(m, d)
    sim.set_state(c.scene._get_object_centric_state())
    arm = Mujoco.arm_geoms(model=m)
    obstacles = Mujoco.geoms(model=m, names=list(OBSTACLES) + ["potsticker_1", "potsticker_2"])
    rng = np.random.default_rng(0)
    bases = [(-0.05, -1.05, 0.0), (0.0, -1.05, 0.0), (0.25, 0.85, 0.0), (0.3, 0.6, 0.4)]
    pairs = []
    for _ in range(1200):
        base = bases[int(rng.integers(len(bases)))]
        q = rng.uniform(np.maximum(c.lower, -np.pi), np.minimum(c.upper, np.pi))
        d.qpos[c.base_q] = base
        d.qpos[c.arm_q] = q
        mujoco.mj_kinematics(m, d)
        expected = Mujoco.distance(model=m, data=d, a=arm, b=obstacles, cap=0.1)
        actual = sim.robot_clearance(base=base, arm=q, cap=0.1)
        if min(expected, actual) < 0.1:
            pairs.append((actual, expected))
    pybullet, mujoco_ = np.array(pairs).T
    assert len(pairs) >= 100
    cleared = pybullet >= PLANNING_MARGIN_M
    assert cleared.sum() >= 50
    assert mujoco_[cleared].min() > 0, np.sort(mujoco_[cleared])[:5]
    apart = (pybullet > 0) & (mujoco_ > 0)
    gaps = pybullet[apart] - mujoco_[apart]
    assert gaps.max() < PLANNING_MARGIN_M, np.sort(gaps)[-5:]
