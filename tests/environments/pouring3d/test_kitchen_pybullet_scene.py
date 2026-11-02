"""The kitchen's PyBullet planning scene, and how far it can be trusted against MuJoCo."""

import mujoco
import numpy as np
import pytest
from kitchen_clearance import Mujoco

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
