"""The rack's PyBullet planning scene against MuJoCo, and the planned scraper pick."""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from tests.environments.pouring3d.kitchen_clearance import Mujoco

pytestmark = pytest.mark.xdist_group("rack_pybullet_scene")

PLANNING_MARGIN_M = 0.015
SCENE = ("cupboard_1", "rack_0", "box_0", "wiper_0")


@pytest.fixture(scope="module")
def rack():
    from hitl_pmp.environments.racktoserving3d.planning import RackPyBulletSim

    env = RackServingEnvironment()
    env.reset_to_seed(seed=0)
    shelf = env.rack_controllers().shelf
    sim = RackPyBulletSim.from_shelf(shelf=shelf)
    yield env, shelf, sim
    sim.close()
    env.close()


def test_the_scene_is_mirrored_once_with_mujocos_poses(*, rack):
    """Every contact geom of each body, at its MuJoCo pose; upstream's hard-coded shelf,
    cube blocks and collider boxes are gone, so nothing is in the scene twice."""
    _, shelf, sim = rack
    m, d = shelf.model, shelf.data
    cubes = [f"cube_{i}" for i in range(5)]
    for name in (*SCENE, *cubes):
        assert sim.mirrored_geom_count(name=name) == len(Mujoco.geoms(model=m, names=[name]))
        position, _ = sim.object_pose(name=name)
        np.testing.assert_allclose(position, d.xpos[m.body(name).id], atol=1e-6)
    assert set(sim.named_collision_bodies().values()) == set(sim.get_collision_bodies())
    assert len(sim.get_collision_bodies()) == len(SCENE) + len(cubes)


def test_planning_with_the_margin_never_lets_mujoco_penetrate(*, rack):
    """Arm and gripper near the counter, rack, plate and scraper: wherever PyBullet
    reports the planning margin, MuJoCo reports no penetration."""
    _, shelf, sim = rack
    from hitl_pmp.environments.racktoserving3d.planning import ScraperMotionPlanner

    m = shelf.model
    d = mujoco.MjData(m)
    d.qpos[:] = shelf.data.qpos
    lower, upper = ScraperMotionPlanner.limits(shelf=shelf)
    arm = Mujoco.arm_geoms(model=m)
    obstacles = Mujoco.geoms(model=m, names=[*SCENE, *(f"cube_{i}" for i in range(5))])
    rng = np.random.default_rng(0)
    bases = [(0.55, -0.34, 0.0), (0.6, 0.0, 0.0), (0.45, -0.3, -0.4)]
    pairs = []
    for _ in range(400):
        base = bases[int(rng.integers(len(bases)))]
        q = rng.uniform(np.maximum(lower, -np.pi), np.minimum(upper, np.pi))
        d.qpos[shelf.ba], d.qpos[shelf.qa] = base, q
        mujoco.mj_kinematics(m, d)
        expected = Mujoco.distance(model=m, data=d, a=arm, b=obstacles, cap=0.1)
        actual = sim.robot_clearance(base=base, arm=q, cap=0.1)
        if min(expected, actual) < 0.1:
            pairs.append((actual, expected))
    pybullet, mujoco_ = np.array(pairs).T
    cleared = pybullet >= PLANNING_MARGIN_M
    assert cleared.sum() >= 30
    assert mujoco_[cleared].min() > 0, np.sort(mujoco_[cleared])[:5]


def test_pick_scraper_holds_the_scraper_from_the_counter():
    env = RackServingEnvironment()
    try:
        env.reset_to_seed(seed=0)
        env.take_action(action=np.array([0.0, 0, 0, 0, 0]))
        assert env.last_skill_error() is None
        assert env.backend().held(name="wiper_0")
    finally:
        env.close()


def test_solid_rack_mirrors_the_slab_and_feet_without_visual_wires(*, rack):
    import pybullet as p

    env, shelf, sim = rack
    model = shelf.model
    assert env.config.surface_kind == "wire_solid"
    wires = [g for g in range(model.ngeom) if model.geom(g).name.startswith("rack_0_wire_")]
    assert wires
    assert all(model.geom_contype[g] == model.geom_conaffinity[g] == 0 for g in wires)
    assert sim.mirrored_geom_count(name="rack_0") == 5
    client = sim.physics_client_id
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.001, physicsClientId=client)
    probe = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, physicsClientId=client)
    try:
        # This point is between the decorative wires, but still has a solid support.
        centre = shelf.data.xpos[model.body("rack_0").id]
        top = centre[2] + env.config.rack_height
        p.resetBasePositionAndOrientation(
            probe,
            (centre[0] + 0.007, centre[1] + 0.007, top + 0.0005),
            (0, 0, 0, 1),
            physicsClientId=client,
        )
        contacts = p.getClosestPoints(
            probe, sim.named_collision_bodies()["rack_0"], distance=0, physicsClientId=client
        )
        assert contacts and min(point[8] for point in contacts) < 0
        p.resetBasePositionAndOrientation(
            probe,
            (centre[0] + env.config.rack_depth / 2 + 0.005, centre[1], top),
            (0, 0, 0, 1),
            physicsClientId=client,
        )
        assert not p.getClosestPoints(
            probe, sim.named_collision_bodies()["rack_0"], distance=0, physicsClientId=client
        )
    finally:
        p.removeBody(probe, physicsClientId=client)
