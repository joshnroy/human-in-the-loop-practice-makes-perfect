"""A closed planning scene is rebuilt in place for the next plan, robot kept.

Every plan built a fresh PyBullet client, and loading the Kinova URDF (twice: upstream's
stock robot, then the vision-bracelet one that replaces it) was 1.5 s of each build, 13
builds in one robot batch return. A reused scene must be the scene a fresh build makes:
the same bodies, shapes and poses, the same robot state, and the same answer, to the
bit, to every collision query.
"""

import numpy as np
import pybullet as p
import pytest
from spatialmath import SE2

from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.planning import RackPyBulletSim

pytestmark = pytest.mark.xdist_group("rack_pybullet_scene")

MARGIN_M = 0.015


@pytest.fixture(scope="module")
def env():
    env = RackServingEnvironment()
    env.reset_to_seed(seed=0)
    yield env
    env.close()


@pytest.fixture(autouse=True)
def no_idle_scenes():
    """Each test starts with no kept scene, as a fresh process does."""
    idle = RackPyBulletSim._idle  # noqa: SLF001
    while idle:
        idle.pop()._finalizer()  # noqa: SLF001 -- upstream's own disconnect
    yield


@pytest.fixture
def no_reuse():
    reuse = RackPyBulletSim.reuse
    RackPyBulletSim.reuse = False
    yield
    RackPyBulletSim.reuse = reuse


class Scene:
    """A static-method container: what a planning scene is, ids resolved to names."""

    @staticmethod
    def body(*, sim, body: int):
        client = sim.physics_client_id
        pose = p.getBasePositionAndOrientation(body, physicsClientId=client)
        links = range(-1, p.getNumJoints(body, physicsClientId=client))
        # Each shape's own body id dropped: ids are the one thing a rebuild may change.
        shapes = [
            [shape[1:] for shape in p.getCollisionShapeData(body, link, physicsClientId=client)]
            for link in links
        ]
        states = [
            p.getLinkState(body, link, physicsClientId=client)[:2]
            for link in range(p.getNumJoints(body, physicsClientId=client))
        ]
        return pose, shapes, states

    @staticmethod
    def signature(*, sim):
        client = sim.physics_client_id
        named = sim.named_collision_bodies()
        name_of = {body: name for name, body in named.items()}
        robot = sim.robot.robot_id
        joints = [
            p.getJointState(robot, j, physicsClientId=client)
            for j in range(p.getNumJoints(robot, physicsClientId=client))
        ]
        return {
            "attributes": sorted(set(vars(sim)) - {"_finalizer"}),
            "bodies": p.getNumBodies(physicsClientId=client),
            "named": {name: Scene.body(sim=sim, body=body) for name, body in sorted(named.items())},
            "collision": sorted(name_of[body] for body in sim.get_collision_bodies()),
            "cubes": {name: name_of[body] for name, body in sim._cubes.items()},  # noqa: SLF001
            "shelf": name_of.get(sim._cupboard1_shelf_id),  # noqa: SLF001
            "static": dict(sim._static_colliders),  # noqa: SLF001
            "chassis": Scene.body(sim=sim, body=sim._chassis),  # noqa: SLF001
            "chassis_links": sim._chassis_links,  # noqa: SLF001
            "mounts": (sim._base_to_arm_pose, sim._base_to_chassis),  # noqa: SLF001
            "counts": dict(sim._mirrored_counts),  # noqa: SLF001
            "movables": sim._movables,  # noqa: SLF001
            "held": sim.base_link_to_held_obj,
            "ignores_food": sim.ignores_food,
            "robot": (Scene.body(sim=sim, body=robot), joints),
        }

    @staticmethod
    def answers(*, sim, seed: int) -> list:
        """Collision answers and exact closest distances at random robot poses."""
        rng = np.random.default_rng(seed)
        client = sim.physics_client_id
        joints = list(sim.robot.get_joint_positions())
        named = sim.named_collision_bodies()
        bodies = [named[name] for name in sorted(named)]
        out = []
        for _ in range(150):
            arm = list(joints)
            arm[:7] = rng.uniform(-2.5, 2.5, 7).tolist()
            base = SE2(rng.uniform(0.2, 1.4), rng.uniform(-0.8, 0.8), rng.uniform(-np.pi, np.pi))
            collides = sim.robot_collides_at_base(
                base, arm, collision_bodies=set(bodies), distance_threshold=MARGIN_M
            )
            distances = [
                sorted(
                    point[8]
                    for point in p.getClosestPoints(
                        sim.robot.robot_id, body, distance=0.1, physicsClientId=client
                    )
                )
                for body in bodies
            ]
            out.append((collides, distances, sim.chassis_clear(margin=MARGIN_M)))
        return out


def test_a_closed_scene_is_reused_by_the_next_plan(*, env):
    shelf = env.rack_controllers().shelf
    first = RackPyBulletSim.from_shelf(shelf=shelf)
    client = first.physics_client_id
    first.close()
    second = RackPyBulletSim.from_shelf(shelf=shelf)
    try:
        assert second is first
        assert second.physics_client_id == client
    finally:
        second.close()


def test_a_scene_in_use_is_never_handed_out_again(*, env):
    shelf = env.rack_controllers().shelf
    outer = RackPyBulletSim.from_shelf(shelf=shelf)
    try:
        inner = RackPyBulletSim.from_shelf(shelf=shelf)
        inner.close()
        inner.close()
        assert inner is not outer
        again = RackPyBulletSim.from_shelf(shelf=shelf)
        other = RackPyBulletSim.from_shelf(shelf=shelf)
        try:
            assert again is inner
            assert other is not inner and other is not outer
        finally:
            other.close()
            again.close()
    finally:
        outer.close()


def test_a_reused_scene_is_the_scene_a_fresh_build_makes(*, env):
    """Used, posed, given a stray body and closed at one state, then rebuilt at
    another: against a fresh build at that state, nothing differs."""
    shelf = env.rack_controllers().shelf
    used = RackPyBulletSim.from_shelf(shelf=shelf, ignores_food=True)
    used.set_robot_base(SE2(0.5, 0.1, 0.3))
    used.base_link_to_held_obj = "stale"
    used.stray_attribute = 1
    shape = p.createCollisionShape(
        p.GEOM_SPHERE, radius=0.01, physicsClientId=used.physics_client_id
    )
    p.createMultiBody(baseCollisionShapeIndex=shape, physicsClientId=used.physics_client_id)
    used.close()
    # Another state: the robot holding the scraper somewhere else on the counter.
    env.take_action(action=np.array([0.0, 0, 0, 0, 0]))
    assert env.last_skill_error() is None
    reused = RackPyBulletSim.from_shelf(shelf=shelf)
    assert reused is used
    RackPyBulletSim.reuse = False
    try:
        fresh = RackPyBulletSim.from_shelf(shelf=shelf)
    finally:
        RackPyBulletSim.reuse = True
    try:
        assert fresh is not reused
        assert Scene.signature(sim=reused) == Scene.signature(sim=fresh)
        assert Scene.answers(sim=reused, seed=0) == Scene.answers(sim=fresh, seed=0)
    finally:
        fresh.close()
        reused.close()


def test_without_reuse_a_closed_scene_is_disconnected(*, env, no_reuse):
    sim = RackPyBulletSim.from_shelf(shelf=env.rack_controllers().shelf)
    client = sim.physics_client_id
    sim.close()
    assert not p.isConnected(physicsClientId=client)


def test_a_scene_is_disconnected_once_restaged_its_limit(*, env, monkeypatch):
    """PyBullet keeps the shapes of removed bodies, so a kept scene grows with every
    restage; it is rebuilt fresh after `MAX_RESTAGES`."""
    from hitl_pmp.environments.racktoserving3d import planning

    monkeypatch.setattr(planning, "MAX_RESTAGES", 2)
    shelf = env.rack_controllers().shelf
    first = RackPyBulletSim.from_shelf(shelf=shelf)
    first.close()
    for _ in range(2):
        again = RackPyBulletSim.from_shelf(shelf=shelf)
        assert again is first
        again.close()
    assert not p.isConnected(physicsClientId=first.physics_client_id)
    fresh = RackPyBulletSim.from_shelf(shelf=shelf)
    try:
        assert fresh is not first
    finally:
        fresh.close()
