"""The planning scene's two exact speed-ups answer every collision query as before.

PyBullet's `performCollisionDetection`, which every upstream check calls, recomputed
all of the scene's resting contacts (the dumplings on the rack: 150 points) for every
configuration, although no planner reads a cached contact: every query is
`getClosestPoints`, which ignores collision filters. And each base-path check swept
every obstacle, though the robot is near one or two of them.
"""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose
from pybullet_helpers.inverse_kinematics import check_collisions_with_held_object
from spatialmath import SE2

from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment

pytestmark = pytest.mark.xdist_group("rack_pybullet_scene")

MARGIN_M = 0.015


@pytest.fixture(scope="module")
def rack():
    from hitl_pmp.environments.racktoserving3d.planning import RackPyBulletSim

    env = RackServingEnvironment()
    env.reset_to_seed(seed=0)
    shelf = env.rack_controllers().shelf
    sim = RackPyBulletSim.from_shelf(shelf=shelf)
    yield sim
    sim.close()
    env.close()


def configurations(*, sim, count: int, seed: int):
    rng = np.random.default_rng(seed)
    joints = list(sim.robot.get_joint_positions())
    out = []
    for _ in range(count):
        arm = np.array(joints)
        if rng.random() < 0.5:
            arm[:7] = rng.uniform(-2.5, 2.5, 7)
        base = SE2(rng.uniform(0.2, 1.4), rng.uniform(-0.8, 0.8), rng.uniform(-np.pi, np.pi))
        out.append((base, list(arm)))
    return out


def test_the_planning_scene_caches_no_contact_pairs(*, rack):
    sim = rack
    p.performCollisionDetection(physicsClientId=sim.physics_client_id)
    assert p.getContactPoints(physicsClientId=sim.physics_client_id) == ()


@pytest.mark.parametrize("holding", [False, True], ids=["empty", "holding"])
def test_a_base_check_answers_as_the_full_obstacle_sweep(*, rack, holding):
    """At random bases and arms, against the upstream check over every obstacle."""
    sim = rack
    client = sim.physics_client_id
    held = sim.mirrored_id(name="cube_0") if holding else None
    sim.base_link_to_held_obj = Pose((0.0, 0.0, 0.1), (0.0, 0.0, 0.0, 1.0))
    offset = sim.base_link_to_held_obj if holding else None
    bodies = sim.get_collision_bodies(held_object=held)
    collided = 0
    for base, arm in configurations(sim=sim, count=600, seed=int(holding)):
        fast = sim.robot_collides_at_base(
            base, arm, held_object=held, collision_bodies=bodies, distance_threshold=MARGIN_M
        )
        sim.set_robot_base(base)
        full = check_collisions_with_held_object(
            sim.robot, bodies, client, held, offset, arm, distance_threshold=MARGIN_M
        )
        assert fast == full
        collided += int(full)
    # Both answers occur, so agreement is not vacuous.
    assert 100 < collided < 500


def test_a_base_check_skips_obstacles_far_from_the_robot(*, rack):
    sim = rack
    sim.set_robot_base(SE2(0.3, 0.0, np.pi))
    joints = list(sim.robot.get_joint_positions())
    sim.robot.set_joints(joints)
    near = sim.near_bodies(bodies=sim.get_collision_bodies(), held_object=None, margin=MARGIN_M)
    assert len(near) < len(sim.get_collision_bodies())
