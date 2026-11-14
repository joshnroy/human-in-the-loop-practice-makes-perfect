"""`IKFastClosest.solve` returns exactly what upstream's closest IKFast solve does."""

import numpy as np
import pybullet as p
import pytest
from pybullet_helpers.geometry import Pose
from pybullet_helpers.ikfast.utils import IKFastHyperparameters, ikfast_closest_inverse_kinematics

from hitl_pmp.environments.pouring3d.kitchen.planning.ikfast_closest import IKFastClosest
from hitl_pmp.environments.vision_bracelet_robot import VisionBraceletKinovaGen3


@pytest.fixture(scope="module")
def robot():
    client = p.connect(p.DIRECT)
    yield VisionBraceletKinovaGen3(client, fixed_base=False, control_mode="reset")
    p.disconnect(physicsClientId=client)


def targets(*, robot, count: int, seed: int):
    """Hand poses the arm reaches (forward kinematics of random joints), some moved
    out of reach, each with the arm left at another random configuration."""
    rng = np.random.default_rng(seed)
    lower = np.maximum(robot.joint_lower_limits[:7], -np.pi)
    upper = np.minimum(robot.joint_upper_limits[:7], np.pi)
    fingers = list(robot.get_joint_positions())[7:]
    out = []
    for index in range(count):
        robot.set_joints([*rng.uniform(lower, upper).tolist(), *fingers])
        pose = robot.get_end_effector_pose()
        position = np.array(pose.position)
        if index % 4 == 3:
            position = position + rng.normal(0.0, 0.3, 3)
        start = [*rng.uniform(lower, upper).tolist(), *fingers]
        out.append((Pose(tuple(position.tolist()), pose.orientation), start))
    return out


@pytest.mark.parametrize("attempts", [1500, 300, 40])
def test_the_closest_solutions_are_upstreams_value_for_value(*, robot, attempts):
    hyperparameters = IKFastHyperparameters(max_time=np.inf, max_attempts=attempts)
    found = 0
    for target, start in targets(robot=robot, count=40, seed=attempts):
        robot.set_joints(start)
        expected = ikfast_closest_inverse_kinematics(
            robot, world_from_target=target, hyperparameters=hyperparameters
        )
        robot.set_joints(start)
        actual = IKFastClosest.solve(
            robot=robot, world_from_target=target, hyperparameters=hyperparameters
        )
        assert actual == expected
        found += int(bool(expected))
    # Both solvable and unsolvable targets occur, so agreement is not vacuous.
    assert 10 < found < 40


def test_a_time_limit_or_distance_limit_is_left_to_upstream(*, robot, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "hitl_pmp.environments.pouring3d.kitchen.planning.ikfast_closest."
        "ikfast_closest_inverse_kinematics",
        lambda *args, **kwargs: calls.append(kwargs) or [],
    )
    target, start = targets(robot=robot, count=1, seed=0)[0]
    robot.set_joints(start)
    for hyperparameters in (
        IKFastHyperparameters(max_time=0.05, max_attempts=10),
        IKFastHyperparameters(max_time=np.inf, max_attempts=10, max_distance=1.0),
    ):
        IKFastClosest.solve(robot=robot, world_from_target=target, hyperparameters=hyperparameters)
    assert len(calls) == 2


def test_the_kitchen_goal_solves_through_it(*, robot, monkeypatch):
    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
    )

    solved = []
    solve = IKFastClosest.solve
    monkeypatch.setattr(
        IKFastClosest, "solve", staticmethod(lambda **kwargs: solved.append(1) or solve(**kwargs))
    )
    target, start = targets(robot=robot, count=1, seed=1)[0]
    robot.set_joints(start)

    class Sim:
        pass

    sim = Sim()
    sim.robot = robot
    KitchenMotionPlanner.goal(
        sim=sim,
        position=np.array(target.position),
        rotation=Rotation.from_quat(target.orientation),
        valid=lambda q: False,
    )
    assert solved == [1]


def test_batched_free_joint_samples_are_upstreams_stream_through_shuffles(*, robot):
    """Interleaved with shuffles that draw from the same generator, as the solve does."""
    from pybullet_helpers.ikfast.utils import free_joints_generator, get_ikfast_joints

    from hitl_pmp.environments.pouring3d.kitchen.planning.ikfast_closest import (
        FreeJointSamples,
    )

    _, free = get_ikfast_joints(robot)
    upstream_rng, batched_rng = np.random.default_rng(3), np.random.default_rng(3)
    upstream = free_joints_generator(robot, free, np.inf, upstream_rng)
    sampler = FreeJointSamples(
        robot=robot, free_joint_infos=free, max_distance=np.inf, rng=batched_rng
    )
    batched = sampler.samples()
    for index in range(2000):
        assert list(next(batched)) == list(next(upstream))
        if index % 97 in (5, 6, 50):
            found_up, found_batched = list(range(index % 7 + 2)), list(range(index % 7 + 2))
            sampler.sync()
            upstream_rng.shuffle(found_up)
            batched_rng.shuffle(found_batched)
            assert found_batched == found_up
