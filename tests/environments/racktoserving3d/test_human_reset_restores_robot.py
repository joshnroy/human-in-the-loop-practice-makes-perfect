"""The toy's human reset puts the robot back at its episode-start pose, base and arm.

B=120 ees05 seed 0 (2026-11-23): a robot_reset aborted mid-carry and left the base at
(0.701, -0.179, 0.124), away from both of its working stances. `reset_movables` homed
the scraper and returned the food but left the robot bit-identical, so every later
gather failed at PickScraper ("no stance reaches the scraper") until the run stalled.
"""

import argparse
import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


def _toy_env():
    from hitl_pmp.environments.racktoserving3d.cli import RackServingCli

    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    problem = RackServingCli.build_problem(args=args, mode="training")
    problem.hard_reset()
    return problem.env


def _robot_qpos(env) -> np.ndarray:
    backend = env.backend()
    model, data = backend.model(), backend.data()
    return np.array([
        data.qpos[model.jnt_qposadr[model.joint(joint).id]] for joint in backend.ROBOT_JOINTS
    ])


def test_human_reset_returns_a_stranded_robot_to_its_episode_start_pose():
    from hitl_pmp.environments.racktoserving3d.toy_interface import HUMAN_RESET

    env = _toy_env()
    try:
        start = _robot_qpos(env)
        backend, shelf = env.backend(), env.rack_controllers().shelf
        data = backend.data()
        # Strand the robot as the aborted robot_reset did, and wind one continuous
        # arm joint a full turn, as long practice leaves it.
        stranded = np.array([0.701, -0.179, 0.124])
        data.qpos[shelf.ba] = stranded
        data.qpos[shelf.qa[0]] += 2 * np.pi
        shelf.base_target = stranded.copy()
        shelf.arm_reference = data.qpos[shelf.qa].copy()
        shelf.sim.forward()
        shelf.hold(count=5, label="stranded")
        assert np.linalg.norm(_robot_qpos(env)[:2] - start[:2]) > 0.5
        env._toy_flags = frozenset({"NeedsReset", "AfterSweep3"})
        assert env.reset_movables(destination=HUMAN_RESET)
        after = _robot_qpos(env)
        np.testing.assert_allclose(after[:3], start[:3], atol=2e-3)
        np.testing.assert_allclose(after[3:10], start[3:10], atol=2e-2)
        np.testing.assert_allclose(shelf.base_target, start[:3], atol=1e-9)
        assert env.toy_flags() == frozenset({"FullRack"})
    finally:
        env.close()


def test_human_reset_leaves_a_robot_at_its_start_pose_where_it_is():
    from hitl_pmp.environments.racktoserving3d.toy_interface import HUMAN_RESET

    env = _toy_env()
    try:
        start = _robot_qpos(env)
        env._toy_flags = frozenset({"NeedsReset", "AfterSweep3"})
        assert env.reset_movables(destination=HUMAN_RESET)
        np.testing.assert_allclose(_robot_qpos(env)[:10], start[:10], atol=2e-3)
    finally:
        env.close()
