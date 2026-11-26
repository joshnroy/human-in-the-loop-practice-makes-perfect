"""The toy's human reset returns the food and homes the scraper; it never moves the robot.

Josh, 2026-11-25: the person resets the scene, not the robot. Getting the robot back to
where its skills work is robot_reset's job (`RackRobotBatchReturn`), so a robot left
anywhere, even stranded, stays exactly there through a human reset.
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


def test_human_reset_leaves_a_stranded_robot_exactly_where_it_is():
    from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
    from hitl_pmp.environments.racktoserving3d.robot_batch_return import RackRobotBatchReturn
    from hitl_pmp.environments.racktoserving3d.toy_interface import HUMAN_RESET

    env = _toy_env()
    try:
        backend, shelf = env.backend(), env.rack_controllers().shelf
        data = backend.data()
        # Strand the robot where the aborted robot_reset left it (B=120 ees05 seed 0),
        # with one continuous arm joint wound a full turn.
        stranded = np.array([0.701, -0.179, 0.124])
        data.qpos[shelf.ba] = stranded
        data.qpos[shelf.qa[0]] += 2 * np.pi
        shelf.base_target = stranded.copy()
        shelf.arm_reference = data.qpos[shelf.qa].copy()
        shelf.sim.forward()
        shelf.hold(count=5, label="stranded")
        before = backend.robot_configuration()
        env._toy_flags = frozenset({"NeedsReset", "AfterSweep3"})
        assert env.reset_movables(destination=HUMAN_RESET)
        np.testing.assert_allclose(backend.robot_configuration()[:10], before[:10], atol=2e-3)
        np.testing.assert_allclose(shelf.base_target, stranded, atol=1e-9)
        assert env.toy_flags() == frozenset({"FullRack"})
        batch = RackRobotBatchReturn(controllers=env.rack_controllers())
        assert all(batch.in_place(name=name) for name in backend.cube_names)
        assert not backend.held(name=Objects.scraper.name)
    finally:
        env.close()


def test_human_reset_has_no_robot_placement_of_its_own():
    from hitl_pmp.environments.racktoserving3d.backend import RackServingBackend
    from hitl_pmp.environments.racktoserving3d.toy_environment import (
        RackServingToyEnvironment,
    )

    assert not hasattr(RackServingBackend, "place_robot")
    assert not hasattr(RackServingToyEnvironment, "_return_robot_to_start")
