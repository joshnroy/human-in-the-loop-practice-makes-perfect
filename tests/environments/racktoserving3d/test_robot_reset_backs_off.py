"""robot_reset leaves the robot where its skills work, whether it succeeds or aborts.

B=120 ees05 seed 0 (2026-11-23): a robot_reset aborted mid-carry and stopped in place,
the hand out over the counter beside the plate. No base path out of there kept the
carried arm 15 mm from the plate, so every later PickScraper refused (about 75 min
each) until the run stalled. And after a robot_reset that succeeds, the arm it parks
with seeds PickScraper's hover IK onto another branch, and gather(0.05, 0.015) failed
5/5 starts that succeeded 5/5 from the episode-start arm (2026-11-25).
"""

import argparse
import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)

# Where the aborted robot_reset left the robot, read off the recording's last tick:
# base (x, y, yaw), then the seven arm joints (two continuous ones wound many turns).
STRANDED_BASE = np.array([0.7011175, -0.1794553, 0.1235147])
STRANDED_ARM = np.array([
    11.787767,
    0.265377,
    -96.21010,
    -1.057221,
    12.159588,
    -1.463130,
    -54.716045,
])


def _toy_problem():
    from hitl_pmp.environments.racktoserving3d.cli import RackServingCli

    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
    args = parser.parse_args([])
    args.seed = 0
    problem = RackServingCli.build_problem(args=args, mode="training")
    problem.hard_reset()
    return problem


def _robot_reset(*, problem):
    from hitl_pmp.environments.racktoserving3d.skill_provider import RackServingActions

    env = problem.env
    env._toy_flags = frozenset({"NeedsReset", "AfterGather"})
    problem.take_action(
        action=RackServingActions.encode(name="robot_reset", params=np.array([]), bounds=())
    )
    return env.last_skill_error()


def _displace_one_dumpling(*, env, name: str = "cube_2", shift: float = 0.04) -> None:
    import mujoco

    backend = env.backend()
    model, data = backend.model(), backend.data()
    address = model.jnt_qposadr[model.joint(f"{name}_joint").id]
    data.qpos[address] += shift
    mujoco.mj_forward(model, data)
    env.rack_controllers().shelf.hold(count=60, label="displace_settle")


def _strand(*, env) -> None:
    shelf = env.rack_controllers().shelf
    data = shelf.data
    data.qpos[shelf.ba] = STRANDED_BASE
    data.qpos[shelf.qa] = STRANDED_ARM
    data.qvel[:] = 0.0
    shelf.base_target = STRANDED_BASE.copy()
    shelf.arm_reference = STRANDED_ARM.copy()
    shelf.sim.forward()
    shelf.hold(count=5, label="stranded")


def _wrapped(*, arm: np.ndarray) -> np.ndarray:
    return (np.asarray(arm, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def _assert_parked(*, env) -> None:
    from hitl_pmp.environments.racktoserving3d.controllers import RackControllers

    shelf, config = env.rack_controllers().shelf, env.config
    np.testing.assert_allclose(
        _wrapped(arm=shelf.data.qpos[shelf.qa] - shelf.home_arm), np.zeros(7), atol=0.05
    )
    stances = RackControllers.stances
    standard = [
        stances.rear_base(y=config.rack_center_y, config=config),
        stances.side_base(side=-1, config=config),
        stances.side_base(side=1, config=config),
    ]
    base = shelf.data.qpos[shelf.ba]
    assert (
        min(
            max(np.linalg.norm(base[:2] - s[:2]), abs(_wrapped(arm=base[2:] - s[2:])[0]))
            for s in standard
        )
        < 0.02
    ), base


def test_a_robot_reset_that_succeeds_ends_with_the_arm_at_its_episode_start_pose():
    problem = _toy_problem()
    env = problem.env
    try:
        _displace_one_dumpling(env=env)
        assert _robot_reset(problem=problem) is None
        shelf = env.rack_controllers().shelf
        np.testing.assert_allclose(
            _wrapped(arm=shelf.data.qpos[shelf.qa] - shelf.home_arm), np.zeros(7), atol=0.05
        )
    finally:
        env.close()


def test_an_aborted_robot_reset_backs_off_to_a_standard_stance_from_which_the_scraper_is_picked(
    *, monkeypatch
):
    from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
    from hitl_pmp.environments.racktoserving3d.robot_batch_return import RackRobotBatchReturn

    problem = _toy_problem()
    env = problem.env
    try:
        _displace_one_dumpling(env=env)
        _strand(env=env)

        def refused(self, *, name, slot):  # noqa: ARG001, PLR0917 -- a method's self
            raise SkillFailed("No safe food placement: refused for this test")

        monkeypatch.setattr(RackRobotBatchReturn, "put_back", refused)
        error = _robot_reset(problem=problem)
        assert error is not None and "refused for this test" in error
        _assert_parked(env=env)
        controllers = env.rack_controllers()
        controllers.pick_scraper()
        assert env.backend().held(name=Objects.scraper.name)
    finally:
        env.close()


def test_a_robot_reset_aborted_with_food_in_the_hand_backs_off_still_holding_it(*, monkeypatch):
    from hitl_pmp.environments.racktoserving3d.controllers import RackControllers
    from hitl_pmp.environments.racktoserving3d.food import WeldedFoodSkills

    problem = _toy_problem()
    env = problem.env
    try:
        _displace_one_dumpling(env=env)

        def refused(*_args, **_kwargs):
            raise SkillFailed("food_lift: refused for this test")

        monkeypatch.setattr(WeldedFoodSkills, "place", refused)
        error = _robot_reset(problem=problem)
        assert error is not None and "refused for this test" in error
        backend, shelf, config = env.backend(), env.rack_controllers().shelf, env.config
        # Nothing hidden: the dumpling it was carrying is still in the hand.
        assert any(backend.held(name=name) for name in backend.cube_names)
        base = shelf.data.qpos[shelf.ba]
        stances = RackControllers.stances
        standard = [
            stances.rear_base(y=config.rack_center_y, config=config),
            stances.side_base(side=-1, config=config),
            stances.side_base(side=1, config=config),
        ]
        assert min(np.linalg.norm(base[:2] - s[:2]) for s in standard) < 0.02, base
        monkeypatch.undo()
        assert _robot_reset(problem=problem) is None
    finally:
        env.close()
