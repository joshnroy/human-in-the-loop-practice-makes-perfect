"""A planned base path must clear the counter legs everywhere the executor checks it.

agni rack-food-rim-contact (c635f67), EES seed 0, 8,000 steps: after ReturnFoodToRack,
PickScraper drove 256 steps round the counter and failed: "Scraper motion would collide,
cost 0.00113: [('cupboard_1_leg_1', 'robot_base_link/geom_341', -0.00013)]". The next 15
PickScraper calls refused at 0 steps, each with leg_1 1.24 mm deep.

The base was never in the leg: it stopped 2.7 mm clear of it. Both depths are those of
the *next* commanded configuration, which `RackScraper.step` checks before commanding
it. The planner had checked its base path against the same MuJoCo guard, but only at
upstream's body-check spacing (1 cm, 0.01 rad), and a chassis corner sweeping past the
thin leg collides for less than that between two clear checks: segment 93 of the first
plan, 85% along; segment 0 of every replan from where it stopped, 15% along. Replanning
is deterministic, so every retry planned the same grazing first segment.

`local_ees0_pick_scraper_before_leg.json`: recording row 6567, the PickScraper's start.
`local_ees0_base_in_counter_leg.json`: recording row 6824, the first zero-step call's
start (the last tick of the failed PickScraper).
"""

from pathlib import Path

import numpy as np
import pytest

from tests.environments.racktoserving3d.test_stuck_states import (
    done_or_refused,
    restored_frozen,
)

FIXTURES = Path(__file__).parent / "fixtures"
BEFORE_LEG = FIXTURES / "local_ees0_pick_scraper_before_leg.json"
BESIDE_LEG = FIXTURES / "local_ees0_base_in_counter_leg.json"
STARTS = pytest.mark.parametrize("start", [BEFORE_LEG, BESIDE_LEG], ids=["before", "beside"])
# Finer than the executor's own check: `RackScraper.step` checks five configurations
# across one 15 mm / 0.015 rad base step.
DENSE_M = 0.001


def planned_pick(*, env):
    from hitl_pmp.environments.racktoserving3d.planning import ScraperMotionPlanner
    from hitl_pmp.environments.racktoserving3d.tool import PRE_GRASP

    tool = env.rack_controllers().tool
    pose = tool.wiper_pose()
    rotation = pose[:3, :3] @ tool.hand_in_body
    grasp = tool.grasp_world(pose=pose)
    return ScraperMotionPlanner.plan_pick(
        tool=tool,
        grasp=grasp,
        pre=grasp - PRE_GRASP * rotation[:, 2],
        rotation=rotation,
        preferred=tool.base,
        after=tool.lift_waypoints(pose=pose),
    )


@STARTS
def test_the_planned_base_path_passes_the_guard_between_its_waypoints(*, start):
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = restored_frozen(fixture=start)
    try:
        controllers = env.rack_controllers()
        shelf, tool = controllers.shelf, controllers.tool
        arm = shelf.data.qpos[shelf.qa].copy()
        with controllers.feedback():
            path = planned_pick(env=env).base_path
        for index, (begin, end) in enumerate(zip(path[:-1], path[1:], strict=True)):
            count = max(1, int(np.ceil(np.max(np.abs(end - begin)) / DENSE_M)))
            for fraction in np.linspace(0.0, 1.0, count + 1):
                try:
                    tool.check_configuration(
                        base=begin + fraction * (end - begin), arm=arm, holding=False
                    )
                except SkillFailed as exc:
                    pytest.fail(f"segment {index} at {fraction:.3f}: {exc}")
    finally:
        env.close()


def test_pick_scraper_beside_the_leg_moves_off_it():
    """The state the 15 zero-step calls started from: the base 2.7 mm clear of leg_1."""
    env = restored_frozen(fixture=BESIDE_LEG)
    try:
        steps, error = done_or_refused(env=env, action=np.array([0.0, 0, 0, 0, 0]))
        assert steps > 0, error
        assert "cupboard_1_leg" not in (error or ""), f"{steps} steps, then: {error}"
    finally:
        env.close()


def test_the_pick_scraper_that_grazed_the_leg_picks_the_scraper():
    env = restored_frozen(fixture=BEFORE_LEG)
    try:
        steps, error = done_or_refused(env=env, action=np.array([0.0, 0, 0, 0, 0]))
        assert error is None, f"{steps} steps, then: {error}"
        assert env.backend().held(name="wiper_0")
    finally:
        env.close()
