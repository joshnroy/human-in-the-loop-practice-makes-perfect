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


# A plan that only touches nothing is not enough (robot batch return after a Sweep, seed
# 2, 2026-11-13): the planned path passed counter leg_1 0.21 mm clear, and the straight
# line the executor checks, from where the base was to its next command a centimetre
# ahead, cut 0.011 mm into it. `plan_base` therefore asks for `BASE_CLEARANCE_M` of
# clearance wherever a path with it exists. No simulator: the search itself is replaced.
START, GOAL = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
GRAZING = [START, np.array([0.5, 0.0, 0.0]), GOAL]
DETOUR = [START, np.array([0.5, 0.2, 0.0]), GOAL]


def _plan_base(*, monkeypatch, searches, clearance_guard):
    """`plan_base` with the search replaced: each search returns the grazing path
    unless its constraint refuses that path's middle, in which case the detour."""
    from hitl_pmp.environments.racktoserving3d import planning
    from hitl_pmp.environments.racktoserving3d.planning import ScraperMotionPlanner

    class Pose:
        def __init__(self, *, pose):
            self.x, self.y, self._theta = (float(v) for v in pose)

        def theta(self):
            return self._theta

    def search(*, also_collides, **_):
        blocked = also_collides(Pose(pose=GRAZING[1]))
        searches.append("with clearance" if blocked else "plain")
        return [p.copy() for p in (DETOUR if blocked else GRAZING)]

    monkeypatch.setattr(planning.KitchenMotionPlanner, "plan_base", search)
    state = type("State", (), {"get_object_names": lambda self: []})()
    return ScraperMotionPlanner.plan_base(
        sim=None,
        state=state,
        base=START.copy(),
        target=GOAL.copy(),
        arm=None,
        what="a test stance",
        guard=lambda pose: True,
        clearance_guard=clearance_guard,
    )


def _as_lists(*, path):
    return [np.round(pose, 6).tolist() for pose in path]


def test_a_base_path_with_clearance_is_the_path_planned_before(*, monkeypatch):
    searches = []
    path = _plan_base(monkeypatch=monkeypatch, searches=searches, clearance_guard=lambda p: True)
    assert _as_lists(path=path) == _as_lists(path=GRAZING)
    assert searches == ["plain"]


def test_a_base_path_without_clearance_is_planned_again_with_it(*, monkeypatch):
    def keeps_clear(pose):  # noqa: PLR0917 -- the guard is called positionally
        return not (abs(pose[0] - 0.5) < 0.05 and abs(pose[1]) < 0.05)

    searches = []
    path = _plan_base(monkeypatch=monkeypatch, searches=searches, clearance_guard=keeps_clear)
    assert _as_lists(path=path) == _as_lists(path=DETOUR)
    assert searches == ["plain", "with clearance"]


def test_where_no_base_path_has_clearance_the_plain_path_still_stands(*, monkeypatch):
    """A base that starts closer to the counter than the clearance (BESIDE_LEG above)
    must still be able to leave."""
    searches = []
    path = _plan_base(monkeypatch=monkeypatch, searches=searches, clearance_guard=lambda p: False)
    assert _as_lists(path=path) == _as_lists(path=GRAZING)
    assert searches[0] == "plain" and set(searches[1:]) == {"with clearance"}


def test_the_chassis_clearance_refuses_a_base_nearer_a_leg_than_it_and_nothing_else():
    from hitl_pmp.environments.racktoserving3d.planning import BASE_CLEARANCE_M
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    assert BASE_CLEARANCE_M == 0.003
    env = restored_frozen(fixture=BESIDE_LEG)
    try:
        controllers = env.rack_controllers()
        shelf, tool = controllers.shelf, controllers.tool
        base, arm = shelf.data.qpos[shelf.ba].copy(), shelf.data.qpos[shelf.qa].copy()
        margins = shelf.model.geom_margin.copy()
        # 2.7 mm clear of leg_1: the executor's check passes, the planner's margin does not.
        tool.check_configuration(base=base, arm=arm, holding=False)
        with pytest.raises(SkillFailed, match="cupboard_1_leg_1"):
            tool.check_configuration(
                base=base, arm=arm, holding=False, chassis_clearance=BASE_CLEARANCE_M
            )
        tool.check_configuration(base=base, arm=arm, holding=False, chassis_clearance=0.002)
        # Away from the counter the clearance refuses nothing, the floor included.
        away = base + [-0.5, 0.5, 0.0]
        tool.check_configuration(
            base=away, arm=arm, holding=False, chassis_clearance=BASE_CLEARANCE_M
        )
        np.testing.assert_array_equal(shelf.model.geom_margin, margins)
    finally:
        env.close()


def test_only_the_robot_returns_food_motions_plan_their_base_with_the_clearance(*, monkeypatch):
    """Every other skill's base plans stay exactly as they were, the return's own
    scraper motions included: the clearance is set on the shelf around the return's
    food pick and placement only, and cleared again whatever their outcome."""
    from types import SimpleNamespace

    from hitl_pmp.environments.racktoserving3d import robot_batch_return
    from hitl_pmp.environments.racktoserving3d.planning import (
        BASE_CLEARANCE_M,
        ScraperMotionPlanner,
    )
    from hitl_pmp.environments.racktoserving3d.robot_batch_return import RackRobotBatchReturn
    from hitl_pmp.environments.racktoserving3d.tool import RackShelfSkills
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    assert RackShelfSkills.base_clearance == 0.0
    shelf = SimpleNamespace(base_clearance=0.0)
    tool = SimpleNamespace(shelf=shelf)
    assert ScraperMotionPlanner.clearance_guard(tool=tool, ok=None, arm=None) is None
    seen = []

    class Food:
        def __init__(self, **_):
            pass

        def pick(self, **_):
            seen.append(("pick", shelf.base_clearance))

        def place(self, **_):
            seen.append(("place", shelf.base_clearance))
            raise SkillFailed("refused for this test")

    monkeypatch.setattr(robot_batch_return, "WeldedFoodSkills", Food)
    monkeypatch.setattr(robot_batch_return, "RackFoodController", Food)
    returner = RackRobotBatchReturn.__new__(RackRobotBatchReturn)
    returner.controllers = SimpleNamespace(shelf=shelf)
    returner.backend = SimpleNamespace(held=lambda **_: False)
    with pytest.raises(SkillFailed):
        returner.put_back(name="cube_0", slot=(1.5, 0.0, 0.0))
    assert seen == [("pick", BASE_CLEARANCE_M), ("place", BASE_CLEARANCE_M)]
    assert shelf.base_clearance == 0.0
    shelf.base_clearance = BASE_CLEARANCE_M
    assert ScraperMotionPlanner.clearance_guard(tool=tool, ok=None, arm=None) is not None
