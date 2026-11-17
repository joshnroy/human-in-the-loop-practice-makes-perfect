"""A welded pick's or placement's descent goes as far as the fixtures allow, in execution too.

Robot batch return after a GatherAndSweep, seed 6 (2026-11-13): PickFood of a dumpling
on the plate planned its straight descent clear all the way down (reached 0.950 m, the
grasp depth itself), and failed in execution with "food_approach: Cartesian fraction
1.000: straight joint segment fraction 1.000: Scraper motion would collide ...
[('box_0_rim_40', 'robot_left_pad1', -2e-05)]", the base exactly on its planned stance.
The planner had checked the descent at its own IK solutions, 1 cm apart; the executor
solves each 5 mm waypoint again, within the IK's 3 mm tolerance of the same hand pose,
and its solution for the last waypoint put a pad 0.02 mm into the plate's rim.

The plan already ends a descent where the fixtures stop it. So does the hand: a waypoint
the execution guard refuses as a collision, before it is commanded, ends the descent at
the waypoint before it, and the dumpling is snapped into the hand from there.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)
SETUP_A = RackServingConfig(
    start_layout="middle_three",
    side_offset=0.13,
    plate_tuck=0.035,
    surface_kind="wire_solid",
    scraper_attachment="grasp_weld",
    two_route_rewards=True,
    bulk_transfer_only=True,
)
RIM = "Scraper motion would collide, cost 0.00102: [('box_0_rim_40', 'robot_left_pad1', -2e-05)]"


def _pick_with_the_last_waypoint_refused(*, monkeypatch, message):
    """PickFood of the middle dumpling, the executor's IK refusing, with `message`,
    every approach waypoint within 3 mm of the planned grasp depth. Returns the
    environment, the food controller's error (or None) and the planned depth."""
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController, WeldedFoodSkills
    from hitl_pmp.environments.racktoserving3d.tool import RackScraper
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

    env = RackServingEnvironment(config=SETUP_A)
    env.reset_to_seed(seed=0)
    planned = {}
    plan_pick = WeldedFoodSkills._plan_pick
    solve = RackScraper.ik

    def planning(self, *, name):  # noqa: PLR0917 -- a method's self
        plan, payload = plan_pick(self, name=name)
        planned["depth"] = float(payload["reached"][2, 3])
        planned["hover"] = float(payload["hover"][2, 3])
        return plan, payload

    def refusing(self, *, position, **kwargs):  # noqa: PLR0917 -- a method's self
        if "depth" in planned and position[2] < planned["depth"] + 0.003:
            raise SkillFailed(message)
        return solve(self, position=position, **kwargs)

    monkeypatch.setattr(WeldedFoodSkills, "_plan_pick", planning)
    monkeypatch.setattr(RackScraper, "ik", refusing)
    error = None
    try:
        RackFoodController(controllers=env.rack_controllers()).pick(name="cube_2")
    except SkillFailed as exc:
        error = str(exc)
    return env, error, planned


def test_a_descent_whose_last_waypoint_the_guard_refuses_stops_there_and_picks(*, monkeypatch):
    env, error, planned = _pick_with_the_last_waypoint_refused(monkeypatch=monkeypatch, message=RIM)
    try:
        assert error is None
        assert planned["hover"] - planned["depth"] > 0.02
        assert env.backend().held(name="cube_2")
        (stopped,) = env.rack_controllers().metrics["descents_stopped_short"]
        assert stopped["label"] == "food_approach"
        assert RIM in stopped["refusal"]
        # One 5 mm waypoint short of the planned depth, not at the hover.
        assert 0.003 <= stopped["short_m"] <= 0.008
    finally:
        env.close()


def test_a_descent_refused_for_anything_but_a_collision_still_fails(*, monkeypatch):
    env, error, _ = _pick_with_the_last_waypoint_refused(
        monkeypatch=monkeypatch, message="Food waypoint IK is inaccurate"
    )
    try:
        assert "food_approach: Cartesian fraction" in error
        assert not env.backend().held(name="cube_2")
        assert "descents_stopped_short" not in env.rack_controllers().metrics
    finally:
        env.close()


def test_an_unrefused_descent_reaches_its_planned_depth_and_records_nothing():
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.food import RackFoodController

    env = RackServingEnvironment(config=SETUP_A)
    try:
        env.reset_to_seed(seed=0)
        RackFoodController(controllers=env.rack_controllers()).pick(name="cube_2")
        assert env.backend().held(name="cube_2")
        assert "descents_stopped_short" not in env.rack_controllers().metrics
        assert np.isfinite(env.rack_controllers().shelf.site_pose()).all()
    finally:
        env.close()
