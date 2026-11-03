"""PlacePitcherAtHome's return is planned upright, its seat stays a scripted contact."""

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_planned_placement")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
POUR1 = np.array([1, 0, 1, 91, 0.01, 10], dtype=float)
PLACE = np.array([2, 0, 1, 0, 0, 0], dtype=float)
HELD_PHASES = {
    "planned tuck jug for return",
    "planned carry jug to placement pose",
    "planned arm to jug hover above home",
}
FREE_PHASES = {"planned clear workspace"}
# As in test_kitchen_planned_carry: planned within 10 degrees, the compliant grasp
# adds a few; 15 bounds the executed tilt.
EXECUTED_TILT_LIMIT_DEG = 15.0


# ~13-16 min on a CI runner: pull requests skip it, main runs it.
@pytest.mark.main_only
def test_the_return_is_refused_whole_or_executed_upright_clear_and_seated(*, monkeypatch):
    # Imported here, not at module level: kitchen_clearance imports mujoco, and the
    # unit job collects every main_only file (test_main_only_marker) with no OSMesa.
    from kitchen_clearance import Clearance

    from hitl_pmp.environments.pouring3d.kitchen.planning import parameterized_skills

    env = KitchenEnvironment(requested_count=1)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    env.take_action(action=POUR1)
    assert env.last_info["error"] is None, env.last_info
    c = env.controller()

    def refuse(**_):
        raise parameterized_skills.PlanningRefused("No collision-free base path to placement")

    monkeypatch.setattr(parameterized_skills.KitchenMotionPlanner, "plan_base", refuse)
    steps = c.steps
    env.take_action(action=PLACE)
    assert env.last_info["error"] == "No collision-free base path to placement"
    assert c.steps == steps and c.held == "pitcher"
    monkeypatch.undo()

    seen = Clearance.record(env=env)
    env.take_action(action=PLACE)
    assert env.last_info["success"] is True, env.last_info
    phases = {row[0] for row in seen}
    assert phases >= HELD_PHASES | FREE_PHASES, phases
    Clearance.check(seen=seen)
    held = [row for row in seen if row[0] in HELD_PHASES]
    tilt = max(row[2] for row in held)
    print(f"largest MuJoCo jug tilt over the return: {tilt:.1f} deg")
    assert tilt <= EXECUTED_TILT_LIMIT_DEG
    position, rotation = c.body_pose(name="pitcher")
    assert np.linalg.norm(position[:2] - c.home[:2]) < 0.01
    assert rotation.magnitude() < np.deg2rad(5)
