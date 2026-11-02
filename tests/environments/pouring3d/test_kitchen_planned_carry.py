"""Pour's carry from the lift to the pour start is planned upright and executed clear."""

import numpy as np
import pytest
from kitchen_clearance import Clearance

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_planned_carry")

PICK_PITCHER = np.array([0, 0, 1, 0, 0, 0], dtype=float)
POUR1 = np.array([1, 0, 1, 91, 0.01, 10], dtype=float)
CARRY_PHASES = {
    "planned tuck jug for carry",
    "planned carry jug to pour pose",
    "planned arm to pour start",
}
# The planner keeps the rigidly modelled jug within 10 degrees of vertical. In MuJoCo the
# handle grasp is compliant and the jug already hangs about 5 degrees after the lift,
# so the executed bound is looser: measured 11.5 degrees on 2026-11-02 (3.6 degrees of
# hang after the lift plus compliance in motion), held here to 15.
EXECUTED_TILT_LIMIT_DEG = 15.0


def test_the_carry_is_refused_whole_or_executed_upright_and_clear(*, monkeypatch):
    from hitl_pmp.environments.pouring3d.kitchen.planning.parameterized_skills import (
        KitchenMotionPlanner,
    )

    env = KitchenEnvironment(requested_count=1)
    env.take_action(action=PICK_PITCHER)
    assert env.last_info["success"] is True, env.last_info
    c = env.controller()

    # No upright path exists for a zero tilt budget (the jug hangs a few degrees in the
    # hand): the whole carry is refused before anything moves.
    monkeypatch.setattr(KitchenMotionPlanner, "tilt_limit_deg", 0.0)
    steps = c.steps
    env.take_action(action=POUR1)
    assert env.last_info["error"].startswith("No collision-free upright ")
    assert c.steps == steps
    assert c.held == "pitcher"
    monkeypatch.undo()

    seen = Clearance.record(env=env)
    env.take_action(action=POUR1)
    assert env.last_info["error"] is None, env.last_info
    assert {row[0] for row in seen} >= CARRY_PHASES
    Clearance.check(seen=seen)
    carry = [row for row in seen if row[0] in CARRY_PHASES]
    tilt = max(row[2] for row in carry)
    print(f"largest MuJoCo jug tilt over the carry: {tilt:.1f} deg")
    assert tilt <= EXECUTED_TILT_LIMIT_DEG
    assert min(row[3] for row in carry) == 3, "a dumpling left the jug during the carry"
