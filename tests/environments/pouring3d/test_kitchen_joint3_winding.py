"""From the state where ees/ees/2 (pouring-one-shot-20261105-abfe4ed) wound joint 3 into
its IK bound, PickPitcher and the Pour that follows still have room to move.

At tick 24 the hand is empty, reaching over the jug after a drop, with joint 3 at
380 deg. The recorded PickPitcher took it to 687 deg (bound 688), and every Pour after
failed before pouring."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

pytestmark = pytest.mark.xdist_group("kitchen_joint3_winding")

TICK = Path(__file__).with_name("kitchen_joint3_wound_tick24.json")
# Pour1, the recorded parameter of the first Pour1 that failed (step_events line 28).
POUR1 = np.array([1.0, 0.0, 1.0, 0.9131806745982157, 0.0, 0.0])


def test_a_pick_from_the_recorded_wound_state_leaves_joint_3_room_to_pour():
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    tick = json.loads(TICK.read_text())
    c.data.qpos[:] = tick["qpos"]
    c.data.qvel[:] = tick["qvel"]
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    assert np.rad2deg(c.data.qpos[c.arm_q[2]]) == pytest.approx(380.4, abs=0.5)
    for _ in range(30):
        c.pause(phase="settle restored state", seconds=0.1, grip=0)
        if c.ready():
            break
    try:
        c.pick_pitcher()
        q3 = np.rad2deg(c.data.qpos[c.arm_q[2]])
        assert 688 - abs(q3) > 180, f"joint 3 at {q3:.0f} deg after PickPitcher"
        env.take_action(action=POUR1)
        assert env.last_info.get("error") is None, env.last_info.get("error")
    finally:
        env.close()
