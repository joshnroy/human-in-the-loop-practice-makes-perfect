"""The validated approach plan is followed no faster than the arm can track it.

From the recorded seed-1 state below, the fourth grasp candidate's plan swings joints 1
and 3 by 26 deg within one 4-tick waypoint while the base drives in. The arm tracks about
0.02 rad per tick, so the old flat timing fell 20 deg behind and failed "position for shrimp
pick", reported as "descend to shrimp body". Timed to the joint speed, the arm follows the
plan to the dumpling. With a 10 mm IK tolerance the descent beside the pan wall, 11 mm away,
now completes and the dumpling is picked.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController

pytestmark = pytest.mark.xdist_group("kitchen_pick_waypoint_timing")

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_dumpling_at_near_pan_wall.json"


def restored() -> KitchenEnvironment:
    recorded = json.loads(FIXTURE.read_text())
    env = KitchenEnvironment(requested_count=1)
    env.reset_to_seed(seed=0, requested=1)
    c = env.controller()
    c.data.qpos[:] = recorded["qpos"]
    c.data.qvel[:] = recorded["qvel"]
    c.data.time = recorded["time"]
    mujoco.mj_forward(c.model, c.data)
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()
    env.current_state = env.observe()
    return env


def test_waypoint_ticks_scale_with_the_largest_joint_change():
    ticks = PotstickerController.waypoint_ticks
    speed, frequency = 0.20, 10.0
    # A small change keeps the phase's own share of time.
    assert ticks(duration=5 / 15, delta=np.full(7, 0.01), speed=speed, frequency=frequency) == 4
    # 26 deg on one joint needs 26 deg / (0.8 * 0.02 rad per tick) = 29 ticks.
    delta = np.zeros(7)
    delta[2] = np.deg2rad(-26.0)
    assert ticks(duration=5 / 15, delta=delta, speed=speed, frequency=frequency) == 29


def test_a_near_wall_pick_follows_its_plan_to_the_dumpling_and_picks_it():
    env = restored()
    c = env.controller()
    phases: list[str] = []
    observed = c.after_step

    def watch() -> None:
        if observed is not None:
            observed()
        phases.append(c.phase)

    c.after_step = watch
    env.take_action(action=np.array(json.loads(FIXTURE.read_text())["action"]))
    # Positioning and the approach no longer fail on tracking lag ...
    assert "descend to shrimp body" in phases
    # ... and the descent beside the wall completes.
    assert env.last_info["success"] is True, env.last_info
    assert c.held in c.scene.cube_names
