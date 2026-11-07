"""The human pour-back from a state whose skillet is off the stove.

The welded skillet never moves under physics, but MuJoCo's own divergence guard
(`mj_checkAcc`) calls `mj_resetData` on a NaN or huge acceleration: every body goes to
the model's `qpos0` (the skillet to the world origin) and every equality to its
`eq_active0`, so the skillet weld is off. One Della run (2026-11-06, curtis p = 1.0
seed 0, action 165) reached the pour-back from exactly that state; the reset stood the
jug home, left the skillet at the origin, returned False and the oracle ended the run.
"""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.humans.oracle import UnconditionalHumanOracle

needs_kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER")
pytestmark = [needs_kinder, pytest.mark.xdist_group("kitchen_human_reset_skillet")]


def _wiped() -> KitchenEnvironment:
    """The state MuJoCo's divergence guard leaves: `mj_resetData`, then `mj_forward`."""
    import mujoco

    env = KitchenEnvironment(requested_count=2)
    c = env.controller()
    mujoco.mj_resetData(c.model, c.data)
    mujoco.mj_forward(c.model, c.data)
    assert not c.vessel_home(name="tray")
    assert not c.data.eq_active[c.model.equality(c.scene.SKILLET_WELD).id]
    return env


def test_the_pour_back_from_a_wiped_state_puts_jug_food_and_skillet_home():
    env = _wiped()
    try:
        c = env.controller()
        assert env.reset_movables(destination=HUMAN_RESET)
        assert env.last_info["success"]
        assert c.vessel_home(name="pitcher")
        assert c.vessel_home(name="tray")
        assert c.scene.cube_locations() == ["pitcher"] * 3
        assert c.ready()
        np.testing.assert_array_equal(c.data.qpos[c.free_qpos(name="tray")], c.skillet_home_qpos)
        # Welded again, so no later pour, pick or spill can move it.
        assert c.data.eq_active[c.model.equality(c.scene.SKILLET_WELD).id]
        before = c.data.qpos[c.free_qpos(name="tray")].copy()
        c.pause(phase="wait", seconds=1)
        np.testing.assert_allclose(c.data.qpos[c.free_qpos(name="tray")], before, atol=1e-4)
    finally:
        env.close()


def test_a_pour_back_that_does_not_reach_ready_is_attempted_and_scored_a_failure(*, monkeypatch):
    env = _wiped()
    try:
        monkeypatch.setattr(type(env.controller()), "ready", lambda self: False)
        # Attempted, as a failed draw of p is (#746): the oracle reports the outcome
        # rather than raising and ending the run.
        UnconditionalHumanOracle.execute_movables_reset(env=env, destination=HUMAN_RESET)
        assert env.last_info["skill"] == HUMAN_RESET
        assert env.last_info["success"] is False
        assert env.human_interventions == 1
    finally:
        env.close()
