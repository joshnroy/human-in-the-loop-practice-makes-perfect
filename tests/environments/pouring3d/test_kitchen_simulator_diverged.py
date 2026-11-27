"""MuJoCo's instability guard resetting the kitchen mid-run ends it as `SimulatorDiverged`.

On a NaN, Inf or huge `qpos`/`qvel`/`qacc`, MuJoCo warns and calls `mj_resetData`: every
body back to `qpos0`, every equality to its default (the skillet weld off), time to 0,
and the simulation carries on. In Della sweep 16997464 (human-reset-p1.0/curtis/pomdp/0)
a 20 g dumpling on the floor spun up to ~2,100 rad/s, the guard fired on DOF 21, and the
run carried on from the wiped scene until the controller raised "Physical pitcher grasp
lost". The fixture is that run's recorded state at t = 7176.0 s, five control ticks
before the guard fired.
"""

import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.core.control_steps import SimulatorDiverged
from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

needs_kinder = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER")
pytestmark = [needs_kinder, pytest.mark.xdist_group("kitchen_simulator_diverged")]

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_diverges_curtis_p1.json"


def _recorded(*, integrator: str | None = None) -> KitchenEnvironment:
    """The Della run's state at t = 7176.0 s, restored as the earlier diagnosis did.

    The run stepped with KinDER's `implicitfast`; the kitchen now uses `implicit`
    (`test_kitchen_integrator.py`), so a replay of the guard firing restores the
    integrator it was recorded under."""
    state = json.loads(FIXTURE.read_text())
    requested = state["requested_count"]
    env = KitchenEnvironment(requested_count=requested)
    env.reset_to_seed(seed=0, requested=requested)
    c = env.controller()
    if integrator is not None:
        c.model.opt.integrator = getattr(mujoco.mjtIntegrator, integrator)
    c.data.qpos[:] = state["qpos"]
    c.data.qvel[:] = state["qvel"]
    c.data.time = state["time"]
    c.scene.weld_skillet()
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.base_target = c.data.qpos[c.base_q].copy()
    return env


def test_stepping_the_recorded_state_raises_at_the_tick_the_guard_fires():
    env = _recorded(integrator="mjINT_IMPLICITFAST")
    try:
        c = env.controller()
        ticks = 0
        with pytest.raises(SimulatorDiverged) as raised:
            for ticks in range(1, 20):  # noqa: B007 -- read after the raise
                c.motor_step(target_q=c.last_q, grip=c.grip)
        # Ticks 1-5 run to t = 7176.5; the guard fires inside the sixth (t = 7176.525).
        assert ticks == 6
        details = raised.value.details
        assert details["time"] == pytest.approx(7176.5, abs=1e-5)
        assert details["warning"] == "mjWARN_BADQACC"
        assert details["index"] == 21
        assert details["body"] == "potsticker_1"
        assert details["time_after_reset"] < 1.0
        assert not isinstance(raised.value, Exception)
        assert "potsticker_1" in str(raised.value)
    finally:
        env.close()


def test_the_kitchen_integrator_steps_the_recorded_state_without_diverging():
    env = _recorded()
    try:
        c = env.controller()
        vadr = int(c.model.jnt_dofadr[c.model.body("potsticker_1").jntadr[0]])
        for _ in range(50):
            c.motor_step(target_q=c.last_q, grip=c.grip)
            assert np.linalg.norm(c.data.qvel[vadr + 3 : vadr + 6]) < 1000.0
        assert c.data.time == pytest.approx(7181.0, abs=1e-5)
    finally:
        env.close()


def _blow_up(*, env: KitchenEnvironment) -> None:
    """A velocity past MuJoCo's `mjMAXVAL`, which `mj_checkVel` resets on."""
    c = env.controller()
    vadr = int(c.model.jnt_dofadr[c.model.body("potsticker_0").jntadr[0]])
    c.data.qvel[vadr + 3] = 1e12


def test_a_skill_does_not_report_a_divergence_as_its_own_failure():
    env = KitchenEnvironment(requested_count=1)
    try:
        env.controller()
        _blow_up(env=env)
        wait = np.array([SKILL_NAMES.index("Wait"), 0, 1, 0, 0, 0], dtype=float)
        with pytest.raises(SimulatorDiverged) as raised:
            env.take_action(action=wait)
        assert raised.value.details["warning"] == "mjWARN_BADQVEL"
        assert raised.value.details["body"] == "potsticker_0"
    finally:
        env.close()


def test_an_ordinary_step_is_unchanged_by_the_check(*, monkeypatch):
    """The check only reads `data.time` and `data.warning`: a trajectory with it is
    bit-for-bit the trajectory without it."""
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenScene

    def trajectory() -> np.ndarray:
        env = KitchenEnvironment(requested_count=1)
        try:
            c = env.controller()
            for _ in range(20):
                c.motor_step(target_q=c.last_q, grip=c.grip)
            return np.r_[c.data.qpos, c.data.qvel, c.data.time]
        finally:
            env.close()

    guarded = trajectory()
    monkeypatch.setattr(KitchenScene, "check_simulator_stable", lambda self, **_: None)
    unguarded = trajectory()
    np.testing.assert_array_equal(guarded, unguarded)
