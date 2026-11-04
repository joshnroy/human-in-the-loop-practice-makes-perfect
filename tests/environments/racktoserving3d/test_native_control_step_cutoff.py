"""Opt-in native cutoff smoke test; run only in an approved bounded heavy-job slot."""

import os

import numpy as np
import pytest

from hitl_pmp.core.control_steps import ControlStepLimitReached
from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
from hitl_pmp.environments.racktoserving3d.objects import RackServing3DObjects as Objects
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.step_protocol import PracticeClock

pytestmark = [
    pytest.mark.main_only,
    pytest.mark.skipif(
        os.environ.get("HITL_RACK_NATIVE_CUTOFF") != "1",
        reason="requires an available run-heavy slot, memory cap and external timeout",
    ),
]


def test_native_pick_scraper_stops_after_three_control_steps_and_adopts_without_moving():
    env = RackServingEnvironment(scene_bg=False)
    try:
        env.hard_reset()
        backend = env.backend()
        initial_steps = backend.control_steps()
        clock = PracticeClock(budget=3, interval=100, robot_active=True, on_measure=lambda: None)
        env.set_control_step_observers(before=clock.before_robot_step, after=clock.after_robot_step)
        action = np.array([SKILL_IDS["PickScraper"], 0, 0, 0, 0], dtype=float)
        with pytest.raises(ControlStepLimitReached):
            env.take_action(action=action)
        env.set_control_step_observers(before=None, after=None)
        cutoff_time = float(backend.data().time)
        state = env.adopt_interrupted_skill(action=action, reason="practice_step_budget")
        assert clock.steps == env.last_controller_steps() == env.rack_controllers().steps == 3
        assert backend.control_steps() - initial_steps == 3
        assert float(backend.data().time) == cutoff_time
        assert state.get(obj=Objects.scene, feature_name="steps_taken") == 1
        assert state.get(obj=Objects.scene, feature_name="elapsed_seconds") > 0
        assert env.last_skill_error() == "interrupted: practice_step_budget"
    finally:
        env.set_control_step_observers(before=None, after=None)
        env.close()
