"""The sideways wiggle must lie in the horizontal plane across the pour."""

import numpy as np
import pytest
from pydantic import ValidationError

from hitl_pmp.environments.pouring3d.lateral_wiggle import (
    LateralPouringController,
    LateralPourParameters,
)


def test_wiggle_positions_are_perpendicular_to_pour_and_parallel_to_ground():
    center = np.array([0.65, 0.20, 0.45])
    points = np.array([
        LateralPouringController.command_position(center=center, elapsed=t, duration=12)
        for t in np.linspace(0, 12, 121)
    ])
    offsets = points - center
    assert np.allclose(offsets @ np.array([0, 1, 0]), 0)
    assert np.allclose(offsets @ np.array([0, 0, 1]), 0)
    assert offsets[:, 0].max() > 0.01 and offsets[:, 0].min() < -0.01
    assert np.max(np.abs(offsets[:, 0])) <= 0.015 + 1e-12
    assert np.allclose(points[[0, -1]], center)


def test_long_wiggle_keeps_two_practiced_parameters():
    assert set(LateralPourParameters.model_fields) == {"tilt_angle", "wiggle_seconds"}
    assert LateralPourParameters(wiggle_seconds=12).wiggle_seconds == 12
    with pytest.raises(ValidationError):
        LateralPourParameters(wiggle_seconds=21)
    with pytest.raises(ValidationError):
        LateralPourParameters(wiggle_hz=2)


def test_native_pour_macro_dispatches_to_long_horizontal_wiggle():
    import inspect
    from pathlib import Path

    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.lateral_environment import LateralPouringEnvironment

    task = (
        Path(inspect.getfile(PitcherPour3DEnv)).parent
        / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
    )
    env = LateralPouringEnvironment(
        seed=0, requested_count=2, cube_material="foam", scene_bg=False, task_config_path=str(task)
    )
    env.hard_reset()
    try:
        assert isinstance(env.sim.controller, LateralPouringController)
        assert env.noop_action().shape == (4,)
        assert all(action.shape == (4,) for action in env.get_valid_actions())
        with pytest.raises(ValueError, match="four-component"):
            env.take_action(action=np.zeros(7))
        with pytest.raises(ValueError, match="Human reset"):
            env.take_action(action=np.array([5, 0, 103, 6]))
        env.take_action(action=np.array([0, 0, 103, 6]))
        assert env.last_info["success"]
        before = env.sim.elapsed_seconds
        env.take_action(action=np.array([1, 0, 103, 6]))
        event = env.last_info
        assert event["controller"] == "horizontal_lateral_wiggle_v1"
        assert event["error"] is None
        assert event["parameters"] == {"tilt_angle": 103, "wiggle_seconds": 6}
        assert event["success"] == event["after"]["goal"]
        assert env.sim.held == "pitcher"
        assert env.sim.events[-1] == event
        assert env.sim.elapsed_seconds - before == pytest.approx(event["cost_seconds"])
        assert event["human_interventions"] == 0
        trace = event["wiggle_trace"]
        assert len(trace) == 60
        commands = np.array([row["commanded_position"] for row in trace])
        assert np.allclose(commands[:, 1:], commands[0, 1:])
        assert np.ptp(commands[:, 0]) > 0.02
        assert all(row["commanded_tilt_degrees"] == 103 for row in trace)
        assert event["wiggle_before"]["time"] < event["wiggle_after"]["time"]
    finally:
        env.sim.close()
