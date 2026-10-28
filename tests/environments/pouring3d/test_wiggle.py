"""Reduced controls and actual retained-grasp wrist wiggling."""

import numpy as np
import pytest
from pydantic import ValidationError


def test_wiggle_parameters_and_action_contract():
    from hitl_pmp.environments.pouring3d.wiggle import (
        TiltWiggleParameters,
        TiltWigglePouringEnvironment,
    )

    assert set(TiltWiggleParameters.model_fields) == {"tilt_angle", "wiggle_seconds"}
    assert TiltWigglePouringEnvironment.action_space.shape == (4,)
    assert TiltWigglePouringEnvironment().noop_action().shape == (4,)
    assert TiltWiggleParameters(tilt_angle=110, wiggle_seconds=0).wiggle_seconds == 0
    for invalid in (
        {"offset_x": 0.1},
        {"wiggle_hz": 2},
        {"tilt_angle": 147},
        {"wiggle_seconds": -1},
        {"wiggle_seconds": float("nan")},
    ):
        with pytest.raises(ValidationError):
            TiltWiggleParameters(**invalid)


def test_native_wiggle_moves_the_pitcher_and_preserves_physical_goal_checks():
    import inspect
    from pathlib import Path

    from kinder.envs.dynamic3d.pitcher_pour3d import PitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.wiggle import TiltWigglePouringEnvironment

    task = (
        Path(inspect.getfile(PitcherPour3DEnv)).parent
        / "tasks/PitcherPour3D/PitcherPour3D-o5-winco.json"
    )
    env = TiltWigglePouringEnvironment(
        seed=0,
        requested_count=2,
        cube_material="foam",
        task_config_path=str(task),
        scene_bg=False,
    )
    env.hard_reset()
    try:
        assert all(action.shape == (4,) for action in env.get_valid_actions())
        with pytest.raises(ValueError, match="four-component"):
            env.take_action(action=np.zeros(7))
        env.take_action(action=np.array([0, 0, 0, 0]))
        assert env.last_info["success"]
        assert env.sim.locations() == ["pitcher"] * 5
        assert any(int(action[0]) == 1 for action in env.get_valid_actions())
        env.take_action(action=np.array([1, 0, 108, 2]))
        event = env.last_info
        assert event["error"] is None
        assert set(event["parameters"]) == {"tilt_angle", "wiggle_seconds"}
        assert event["fixed_settings"]["aim_offset_m"] == [0, 0]
        assert event["success"] == env.sim.snapshot()["goal"]
        assert env.sim.held == "pitcher"
        wave = env.sim.controller.wiggle_trace
        assert len(wave) == 20
        assert max(abs(row["offset_degrees"]) for row in wave) <= 4
        assert abs(wave[-1]["offset_degrees"]) < 1e-8
        measured = np.array([row["actual_tilt_degrees"] for row in wave])
        assert np.ptp(measured) > 0.5
        # A constant command can drift while the grasp settles. Require actual
        # motion in both directions to distinguish oscillation from that drift.
        changes = np.diff(measured)
        assert np.count_nonzero(changes > 0.15) >= 3
        assert np.count_nonzero(changes < -0.15) >= 3
        assert env.sim.controller.steps > 0
        assert env.sim.elapsed_seconds > 0
    finally:
        env.sim.close()
