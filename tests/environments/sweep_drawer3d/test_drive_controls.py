"""Optional carried-base command bounds preserve default Drawer behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError, Motion


def test_default_drive_preserves_command_and_fixed_arm_tracking() -> None:
    recorded = []
    session = SimpleNamespace(
        arm=lambda: np.full(7, 0.02),
        base=lambda: (0.0, 0.0, 0.0),
        step=lambda *, action: recorded.append(action.copy()),
    )
    motion = Motion.model_construct(session=session, scene=None)
    assert not motion.drive(path=[(0.06, 0.08, 0.2)], grip=1.0, arm=np.zeros(7), max_ticks=1)
    expected = np.array([0.06, 0.08, 0.2, *([-0.02] * 7), 1.0])
    np.testing.assert_allclose(recorded, [expected], atol=1e-15)


def test_drive_bounds_norm_and_command_changes_through_direction_reversal() -> None:
    recorded = []
    session = SimpleNamespace(
        arm=lambda: np.zeros(7),
        base=lambda: (0.0, 0.0, 0.0) if len(recorded) < 6 else (2.0, 2.0, 2.0),
        step=lambda *, action: recorded.append(action.copy()),
    )
    motion = Motion.model_construct(session=session, scene=None)
    assert not motion.drive(
        path=[(1.0, 1.0, 1.0)],
        grip=1.0,
        max_ticks=18,
        max_translation_step=0.01,
        max_yaw_step=0.005,
        translation_step_change=0.002,
        yaw_step_change=0.001,
    )
    commands = np.asarray(recorded)
    changes = np.diff(np.vstack([np.zeros((1, 3)), commands[:, :3]]), axis=0)
    assert np.all(np.linalg.norm(commands[:, :2], axis=1) <= 0.01 + 1e-12)
    assert np.all(np.abs(commands[:, 2]) <= 0.005 + 1e-12)
    assert np.all(np.linalg.norm(changes[:, :2], axis=1) <= 0.002 + 1e-12)
    assert np.all(np.abs(changes[:, 2]) <= 0.001 + 1e-12)
    assert commands[0, 0] > 0 and commands[-1, 0] < 0
    assert commands[0, 2] > 0 and commands[-1, 2] < 0
    np.testing.assert_array_equal(commands[:, -1], np.ones(18))


def test_drive_tick_guard_stops_immediately_after_executed_tick() -> None:
    recorded = []
    session = SimpleNamespace(
        arm=lambda: np.zeros(7),
        base=lambda: (0.0, 0.0, 0.0),
        step=lambda *, action: recorded.append(action.copy()),
    )
    motion = Motion.model_construct(session=session, scene=None)

    def guard() -> None:
        assert len(recorded) == 1
        raise ExecutionError("bilateral grasp lost")

    with pytest.raises(ExecutionError, match="bilateral grasp lost"):
        motion.drive(path=[(1.0, 0.0, 0.0)], grip=1.0, max_ticks=20, tick_guard=guard)
    assert len(recorded) == 1
