"""Loaded contact stops with reserve; unloaded planning retains native bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import (
    ContactJointReserveLimit,
    ContactTravelLimit,
    FloorPrimitives,
)


@pytest.mark.parametrize("phase", ["contact drive", "contact correction"])
@pytest.mark.parametrize("joint_2", [2.231, 2.2401583194732666])
def test_loaded_motion_requests_retreat_without_changing_state_or_limits(
    *, phase: str, joint_2: float
) -> None:
    arm = np.array([0.0, joint_2, 0.0, 0.0, 0.0, 0.0, 0.0])
    limits = np.array([
        [-np.inf, np.inf],
        [-2.24, 2.24],
        [-np.inf, np.inf],
        [-2.57, 2.57],
        [-np.inf, np.inf],
        [-2.09, 2.09],
        [-np.inf, np.inf],
    ])
    original_arm, original_limits = arm.copy(), limits.copy()
    records = []
    primitive = SimpleNamespace(
        session=SimpleNamespace(
            arm=lambda: arm, ticks=18399, _write=lambda **kwargs: records.append(kwargs["record"])
        ),
        scene=SimpleNamespace(_arm_limits=limits),
        wiper_loaded_by_cube=lambda: True,
    )
    # The base drive catches ContactTravelLimit and follows checked retreat;
    # corrective descent also catches this specific reserve stop explicitly.
    with pytest.raises(ContactTravelLimit) as stopped:
        FloorPrimitives.guard_loaded_joint_reserve(primitive, phase=phase)
    assert isinstance(stopped.value, ContactJointReserveLimit)
    assert records[0]["phase"] == phase
    assert records[0]["native_joint_margin"] == pytest.approx(2.24 - joint_2)
    assert records[0]["already_outside_native_limits"] == (joint_2 > 2.24)
    np.testing.assert_array_equal(arm, original_arm)
    np.testing.assert_array_equal(limits, original_limits)


def test_unloaded_pose_never_reads_or_restricts_native_arm() -> None:
    primitive = SimpleNamespace(wiper_loaded_by_cube=lambda: False)
    FloorPrimitives.guard_loaded_joint_reserve(primitive, phase="contact correction")


def test_loaded_pose_with_reserve_continues() -> None:
    primitive = SimpleNamespace(
        session=SimpleNamespace(arm=lambda: np.zeros(7)),
        scene=SimpleNamespace(_arm_limits=np.tile([-2.0, 2.0], (7, 1))),
        wiper_loaded_by_cube=lambda: True,
    )
    FloorPrimitives.guard_loaded_joint_reserve(primitive, phase="contact drive")
