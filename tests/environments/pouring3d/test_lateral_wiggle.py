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
