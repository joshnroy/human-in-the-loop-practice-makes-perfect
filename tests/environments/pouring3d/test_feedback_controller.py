"""Controller semantics independent of the expensive room simulation."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.feedback_controller import (
    FeedbackPourController,
    FeedbackPourParameters,
)


def test_amplitude_scales_the_same_waveform() -> None:
    """Duration truncates a fixed waveform; it cannot secretly rescale amplitude."""
    times = np.arange(0.1, 2.01, 0.1)
    small = np.array([FeedbackPourController.offset(elapsed=t, amplitude_m=0.005) for t in times])
    large = np.array([FeedbackPourController.offset(elapsed=t, amplitude_m=0.015) for t in times])
    np.testing.assert_allclose(large, 3 * small, atol=1e-12)
    assert small.min() < -0.004 and small.max() > 0.004
    assert (
        FeedbackPourParameters(duration_seconds=0.3).amplitude_m
        == FeedbackPourParameters(duration_seconds=8).amplitude_m
    )


def test_mouth_observation_counts_crossings_once_and_ignores_side_protrusion() -> None:
    """The cavity-core classifier can say air before a piece crosses the mouth."""
    origin = np.array([1.0, 2.0, 3.0])
    rotation = Rotation.from_euler("x", 100, degrees=True)
    scene = SimpleNamespace(
        cube_names=("shrimp_0", "shrimp_1", "shrimp_2"),
        get_object=lambda _: SimpleNamespace(height=0.14),
        cube_locations=lambda: ["air", "air", "pitcher"],
    )
    positions = {
        "pitcher": (origin, rotation),
        "shrimp_0": (origin + rotation.apply([0.1, 0, 0.12]), Rotation.identity()),
        "shrimp_1": (origin + rotation.apply([0.02, 0, 0.15]), Rotation.identity()),
        "shrimp_2": (origin + rotation.apply([0, 0, 0.05]), Rotation.identity()),
    }
    c = FeedbackPourController.model_construct(scene=scene, data=SimpleNamespace(time=0))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FeedbackPourController, "body_pose", lambda self, *, name: positions[name])
        assert c.observe_mouth() == 1
        assert c.observe_mouth() == 1
        assert c.outbound == {"shrimp_1"}
        positions["shrimp_0"] = (origin + rotation.apply([0.1, 0, 0.16]), Rotation.identity())
        assert c.observe_mouth() == 2


def test_reject_nonphysical_parameters() -> None:
    for values in (
        {"amplitude_m": -0.001},
        {"tilt_degrees": float("nan")},
        {"duration_seconds": -1},
    ):
        with pytest.raises(ValidationError):
            FeedbackPourParameters(**values)
