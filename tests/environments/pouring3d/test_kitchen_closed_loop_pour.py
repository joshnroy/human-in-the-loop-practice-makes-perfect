"""The kitchen Pour's closed-loop control law, on a stubbed controller (no physics).

Josh, 2026-11-04: approach to 80 deg at the carry tilt speed, then ramp at the learned
rate with a fixed 40 mm sideways wiggle, capped at 95 deg; stop the moment the requested
count has left the jug; return upright at a fixed 45 deg/s.
"""

import importlib.util
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytestmark = pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER")

FREQUENCY = 10.0


class FakeScene:
    """Piece i leaves the jug once the commanded tilt has reached `exits[i]` degrees."""

    def __init__(self, *, exits: tuple[float, ...]) -> None:
        self.exits = exits
        self.cube_names = [f"shrimp_{i}" for i in range(len(exits))]
        self.requested_count = 0
        self.max_tilt = 0.0

    def cube_locations(self) -> list[str]:
        return ["tray" if self.max_tilt >= e else "pitcher" for e in self.exits]

    def physical_snapshot(self) -> dict[str, Any]:
        return {"locations": self.cube_locations(), "settled": True}


class ClosedLoopFixture:
    """A static-method container, never instantiated."""

    @staticmethod
    def run(*, monkeypatch, exits: tuple[float, ...], requested: int, ramp: float) -> tuple:
        import mujoco

        from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
        from hitl_pmp.environments.pouring3d.kitchen.types import ClosedLoopPourParameters

        steps: list[tuple[str, float, float]] = []
        pauses: list[tuple[str, float]] = []

        class Stubbed(KitchenController):
            commanded: float = 0.0

            def tracked_step(self, *, center: np.ndarray, angle: float, offset: float = 0) -> None:
                del center
                steps.append((self.phase, angle, offset))
                self.scene.max_tilt = max(self.scene.max_tilt, angle)
                self.commanded = angle

            def body_pose(self, *, name: str) -> tuple[np.ndarray, Rotation]:
                del name
                return np.zeros(3), Rotation.identity()

            def hand_pose(self) -> tuple[np.ndarray, Rotation]:
                return np.zeros(3), Rotation.from_euler("x", self.commanded, degrees=True)

            def calibrate_joint_payload(self) -> None:
                pass

            def pause(self, *, phase: str, seconds: float, grip: float | None = None) -> None:
                del grip
                pauses.append((phase, seconds))

        monkeypatch.setattr(mujoco, "mj_forward", lambda *_: None)
        c = Stubbed.model_construct(
            scene=FakeScene(exits=exits),
            frequency=FREQUENCY,
            held="pitcher",
            rotation_offset=Rotation.identity(),
            model=None,
            data=None,
        )
        result = c.closed_loop_pour(
            parameters=ClosedLoopPourParameters(ramp_rate_deg_s=ramp), requested=requested
        )
        return c, result, steps, pauses


def _ramp(*, steps):
    return [(a, o) for phase, a, o in steps if phase != "feedback return upright"]


def test_the_fixed_settings_are_the_approved_ones():
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

    assert KitchenController.POUR_APPROACH_DEGREES == 80.0
    assert KitchenController.POUR_MAX_TILT_DEGREES == 95.0
    assert KitchenController.POUR_WIGGLE_M == 0.04
    assert KitchenController.POUR_RETURN_DEG_S == 45.0
    assert KitchenController.TILT_SPEED_DEG_S == 10.0


def test_the_pour_stops_the_moment_the_requested_count_has_left(*, monkeypatch):
    c, result, steps, pauses = ClosedLoopFixture.run(
        monkeypatch=monkeypatch, exits=(85.05, 88.05, 93.05), requested=2, ramp=1.0
    )
    ramp = _ramp(steps=steps)
    angles = np.array([a for a, _ in ramp])
    offsets = np.array([o for _, o in ramp])
    # Approach: 10 deg/s to 80 deg with no wiggle.
    approach = angles <= 80.0 + 1e-9
    np.testing.assert_allclose(np.diff(angles[approach]), 10.0 / FREQUENCY)
    assert np.all(offsets[approach][:-1] == 0)
    # Ramp: the learned 1 deg/s, wiggling up to 40 mm (a 1 Hz sine sampled at 10 Hz
    # peaks at sin(0.4 pi) of its amplitude, 38 mm).
    np.testing.assert_allclose(np.diff(angles[~approach]), 1.0 / FREQUENCY)
    assert np.max(np.abs(offsets)) <= 0.04 + 1e-12
    assert np.max(np.abs(offsets)) == pytest.approx(0.04 * np.sin(0.4 * np.pi))
    # The second piece left on the first waypoint past 88.05 deg; nothing more was commanded.
    assert angles.max() == pytest.approx(88.1, abs=1e-6)
    assert result["cutoff_tilt_degrees"] == angles.max()
    assert "2 left the jug" in result["cutoff_reason"]
    # Return at 45 deg/s, then the production settle.
    back = np.array([a for phase, a, _ in steps if phase == "feedback return upright"])
    assert back[-1] == 0
    np.testing.assert_allclose(-np.diff(np.r_[angles.max(), back])[:-1], 45.0 / FREQUENCY)
    assert sum(seconds for _, seconds in pauses) == pytest.approx(c.pour_settle_seconds)
    assert result["success"]
    assert (result["delivered"], result["retained"]) == (2, 1)
    assert result["parameters"] == {"ramp_rate_deg_s": 1.0}
    assert result["mode"] == "closed_loop"


def test_the_ramp_is_capped_at_95_degrees(*, monkeypatch):
    _, result, steps, _ = ClosedLoopFixture.run(
        monkeypatch=monkeypatch, exits=(99.0, 99.0, 99.0), requested=1, ramp=5.0
    )
    angles = np.array([a for a, _ in _ramp(steps=steps)])
    assert angles.max() == 95.0
    np.testing.assert_allclose(np.diff(angles[angles > 80.0 + 1e-9]), 5.0 / FREQUENCY)
    assert result["cutoff_reason"] == "reached max tilt"
    assert not result["success"]
    assert result["delivered"] == 0


def test_an_overshoot_is_a_failure_not_a_success(*, monkeypatch):
    # Two pieces leave together: the cutoff cannot take one back.
    _, result, _, _ = ClosedLoopFixture.run(
        monkeypatch=monkeypatch, exits=(86.0, 86.0, 93.0), requested=1, ramp=1.0
    )
    assert result["delivered"] == 2
    assert not result["success"]


def test_the_ramp_rate_is_bounded_to_the_approved_range():
    from pydantic import ValidationError

    from hitl_pmp.environments.pouring3d.kitchen.types import ClosedLoopPourParameters

    assert ClosedLoopPourParameters.LIMITS == (0.25, 5.0)
    for ok in (0.25, 1.0, 5.0):
        assert ClosedLoopPourParameters(ramp_rate_deg_s=ok).ramp_rate_deg_s == ok
    for bad in (0.2, 5.01, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            ClosedLoopPourParameters(ramp_rate_deg_s=bad)
