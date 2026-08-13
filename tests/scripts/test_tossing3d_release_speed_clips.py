"""The release-speed clip probe's pure arithmetic: the recorded trajectory turned into
"how far the cube went". Nothing here starts MuJoCo."""

import numpy as np
import pytest

from scripts.tossing3d_release_speed_clips import (
    CUBE_RESTING_HALF_HEIGHT,
    DEFAULT_SPEED_COUNT,
    DEFAULT_SPEED_START,
    DEFAULT_SPEED_STOP,
    ballistic_ground_crossing,
    evenly_spaced_speeds,
)


def test_evenly_spaced_speeds_spans_the_whole_range_inclusive() -> None:
    speeds = evenly_spaced_speeds(start=60.0, stop=240.0, count=5)
    assert speeds == [60.0, 105.0, 150.0, 195.0, 240.0]
    gaps = {round(b - a, 9) for a, b in zip(speeds[:-1], speeds[1:], strict=True)}
    assert gaps == {45.0}


def test_the_default_speeds_are_all_on_pr_227s_five_deg_grid() -> None:
    """#227's distance grid stepped 5 deg/s, so a default off that grid leaves its clip
    with no measured cell to be checked against."""
    speeds = evenly_spaced_speeds(
        start=DEFAULT_SPEED_START, stop=DEFAULT_SPEED_STOP, count=DEFAULT_SPEED_COUNT
    )
    assert all(speed % 5.0 == 0.0 for speed in speeds), speeds


def test_evenly_spaced_speeds_refuses_fewer_than_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        evenly_spaced_speeds(start=60.0, stop=240.0, count=1)


def test_ballistic_ground_crossing_recovers_a_synthetic_parabola_exactly() -> None:
    """The expected crossing is solved in closed form, independently of the code under
    test."""
    g, z0, vz, x0, vx = 9.81, 0.30, 3.0, 0.40, 2.0
    times = np.linspace(0.0, 0.8, 161)
    zs = z0 + vz * times - 0.5 * g * times**2
    xs = x0 + vx * times
    # Descending root of z(t) = h.
    h = CUBE_RESTING_HALF_HEIGHT
    expected_t = (vz + np.sqrt(vz**2 - 2 * g * (h - z0))) / g

    crossing = ballistic_ground_crossing(times=times, xs=xs, zs=zs)

    assert crossing.t == pytest.approx(expected_t, abs=1e-9)
    assert crossing.x == pytest.approx(x0 + vx * expected_t, abs=1e-9)
    assert crossing.residual_m < 1e-9
    # Launch velocities, not arrival ones.
    assert crossing.launch_vx == pytest.approx(vx, abs=1e-9)
    assert crossing.launch_vz == pytest.approx(vz, abs=1e-9)


def test_ballistic_ground_crossing_takes_the_descending_root_not_the_ascending_one() -> None:
    """Launched from *below* the resting height, both roots are real and positive; the
    ascending one is the cube first passing 0.025 m going up."""
    g, z0, vz, x0, vx = 9.81, 0.0, 2.0, 0.0, 1.0
    times = np.linspace(0.0, 0.45, 91)
    zs = z0 + vz * times - 0.5 * g * times**2
    xs = x0 + vx * times

    crossing = ballistic_ground_crossing(times=times, xs=xs, zs=zs)

    ascending = (vz - np.sqrt(vz**2 - 2 * g * (CUBE_RESTING_HALF_HEIGHT - z0))) / g
    descending = (vz + np.sqrt(vz**2 - 2 * g * (CUBE_RESTING_HALF_HEIGHT - z0))) / g
    assert crossing.t == pytest.approx(descending, abs=1e-9)
    assert crossing.t > ascending


def test_ballistic_ground_crossing_refuses_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        ballistic_ground_crossing(
            times=np.array([0.0, 0.1]), xs=np.array([0.0, 0.1]), zs=np.array([0.3, 0.3])
        )
