"""The impact probe's pure arithmetic, tested without a simulator.

`ballistic_ground_crossing` is what turns a recorded flight into the number the whole
range-vs-speed fit is built on, and `longest_contact_free_window` is what decides which
samples that fit sees. Both are pure, so both are testable here; everything else in the
probe needs MuJoCo and is exercised by running it.
"""

import numpy as np
import pytest

from scripts.tossing3d_toss_impact_probe import (
    GROUND_CUBE_CENTRE_Z,
    ballistic_ground_crossing,
    longest_contact_free_window,
)

GRAVITY = 9.81


def _synthetic_flight(
    *,
    x0: float,
    z0: float,
    vx: float,
    vz: float,
    t_start: float = 0.0,
    duration: float = 0.5,
    dt: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An exact drag-free parabola, sampled the way the probe samples the simulator."""
    times = np.arange(t_start, t_start + duration, dt)
    rel = times - t_start
    return times, x0 + vx * rel, z0 + vz * rel - 0.5 * GRAVITY * rel**2


def test_ballistic_ground_crossing_recovers_a_known_landing_point() -> None:
    """The analytic crossing of a synthetic parabola is recovered to sub-millimetre."""
    x0, z0, vx, vz = 1.0, 0.6, 2.6, 1.2
    times, xs, zs = _synthetic_flight(x0=x0, z0=z0, vx=vx, vz=vz)

    impact_x, impact_t, residual, _vx, _vz = ballistic_ground_crossing(times=times, xs=xs, zs=zs)

    # Solve 0.5*g*t^2 - vz*t + (GROUND - z0) = 0 for the descending root by hand.
    a, b, c = -0.5 * GRAVITY, vz, z0 - GROUND_CUBE_CENTRE_Z
    expected_t = (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert impact_t == pytest.approx(expected_t, abs=1e-9)
    assert impact_x == pytest.approx(x0 + vx * expected_t, abs=1e-6)
    assert residual < 1e-9


def test_ballistic_ground_crossing_takes_the_descending_root() -> None:
    """A flight that starts *below* the ground height still lands in front of the throw.

    The parabola crosses the ground height twice; the ascending crossing is behind the
    release point. Taking the wrong root would report a landing that is metres short and
    would still look like a plausible number.
    """
    times, xs, zs = _synthetic_flight(x0=0.0, z0=0.0, vx=3.0, vz=2.0)

    impact_x, impact_t, _, _, _ = ballistic_ground_crossing(times=times, xs=xs, zs=zs)

    assert impact_t > 0.0
    assert impact_x > 0.0


def test_ballistic_ground_crossing_rejects_a_window_that_is_not_free_flight() -> None:
    """An upward-curving window is not a ballistic arc, and must raise rather than fit.

    This is the guard that stops a cell whose "free flight" window actually caught part of
    the swing from contributing a silently wrong number to the fit.
    """
    times = np.linspace(0.0, 0.5, 100)
    zs = 0.3 + 0.5 * times**2  # curving the wrong way
    xs = times * 2.0

    with pytest.raises(ValueError, match="not a downward parabola"):
        ballistic_ground_crossing(times=times, xs=xs, zs=zs)


def test_ballistic_ground_crossing_needs_at_least_three_samples() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        ballistic_ground_crossing(
            times=np.array([0.0, 0.1]), xs=np.array([0.0, 1.0]), zs=np.array([1.0, 0.9])
        )


def test_longest_contact_free_window_prefers_the_longest_run_not_the_first() -> None:
    """The gripper's release produces a short contact-free blip before the real flight.

    Taking the first run would fit the parabola to three samples of gripper opening.
    """
    contacted = [True, True, False, True, True, False, False, False, False, True, True]

    start, stop = longest_contact_free_window(contacted=contacted)

    assert (start, stop) == (5, 9)


def test_longest_contact_free_window_handles_a_run_reaching_the_end() -> None:
    """A cube still airborne when recording stops has its window closed at the end."""
    contacted = [True, False, False, False]

    assert longest_contact_free_window(contacted=contacted) == (1, 4)


def test_longest_contact_free_window_on_an_all_contact_recording_is_empty() -> None:
    """A throw that never released leaves no window, and must not look like a 1-sample one."""
    start, stop = longest_contact_free_window(contacted=[True, True, True])

    assert stop - start == 0
