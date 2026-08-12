"""The 2-D grid probe's two pure pieces, tested without a simulator.

Everything else in that probe needs MuJoCo and is exercised by running it. What is
testable here is exactly what would silently corrupt the surface if it were wrong:

  * **`refinement_order`** decides which standoff rows run first. It exists so a
    half-finished sweep is a *coarse sweep of the whole standoff range* rather than a fine
    sweep of its bottom third -- which is the difference between a partial heatmap that
    shows the shape and one that shows an edge. If it ever dropped or repeated an index the
    finished grid would be missing rows, and a heatmap with a missing row looks like a
    measurement rather than a gap;
  * **`fit_free_flight`** turns the recorded parabola into the range the geometry
    prediction is compared against. Its ground-crossing is extrapolated to a fixed height
    on purpose (see the probe's own docstring), so it is a function of the throw and not of
    whatever the cube happened to hit first -- an easy thing to get wrong in a way that
    quietly makes bin-wall cells incomparable with open-floor ones.
"""

import itertools

import numpy as np
import pytest

from scripts.tossing3d_standoff_speed_grid import fit_free_flight, refinement_order


def test_refinement_order_is_a_permutation() -> None:
    """Every standoff row runs exactly once -- no gap in the finished grid, no duplicate.

    Swept over sizes rather than parametrized: `parametrize` injects its argument
    positionally, which this project's `max-positional-args = 0` rule forbids.
    """
    for n in (1, 2, 3, 5, 8, 33, 57, 64):
        assert sorted(refinement_order(n=n)) == list(range(n)), f"n={n}"


def test_refinement_order_takes_the_endpoints_then_the_midpoint() -> None:
    """A prefix of the schedule must span the range, or partial data shows only an edge."""
    order = refinement_order(n=33)
    assert order[:3] == [0, 32, 16]


def test_an_early_prefix_already_covers_the_whole_range() -> None:
    """After a fifth of the rows, the coarse grid's largest gap is a small fraction."""
    n = 57
    order = refinement_order(n=n)
    prefix = sorted(order[: n // 5])
    largest_gap = max(b - a for a, b in itertools.pairwise(prefix))
    assert largest_gap <= 8


def test_fit_free_flight_recovers_a_known_parabola() -> None:
    """Exact recovery on synthetic data: the fit is the measurement, so it must be exact."""
    launch_vx, launch_vz, gravity, z_start = 3.0, 4.0, -9.81, 1.2
    times = np.linspace(0.0, 0.6, 61)
    pos_x = 0.5 + launch_vx * times
    pos_z = z_start + launch_vz * times + 0.5 * gravity * times**2

    fit = fit_free_flight(times=times, pos_x=pos_x, pos_z=pos_z, ground_z=0.025)

    assert fit is not None
    assert fit.launch_vx == pytest.approx(launch_vx, abs=1e-9)
    assert fit.launch_vz == pytest.approx(launch_vz, abs=1e-9)
    assert fit.launch_elevation_deg == pytest.approx(np.degrees(np.arctan2(4.0, 3.0)), abs=1e-9)
    assert fit.residual_m == pytest.approx(0.0, abs=1e-9)

    # Ground crossing solved by hand from the same coefficients, not from the fit.
    t_impact = (-launch_vz - np.sqrt(launch_vz**2 - 2 * gravity * (z_start - 0.025))) / gravity
    assert fit.impact_t == pytest.approx(t_impact, abs=1e-9)
    assert fit.impact_x == pytest.approx(0.5 + launch_vx * t_impact, abs=1e-9)


def test_fit_free_flight_extrapolates_past_the_recorded_samples() -> None:
    """The crossing is a property of the throw, not of where the recording stopped.

    A cube that hits the bin's near wall stops being recorded well above the floor. Its
    range must still be the range it *would* have flown, or wall cells and floor cells are
    measured on two different scales and the geometry overlay compares against neither.
    """
    times = np.linspace(0.0, 0.2, 21)  # cut off long before the ground
    pos_x = 4.0 * times
    pos_z = 1.0 + 2.0 * times - 4.905 * times**2

    fit = fit_free_flight(times=times, pos_x=pos_x, pos_z=pos_z, ground_z=0.0)

    assert fit is not None
    assert fit.impact_t > times[-1]
    assert fit.impact_x > pos_x[-1]


def test_fit_free_flight_declines_a_window_too_short_to_fit() -> None:
    """Two samples cannot determine a parabola; returning a number anyway would be a lie."""
    times = np.array([0.0, 0.01])
    assert fit_free_flight(times=times, pos_x=times, pos_z=times, ground_z=0.0) is None


def test_fit_free_flight_declines_a_trajectory_that_never_reaches_the_ground() -> None:
    """No real root means no range -- report nothing rather than an imaginary crossing."""
    times = np.linspace(0.0, 0.2, 21)
    pos_z = 1.0 + 2.0 * times + 4.905 * times**2  # accelerating upward, never descends
    fit = fit_free_flight(times=times, pos_x=times, pos_z=pos_z, ground_z=0.0)
    assert fit is not None
    assert fit.impact_x is None
