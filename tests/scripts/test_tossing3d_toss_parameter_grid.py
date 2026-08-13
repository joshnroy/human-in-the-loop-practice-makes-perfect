"""The grid probe's pure pieces, tested without a simulator.

Everything else in that probe needs MuJoCo and is exercised by running it. What is
testable here is exactly what would silently corrupt the surface if it were wrong:

  * **`refinement_order`** decides which rows of the outer axis run first. It exists so a
    half-finished sweep is a *coarse sweep of the whole range* rather than a fine
    sweep of its bottom third -- which is the difference between a partial heatmap that
    shows the shape and one that shows an edge. If it ever dropped or repeated an index the
    finished grid would be missing rows, and a heatmap with a missing row looks like a
    measurement rather than a gap;
  * **`fit_free_flight`** turns the recorded parabola into the ballistic distance that is
    this probe's primary criterion. Its ground-crossing is extrapolated to a fixed height
    on purpose (see the probe's own docstring), so it is a function of the throw and not of
    whatever the cube happened to hit first -- an easy thing to get wrong in a way that
    quietly makes bin-wall cells incomparable with open-floor ones;
  * **`linear_axis`** builds each parameter axis. A 20-point axis over `(60, 140)` has an
    irrational step, so the `start/stop/step` form this replaces could not express it
    without the endpoint drifting off the bound the sampler actually draws from. An axis
    whose top is 139.99 rather than 140.0 sweeps a range that is not the shipping one.

The axis **defaults** are tested against `predicates` rather than against literals, for the
reason the probe's own docstring gives: a sweep that hardcodes its bounds can silently
disagree with the interval the sampler draws from, and two of this task's three prior
measurement errors were exactly that (a 240 that was never shipped, and a 723 that was
never the default).
"""

import inspect
import itertools

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.predicates import (
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_SPEED_BOUNDS,
)
from scripts.tossing3d_release_angle_probe import assert_kinder_pins
from scripts.tossing3d_toss_parameter_grid import (
    DEFAULT_RELEASE_MS_START,
    DEFAULT_RELEASE_MS_STOP,
    DEFAULT_SPEED_START,
    DEFAULT_SPEED_STOP,
    fit_free_flight,
    linear_axis,
    refinement_order,
)


def test_the_speed_axis_defaults_to_the_interval_the_sampler_draws_from() -> None:
    """Read from `predicates`, never retyped -- a retyped 240 is what #239 had to correct."""
    assert (DEFAULT_SPEED_START, DEFAULT_SPEED_STOP) == TOSS_SPEED_BOUNDS


def test_the_release_ms_axis_defaults_to_the_interval_the_sampler_draws_from() -> None:
    """Same rule for the second dial, and for the same reason."""
    assert (DEFAULT_RELEASE_MS_START, DEFAULT_RELEASE_MS_STOP) == TOSS_RELEASE_MS_BOUNDS


def test_linear_axis_hits_both_endpoints_exactly() -> None:
    """A 20-point axis over an irrational step must still *end* on the bound."""
    axis = linear_axis(start=60.0, stop=140.0, points=20)
    assert len(axis) == 20
    assert axis[0] == 60.0
    assert axis[-1] == 140.0


def test_linear_axis_is_evenly_spaced_and_strictly_increasing() -> None:
    """Uneven spacing would make a heatmap's cells lie about which parameters they cover.

    The tolerance is `2e-6` rather than something tighter because the values are
    deliberately rounded to 6 decimals so they survive a JSON round trip as dict keys (see
    `linear_axis`). Two such roundings can each move an endpoint by 5e-7, so a *gap* can
    differ from its neighbour by up to 1e-6 -- and that is the design, not slop. Anything
    beyond it would mean the axis is genuinely uneven.
    """
    axis = linear_axis(start=300.0, stop=1400.0, points=20)
    gaps = [b - a for a, b in itertools.pairwise(axis)]
    assert min(gaps) > 0.0
    assert max(gaps) - min(gaps) <= 2e-6


def test_linear_axis_of_one_point_is_the_start() -> None:
    """A held-fixed axis is `points=1`, which is how the standoff is pinned for this grid."""
    assert linear_axis(start=1.35, stop=1.35, points=1) == [1.35]


def test_axis_values_round_trip_as_json_dict_keys() -> None:
    """The analysis indexes cells by their float axis value, so the values must compare equal.

    `60 + 3 * 80/19` recomputed is not bit-identical to the value written into the JSON
    unless both go through the same rounding, and a mismatch silently drops a column.
    """
    import json

    axis = linear_axis(start=60.0, stop=140.0, points=20)
    assert json.loads(json.dumps(axis)) == axis


def test_assert_kinder_pins_requires_the_gripper_release_millisecond() -> None:
    """The exact silent failure this whole grid is exposed to: a pin without the second dial.

    `release_speed` alone is not enough any more. At a pin that has `release_speed` but no
    `gripper_release_ms`, every cell of the millisecond axis runs the same throw and the
    sweep is 20 copies of one column -- plausible numbers, no error. The guard therefore has
    to test the capability that is actually depended on, which is now both parameters.
    """

    class _OnlySpeed:
        @staticmethod
        def reset(*, release_speed: float = 140.0) -> None:
            """A pin older than kb#12: the speed dial only."""

    import scripts.tossing3d_release_angle_probe as probe

    fake_models = type("m", (), {"__file__": str(probe._REPO_ROOT / "reference" / "x" / "y.py")})

    with pytest.raises(RuntimeError, match="gripper_release_ms"):
        assert_kinder_pins(kinder_models=fake_models, toss_controller=_OnlySpeed)

    # And the guard still passes on a pin that has both, so it is not simply always raising.
    class _Both:
        @staticmethod
        def reset(*, release_speed: float = 140.0, gripper_release_ms: int = 720) -> None:
            """The pin this grid requires."""

    assert_kinder_pins(kinder_models=fake_models, toss_controller=_Both)
    assert "gripper_release_ms" in inspect.signature(_Both.reset).parameters


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
