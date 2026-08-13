"""The grid probe's pure pieces -- `refinement_order`, `fit_free_flight`, `linear_axis` --
tested without a simulator. Everything else there needs MuJoCo.

Axis defaults are asserted against `predicates` rather than literals: two of this task's
three prior measurement errors were retyped bounds (a 240 that was never shipped, a 723
that was never the default).
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
    assert (DEFAULT_SPEED_START, DEFAULT_SPEED_STOP) == TOSS_SPEED_BOUNDS


def test_the_release_ms_axis_defaults_to_the_interval_the_sampler_draws_from() -> None:
    assert (DEFAULT_RELEASE_MS_START, DEFAULT_RELEASE_MS_STOP) == TOSS_RELEASE_MS_BOUNDS


def test_linear_axis_hits_both_endpoints_exactly() -> None:
    axis = linear_axis(start=60.0, stop=140.0, points=20)
    assert len(axis) == 20
    assert axis[0] == 60.0
    assert axis[-1] == 140.0


def test_linear_axis_is_evenly_spaced_and_strictly_increasing() -> None:
    """The `2e-6` tolerance is `linear_axis`'s 6-decimal rounding: two roundings move an
    endpoint by 5e-7 each, so a gap can differ from its neighbour by up to 1e-6."""
    axis = linear_axis(start=300.0, stop=1400.0, points=20)
    gaps = [b - a for a, b in itertools.pairwise(axis)]
    assert min(gaps) > 0.0
    assert max(gaps) - min(gaps) <= 2e-6


def test_linear_axis_of_one_point_is_the_start() -> None:
    """A held-fixed axis is `points=1`, which is how the standoff is pinned."""
    assert linear_axis(start=1.35, stop=1.35, points=1) == [1.35]


def test_axis_values_round_trip_as_json_dict_keys() -> None:
    """The analysis indexes cells by their float axis value, so a mismatch drops a column."""
    import json

    axis = linear_axis(start=60.0, stop=140.0, points=20)
    assert json.loads(json.dumps(axis)) == axis


def test_assert_kinder_pins_requires_the_gripper_release_millisecond() -> None:
    """At a pin with `release_speed` but no `gripper_release_ms`, every cell of the
    millisecond axis runs the same throw: 20 copies of one column, no error."""

    class _OnlySpeed:
        @staticmethod
        def reset(*, release_speed: float = 140.0) -> None:
            """A pin older than kb#12: the speed dial only."""

    import scripts.tossing3d_release_angle_probe as probe

    fake_models = type("m", (), {"__file__": str(probe._REPO_ROOT / "reference" / "x" / "y.py")})

    with pytest.raises(RuntimeError, match="gripper_release_ms"):
        assert_kinder_pins(kinder_models=fake_models, toss_controller=_OnlySpeed)

    # And it is not simply always raising.
    class _Both:
        @staticmethod
        def reset(*, release_speed: float = 140.0, gripper_release_ms: int = 720) -> None:
            """The pin this grid requires."""

    assert_kinder_pins(kinder_models=fake_models, toss_controller=_Both)
    assert "gripper_release_ms" in inspect.signature(_Both.reset).parameters


def test_refinement_order_is_a_permutation() -> None:
    """Swept over sizes rather than parametrized: `parametrize` injects its argument
    positionally, which this project's `max-positional-args = 0` rule forbids."""
    for n in (1, 2, 3, 5, 8, 33, 57, 64):
        assert sorted(refinement_order(n=n)) == list(range(n)), f"n={n}"


def test_refinement_order_takes_the_endpoints_then_the_midpoint() -> None:
    order = refinement_order(n=33)
    assert order[:3] == [0, 32, 16]


def test_an_early_prefix_already_covers_the_whole_range() -> None:
    n = 57
    order = refinement_order(n=n)
    prefix = sorted(order[: n // 5])
    largest_gap = max(b - a for a, b in itertools.pairwise(prefix))
    assert largest_gap <= 8


def test_fit_free_flight_recovers_a_known_parabola() -> None:
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

    # Solved from the same coefficients, not from the fit.
    t_impact = (-launch_vz - np.sqrt(launch_vz**2 - 2 * gravity * (z_start - 0.025))) / gravity
    assert fit.impact_t == pytest.approx(t_impact, abs=1e-9)
    assert fit.impact_x == pytest.approx(0.5 + launch_vx * t_impact, abs=1e-9)


def test_fit_free_flight_extrapolates_past_the_recorded_samples() -> None:
    """A cube stopped by the bin's wall must still report the range it would have flown, or
    wall cells and floor cells are measured on two different scales."""
    times = np.linspace(0.0, 0.2, 21)  # cut off long before the ground
    pos_x = 4.0 * times
    pos_z = 1.0 + 2.0 * times - 4.905 * times**2

    fit = fit_free_flight(times=times, pos_x=pos_x, pos_z=pos_z, ground_z=0.0)

    assert fit is not None
    assert fit.impact_t > times[-1]
    assert fit.impact_x > pos_x[-1]


def test_fit_free_flight_declines_a_window_too_short_to_fit() -> None:
    times = np.array([0.0, 0.01])
    assert fit_free_flight(times=times, pos_x=times, pos_z=times, ground_z=0.0) is None


def test_fit_free_flight_declines_a_trajectory_that_never_reaches_the_ground() -> None:
    """No real root means no range, rather than an imaginary crossing."""
    times = np.linspace(0.0, 0.2, 21)
    pos_z = 1.0 + 2.0 * times + 4.905 * times**2  # accelerating upward, never descends
    fit = fit_free_flight(times=times, pos_x=times, pos_z=pos_z, ground_z=0.0)
    assert fit is not None
    assert fit.impact_x is None
