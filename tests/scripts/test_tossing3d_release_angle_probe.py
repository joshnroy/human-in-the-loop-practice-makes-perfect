"""The release probe's quantisation rule, tested without a simulator.

`release_index_and_fraction` is the whole mechanism the release-angle figure reports: which
control step the gripper opens on, and therefore what fraction of the swing the arm has
covered when it lets go. Everything else in the probe needs MuJoCo and is exercised by
running it.

The rule has one detail that is easy to get wrong and impossible to notice: the denominator
is the trapezoidal profile's **last sample**, not the `total_dist` the profile was asked
for. The profile's time grid overshoots the motion's duration, so the two differ -- and
using the wrong one shifts the realised fraction enough to move which step the release
lands on, which moves every angle in the figure.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.tossing3d_release_angle_probe import (
    RELEASE_FRACTION,
    assert_kinder_pins,
    longest_contact_free_window,
    release_index_and_fraction,
)


# Both stand in for `TossController.reset`, so both must mirror upstream's *positional*
# signature -- that is exactly what `assert_kinder_pins` inspects, and rewriting them
# keyword-only would test a signature no KINDER pin has ever had.
def _reset_with_speed(self: object, x: object, params: object, release_speed: float = 1.0) -> None:  # noqa: PLR0917
    """A `reset` from a pin at or after 1b564a1."""


def _reset_without_speed(self: object, x: object, params: object) -> None:  # noqa: PLR0917
    """A `reset` from a pin before 1b564a1, where the dial does not exist."""


def test_the_probe_refuses_a_kinder_models_from_another_checkout() -> None:
    """The silent-skew guard: a different checkout's pin would measure a different toss."""
    with pytest.raises(RuntimeError, match="outside this checkout"):
        assert_kinder_pins(
            kinder_models=SimpleNamespace(__file__="/somewhere/else/kinder_models/__init__.py"),
            toss_controller=SimpleNamespace(reset=_reset_with_speed),
        )


def test_the_probe_refuses_a_pin_whose_toss_has_no_release_speed() -> None:
    """On a pin older than 1b564a1 every cell would silently run the default 140 deg/s."""
    with pytest.raises(RuntimeError, match="no release_speed parameter"):
        assert_kinder_pins(
            kinder_models=SimpleNamespace(__file__=__file__),
            toss_controller=SimpleNamespace(reset=_reset_without_speed),
        )


def test_a_correctly_pinned_kinder_inside_this_checkout_is_accepted() -> None:
    """The guard must not be so strict that the real, correct setup fails to start."""
    assert_kinder_pins(
        kinder_models=SimpleNamespace(__file__=__file__),
        toss_controller=SimpleNamespace(reset=_reset_with_speed),
    )


def test_free_flight_is_the_longest_contact_free_run_not_the_first() -> None:
    """The gripper opening produces brief contact-free moments before the throw proper."""
    contacted = [True, False, True, True, False, False, False, False, True]
    assert longest_contact_free_window(contacted=contacted) == (4, 8)


def test_a_recording_that_never_touches_anything_is_one_window() -> None:
    assert longest_contact_free_window(contacted=[False] * 5) == (0, 5)


def test_a_recording_always_in_contact_has_an_empty_window() -> None:
    """An empty window must be empty rather than a spurious one-sample flight."""
    start, stop = longest_contact_free_window(contacted=[True] * 5)
    assert stop - start == 0


def test_release_is_the_first_sample_at_or_past_the_target_fraction() -> None:
    """`fraction_covered >= 0.46` fires on the first sample that reaches it, not the nearest."""
    trajectory = np.array([0.0, 0.2, 0.4, 0.5, 0.8, 1.0])
    index, fraction, end = release_index_and_fraction(trajectory=trajectory)
    assert index == 3
    assert fraction == 0.5
    assert end == 1.0


def test_the_denominator_is_the_profiles_last_sample_not_the_requested_distance() -> None:
    """The profile overshoots its own total; normalising by the wrong one moves the step.

    This trajectory ends at 1.10 while a caller would have asked for 1.00. Against 1.10 the
    first sample past 0.46 is index 3 (0.50/1.10 = 0.4545 is short, 0.55/1.10 = 0.50 is
    not). Against 1.00 it would have been index 2. One step earlier is a different throw.
    """
    trajectory = np.array([0.0, 0.25, 0.50, 0.55, 0.90, 1.10])
    index, fraction, _ = release_index_and_fraction(trajectory=trajectory)
    assert index == 3
    assert abs(fraction - 0.5) < 1e-12


def test_a_sample_landing_exactly_on_the_threshold_releases_there() -> None:
    """Measured live: at 145 deg/s one seed realises 0.46000022, right on the knife edge."""
    trajectory = np.array([0.0, 0.2, RELEASE_FRACTION, 1.0])
    index, fraction, _ = release_index_and_fraction(trajectory=trajectory)
    assert index == 2
    assert fraction == RELEASE_FRACTION


def test_the_realised_fraction_is_never_below_the_target() -> None:
    """The rule is `>=`, so overshoot is possible and undershoot is not -- for any profile."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        trajectory = np.sort(rng.uniform(0.0, 5.0, size=rng.integers(4, 40)))
        _, fraction, _ = release_index_and_fraction(trajectory=trajectory)
        assert fraction >= RELEASE_FRACTION


def test_a_coarser_profile_never_releases_earlier_in_the_swing() -> None:
    """The mechanism behind the sawtooth: fewer control steps means a coarser sampling.

    Coarsening cannot move the release *earlier* in path terms -- a sample grid that is a
    subset of a finer one can only skip past the threshold, never land short of it. It
    often lands in the same place, so the property is `>=` and not `>`; the strict case is
    covered separately below. This is why raising the commanded speed pushes the realised
    fraction up until the step index drops and it resets.
    """
    fine = np.linspace(0.0, 1.0, 21)
    coarse = np.linspace(0.0, 1.0, 5)
    _, fine_fraction, _ = release_index_and_fraction(trajectory=fine)
    _, coarse_fraction, _ = release_index_and_fraction(trajectory=coarse)
    assert coarse_fraction >= fine_fraction
    # Both grids happen to carry a sample at exactly 0.5, so this pair is the equality case.
    assert coarse_fraction == fine_fraction == 0.5


def test_coarsening_far_enough_strictly_overshoots_the_target() -> None:
    """Four samples straddle 0.46 so widely that the release lands at 0.667 of the path."""
    _, fraction, _ = release_index_and_fraction(trajectory=np.linspace(0.0, 1.0, 4))
    assert abs(fraction - 2 / 3) < 1e-12
