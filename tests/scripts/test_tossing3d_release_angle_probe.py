"""The release probe's quantisation rule -- which control step the gripper opens on, and
what fraction of the swing that is -- tested without a simulator.

The detail that is easy to get wrong: the denominator is the trapezoidal profile's last
sample, not the `total_dist` it was asked for, and the two differ because the time grid
overshoots. Using the wrong one moves which step the release lands on.
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


# These stand in for `TossController.reset`, so they mirror upstream's *positional*
# signature, which is what `assert_kinder_pins` inspects.
def _reset_with_both_dials(  # noqa: PLR0917
    self: object,
    x: object,
    params: object,
    release_speed: float = 1.0,
    gripper_release_ms: int = 720,
) -> None:
    """A `reset` from a pin at or after kb#12 (88b5eb3): both dials present."""


def _reset_with_speed(self: object, x: object, params: object, release_speed: float = 1.0) -> None:  # noqa: PLR0917
    """A `reset` from a pin at or after 1b564a1 but before kb#12: the speed dial only."""


def _reset_without_speed(self: object, x: object, params: object) -> None:  # noqa: PLR0917
    """A `reset` from a pin before 1b564a1, where neither dial exists."""


def test_the_probe_refuses_a_kinder_models_from_another_checkout() -> None:
    with pytest.raises(RuntimeError, match="outside this checkout"):
        assert_kinder_pins(
            kinder_models=SimpleNamespace(__file__="/somewhere/else/kinder_models/__init__.py"),
            toss_controller=SimpleNamespace(reset=_reset_with_both_dials),
        )


def test_the_probe_refuses_a_pin_whose_toss_has_no_release_speed() -> None:
    """On a pin older than 1b564a1 every cell would silently run the default 140 deg/s."""
    with pytest.raises(RuntimeError, match="no release_speed parameter"):
        assert_kinder_pins(
            kinder_models=SimpleNamespace(__file__=__file__),
            toss_controller=SimpleNamespace(reset=_reset_without_speed),
        )


def test_the_probe_refuses_a_pin_whose_toss_has_no_gripper_release_millisecond() -> None:
    """On a pin before kb#12 every cell of a millisecond axis is the same throw."""
    with pytest.raises(RuntimeError, match="no gripper_release_ms parameter"):
        assert_kinder_pins(
            kinder_models=SimpleNamespace(__file__=__file__),
            toss_controller=SimpleNamespace(reset=_reset_with_speed),
        )


def test_a_correctly_pinned_kinder_inside_this_checkout_is_accepted() -> None:
    assert_kinder_pins(
        kinder_models=SimpleNamespace(__file__=__file__),
        toss_controller=SimpleNamespace(reset=_reset_with_both_dials),
    )


def test_free_flight_is_the_longest_contact_free_run_not_the_first() -> None:
    """The gripper opening produces brief contact-free moments before the throw proper."""
    contacted = [True, False, True, True, False, False, False, False, True]
    assert longest_contact_free_window(contacted=contacted) == (4, 8)


def test_a_recording_that_never_touches_anything_is_one_window() -> None:
    assert longest_contact_free_window(contacted=[False] * 5) == (0, 5)


def test_a_recording_always_in_contact_has_an_empty_window() -> None:
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
    """This trajectory ends at 1.10 while a caller would have asked for 1.00. Against 1.10
    the first sample past 0.46 is index 3; against 1.00 it would have been index 2."""
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
    """A coarser sample grid can only skip past the threshold, never land short of it, so
    the property is `>=` and not `>`. The strict case is covered below."""
    fine = np.linspace(0.0, 1.0, 21)
    coarse = np.linspace(0.0, 1.0, 5)
    _, fine_fraction, _ = release_index_and_fraction(trajectory=fine)
    _, coarse_fraction, _ = release_index_and_fraction(trajectory=coarse)
    assert coarse_fraction >= fine_fraction
    # Both grids carry a sample at exactly 0.5, so this pair is the equality case.
    assert coarse_fraction == fine_fraction == 0.5


def test_coarsening_far_enough_strictly_overshoots_the_target() -> None:
    """Four samples straddle 0.46 so widely that the release lands at 0.667 of the path."""
    _, fraction, _ = release_index_and_fraction(trajectory=np.linspace(0.0, 1.0, 4))
    assert abs(fraction - 2 / 3) < 1e-12
