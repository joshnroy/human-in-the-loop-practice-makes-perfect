"""`analysis/tossing3d_range_and_release_angle.py` carries real logic underneath its
plotting, and all of it fails silently rather than loudly: a permutation test that returns
a plausible-looking number for any implementation, a Holm correction whose step-down
monotonicity is easy to drop, and the reset detection that decides where the figure draws
its eight vertical lines. None of these crash when wrong -- they land in an experiment log
as claims.

So these pin the properties that make the reported figures mean what the entry says they
mean: the exact 2/1024 floor, which is the reason that entry reports uncorrected p-values
and says so, and the fact that a reset is read off the discrete release *step index*
rather than off a threshold on the fraction, which is what makes a mixture of two indices
not register as a reset of its own.
"""

import numpy as np

from analysis.tossing3d_range_and_release_angle import (
    PERMUTATION_FLOOR,
    _exact_paired_permutation_p,
    _holm,
    _impact_range,
    _resting_range,
    reset_speeds,
)


def test_permutation_p_is_one_when_the_paired_difference_is_symmetric_about_zero() -> None:
    """A sample whose sign-flips reproduce itself cannot evidence against the null."""
    diffs = np.array([1.0, -1.0, 2.0, -2.0])
    assert _exact_paired_permutation_p(diffs=diffs) == 1.0


def test_permutation_p_hits_the_exact_floor_when_every_pair_moves_the_same_way() -> None:
    """All-same-sign differences are the most extreme arrangement of n=10 pairs.

    Only the all-positive and all-negative sign assignments reach the observed mean, so
    the two-sided p is exactly 2/1024 -- the floor the entry reports against.
    """
    diffs = np.arange(1.0, 11.0)
    assert _exact_paired_permutation_p(diffs=diffs) == PERMUTATION_FLOOR
    assert PERMUTATION_FLOOR == 2 / 1024


def test_permutation_floor_sits_above_the_holm_threshold_for_the_committed_grid() -> None:
    """The reason the log reports uncorrected p-values, pinned as a fact about the test.

    PR #226's grid steps 37 speeds, so 36 consecutive comparisons. If this ever stops
    holding, the entry's "the Holm column cannot fire" caveat has become wrong.
    """
    n_steps = 36
    assert 0.05 / n_steps < PERMUTATION_FLOOR


def test_permutation_p_is_symmetric_under_negating_the_differences() -> None:
    """Two-sided means the test cannot care which arm was labelled first."""
    diffs = np.array([0.4, -0.1, 0.3, 0.9, 0.2, 0.7, -0.3, 0.5, 0.6, 0.1])
    assert _exact_paired_permutation_p(diffs=diffs) == _exact_paired_permutation_p(diffs=-diffs)


def test_holm_scales_the_smallest_p_by_the_full_family_size() -> None:
    adjusted = _holm(pvalues=np.array([0.01, 0.02, 0.03, 0.04]))
    assert adjusted[0] == 0.04


def test_holm_is_monotone_non_decreasing_in_the_sorted_p_values() -> None:
    """The step-down property: a larger raw p can never adjust to a smaller one."""
    raw = np.array([0.001, 0.2, 0.04, 0.5, 0.009])
    adjusted = _holm(pvalues=raw)
    assert list(np.argsort(raw)) == list(np.argsort(adjusted, kind="stable"))
    assert np.all(np.diff(np.sort(adjusted)) >= 0)


def test_holm_never_exceeds_one() -> None:
    assert np.all(_holm(pvalues=np.array([0.4, 0.6, 0.9, 0.95])) <= 1.0)


def test_ranges_are_measured_from_the_base_not_from_the_world_origin() -> None:
    """Both distances are relative to the robot, which is what "how far it goes" means."""
    row = {
        "ballistic_impact_x": 2.0,
        "cube_x_final": 2.1,
        "base_x_before_toss": 0.65,
    }
    assert _impact_range(r=row) == 1.35
    assert abs(_resting_range(r=row) - 1.45) < 1e-12


def test_ranges_are_none_when_the_cell_is_missing_a_component() -> None:
    """An errored cell must drop out rather than be silently read as a distance of 0."""
    assert _impact_range(r={"ballistic_impact_x": None, "base_x_before_toss": 0.65}) is None
    assert _resting_range(r={"cube_x_final": 2.1, "base_x_before_toss": None}) is None


def test_a_reset_is_a_drop_in_the_release_step_index() -> None:
    """The release quantisation is discrete; a reset is the index falling by one."""
    index = np.array([[8, 8, 7, 7, 6]] * 10, dtype=float)
    assert reset_speeds(index=index, speeds=[100.0, 105.0, 110.0, 115.0, 120.0]) == [110.0, 120.0]


def test_the_index_rising_is_not_a_reset() -> None:
    index = np.array([[6, 7, 8]] * 10, dtype=float)
    assert reset_speeds(index=index, speeds=[100.0, 105.0, 110.0]) == []


def test_a_split_population_resets_on_its_majority_not_on_its_mean() -> None:
    """At 120 and 145 deg/s the seeds straddle two indices; the modal index is what counts.

    A mean would slide fractionally between the two and could register a reset at the
    *following* speed, where the minority subgroup drops out but nothing actually changed
    about the majority's release. That artifact is exactly what put "150" into circulation
    as a reset speed when the measured drop is at 145.
    """
    # 7/10 seeds at index 7, 3/10 at index 8, then all at 7: the modal index never falls.
    split = [7] * 7 + [8] * 3
    index = np.array([[s, 7] for s in split], dtype=float)
    assert reset_speeds(index=index, speeds=[145.0, 150.0]) == []
