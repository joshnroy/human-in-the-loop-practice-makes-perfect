"""Tests for the exact paired tests, pinned against hand-computable ground truth.

`PairedTests` computes its own exact p-values rather than quoting values produced
by scipy elsewhere (scipy is not a dependency here). That buys reproducibility but
it means the *statistics themselves* are project code that can be wrong, and two
results have already been retracted on this project for being overclaimed. So the
tests below pin every exact test against hand-computable ground truth: an
all-positive sample of size n has exactly one of 2**n sign assignments at least as
extreme in each direction, so its two-sided p is exactly 2 / 2**n, no table lookup
needed. `test_wilcoxon_matches_a_published_example` reproduces a worked textbook
case where the answer does not follow from that shortcut, and
`test_subset_sums_agrees_with_naive_enumeration` pins the meet-in-the-middle
enumeration against the brute-force one it replaced.

These moved here with `PairedTests` itself, from `test_tossingroom_reset_interval.py`.
They are unchanged: the helper is domain-agnostic and so are its tests, and leaving them
in a file named for one experiment is what let a second, weaker copy of the class survive
in `tossingroom_reset_frequency.py` unnoticed.
"""

import itertools
import math

import pytest

from analysis.practice_makes_perfect.paired_tests import PairedTests


def _naive_subset_sums(*, weights):
    return sorted(sum(combo) for combo in itertools.product(*[(0.0, w) for w in weights]))


def test_subset_sums_agrees_with_naive_enumeration():
    """The meet-in-the-middle split exists only because 2**20 direct iterations are
    too slow; it must enumerate exactly the same multiset of subset sums, including
    duplicates, or the p-values it feeds are wrong."""
    for weights in ([1.0, 2.0, 3.0], [0.5, -1.5, 2.0, 2.0, 7.25], [3.0] * 6):
        assert sorted(PairedTests._subset_sums(weights=weights)) == pytest.approx(
            _naive_subset_sums(weights=weights)
        )


def test_wilcoxon_on_an_all_positive_sample_is_exactly_two_over_two_to_the_n():
    """The one case whose exact p needs no table: every difference positive means
    exactly one of the 2**n sign assignments is this extreme in each direction."""
    for n in (5, 8, 10, 20):
        result = PairedTests.wilcoxon_signed_rank(differences=[float(i) for i in range(1, n + 1)])
        assert result.p_value == pytest.approx(2 / 2**n)
        assert result.num_zero_differences == 0


def test_sign_flip_on_an_all_positive_sample_is_exactly_two_over_two_to_the_n():
    for n in (5, 8, 10, 20):
        result = PairedTests.sign_flip(differences=[float(i) for i in range(1, n + 1)])
        assert result.p_value == pytest.approx(2 / 2**n)


def test_wilcoxon_matches_a_published_example():
    """A worked example with mixed signs, where the answer does not fall out of the
    all-positive shortcut: |d| ranks 1..8, only the rank-2 difference is negative,
    so W+ = 34, W- = 2, and the exact two-sided p is 6/256 = 0.0234375."""
    result = PairedTests.wilcoxon_signed_rank(differences=[0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert result.statistic == pytest.approx(34.0)
    assert result.p_value == pytest.approx(0.0234375)


def test_zero_differences_are_dropped_and_counted_not_silently_ignored():
    """At 14 tasks per family the gap lands on multiples of ~7pp, so exact ties are
    common. A test that quietly dropped them would report a small n as the full
    twenty."""
    result = PairedTests.wilcoxon_signed_rank(differences=[0.0, 0.0, 1.0, 2.0, 3.0])
    assert result.num_zero_differences == 2
    assert result.p_value == pytest.approx(2 / 2**3)


def test_an_all_zero_sample_is_p_one_and_needs_infinitely_many_seeds():
    assert PairedTests.wilcoxon_signed_rank(differences=[0.0] * 20).p_value == 1.0
    assert PairedTests.sign_flip(differences=[0.0] * 20).p_value == 1.0
    assert math.isinf(PairedTests.seeds_for_80_percent_power(differences=[0.0] * 20))


def test_tied_absolute_differences_get_average_ranks():
    assert PairedTests._average_ranks(values=[3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]


def test_power_calculation_matches_the_closed_form():
    """n = (z_0.975 + z_0.80)^2 * (sd/mean)^2 -- ~7.849 seeds for a one-sd effect."""
    needed = PairedTests.seeds_for_80_percent_power(differences=[0.0, 1.0, 2.0, 3.0, 4.0])
    # mean 2.0, sd 1.5811 -> 7.849 * (1.5811/2)^2 = 4.905
    assert needed == pytest.approx(4.905, abs=0.01)


def test_minimum_detectable_effect_matches_the_closed_form():
    """MDE = (z_0.975 + z_0.80) * sd / sqrt(n) -- the number that says what a null
    result could and could not have found."""
    differences = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert PairedTests.minimum_detectable_effect(differences=differences) == pytest.approx(
        2.801585 * 1.5811388 / 5**0.5, abs=1e-4
    )
