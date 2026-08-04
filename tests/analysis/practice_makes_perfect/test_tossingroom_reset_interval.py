"""Tests for the Tossing Room reset-*interval* analysis.

Two jobs, and the first matters more than usual on this project.

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
enumeration this file added for n = 20 against the brute-force one it replaced.

The second job is the usual one (see `test_ballring_sampler_iters.py`): pin the
analysis to the *real* committed aggregate, so no number quoted in the experiment
log or the PR can drift away from the data that produced it without a test
failing.
"""

import itertools
import math
import statistics
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroom_reset_interval import (
    _ARM_INTERVAL,
    PairedTests,
    ResetIntervalReport,
    expected_denominators,
)

_ARMS_JSON = (
    Path(__file__).parents[3] / "docs/experiment-logs/2026-08-04-tossingroom-reset-interval.json"
)


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


def test_expected_resets_matches_the_designed_table():
    """The four arms' intended reset counts, which the manipulation check compares
    the realised counts against: 25 cycles x (100 // interval)."""
    assert {arm: ResetIntervalReport.expected_resets(arm=arm) for arm in _ARM_INTERVAL} == {
        "armA": 250,
        "armB": 100,
        "armC": 50,
        "armD": 25,
    }


def test_predicted_gap_noise_at_the_designed_composition_is_the_quoted_floor():
    """18.9pp is the number the log reports as this design's noise floor; it comes
    from 14 TRASH and 14 RECYCLING tasks, so it moves if the composition does."""
    arms = {
        arm: {
            "0": {
                "resets": 0,
                "families": {
                    family: [[0, 0, count]] for family, count in expected_denominators().items()
                },
            }
        }
        for arm in _ARM_INTERVAL
    }
    assert ResetIntervalReport.predicted_gap_noise(arms=arms, arm="armA") == pytest.approx(
        18.898, abs=0.01
    )


def test_family_labels_cover_every_goal_the_committed_aggregate_contains():
    """Every arm/seed/sweep must carry all three families with a non-zero
    denominator -- an unrecognised goal raises during aggregation, but a family
    that silently vanished from the test draw would not."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for arm, seeds in arms.items():
        for seed, record in seeds.items():
            assert set(record["families"]) == {"RECYCLING", "TRASH", "EMPTY"}
            for family, triples in record["families"].items():
                for _transitions, solved, total in triples:
                    assert total > 0, f"{arm}/{seed} has no {family} tasks"
                    assert 0 <= solved <= total


def test_every_arm_shares_the_same_seeds_so_pairing_is_valid():
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    seeds = ResetIntervalReport.seeds(arms=arms)
    assert seeds == [str(seed) for seed in range(20)]
    for arm in arms:
        assert sorted(arms[arm], key=int) == seeds


def test_the_committed_data_shows_the_manipulation_actually_happened():
    """The experiment's first claim, pinned against its own data: each arm really
    performed the number of free resets its interval implies. A design that
    silently did not vary what it claimed is how the previous reset experiment went
    wrong."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for arm in _ARM_INTERVAL:
        expected = ResetIntervalReport.expected_resets(arm=arm)
        assert set(ResetIntervalReport.reset_counts(arms=arms, arm=arm)) == {expected}


def test_the_committed_data_has_the_designed_test_set_composition():
    """14 TRASH / 14 RECYCLING / 2 EMPTY in every seed of every arm -- asserted, not
    assumed, because it is what sets this experiment's noise floor."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    assert ResetIntervalReport.composition_violations(arms=arms) == []


def test_every_arm_got_the_same_experience():
    """2500 online transitions everywhere. Not automatic: a mid-period reset can
    revive a robot that would otherwise have raised InteractionComplete, so a
    frequently-reset arm could buy extra experience and reintroduce the progress
    confound."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for arm in _ARM_INTERVAL:
        assert set(ResetIntervalReport.achieved_transitions(arms=arms, arm=arm)) == {2500.0}


def test_the_headline_numbers_quoted_in_the_log_come_from_the_committed_data():
    """Pins the experiment log's and PR's headline figures to the aggregate that
    produced them, so no quoted number can drift without a test failing.

    The final-gap null is the pre-specified result; the area-under-curve speed-up
    is the post-hoc one. Both are pinned, since both are quoted."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    extremes = [
        d - a
        for a, d in zip(
            ResetIntervalReport.gaps(arms=arms, arm="armA"),
            ResetIntervalReport.gaps(arms=arms, arm="armD"),
            strict=True,
        )
    ]
    assert statistics.mean(extremes) == pytest.approx(0.36, abs=0.01)
    assert PairedTests.wilcoxon_signed_rank(differences=extremes).p_value == pytest.approx(
        0.9531, abs=0.0001
    )

    speedups = {}
    for family in ("RECYCLING", "TRASH", "EMPTY"):
        speedups[family] = [
            a - d
            for a, d in zip(
                ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armA", family=family),
                ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armD", family=family),
                strict=True,
            )
        ]
    assert statistics.mean(speedups["RECYCLING"]) == pytest.approx(18.4, abs=0.05)
    assert statistics.mean(speedups["TRASH"]) == pytest.approx(11.1, abs=0.05)
    # The control does not move at all -- the reason the speed-up is attributable
    # to the stochastic Throw rather than to some generic harness artifact.
    assert speedups["EMPTY"] == [0.0] * 20
    for family in ("RECYCLING", "TRASH"):
        assert PairedTests.wilcoxon_signed_rank(differences=speedups[family]).p_value < 0.001


def test_the_irreversibility_specific_claim_is_reported_as_unestablished():
    """The differential -- does the terminal family gain MORE than the recoverable
    one? -- is the claim that would make this about irreversibility rather than
    about wasted traversal. It does not reach p < 0.05, and this pins that, so the
    log cannot quietly start asserting it."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    differential = [
        (a_rec - d_rec) - (a_tra - d_tra)
        for a_rec, d_rec, a_tra, d_tra in zip(
            ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armA", family="RECYCLING"),
            ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armD", family="RECYCLING"),
            ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armA", family="TRASH"),
            ResetIntervalReport.mean_rate_over_training(arms=arms, arm="armD", family="TRASH"),
            strict=True,
        )
    ]
    assert statistics.mean(differential) == pytest.approx(7.3, abs=0.05)
    p_value = PairedTests.wilcoxon_signed_rank(differences=differential).p_value
    assert p_value == pytest.approx(0.0623, abs=0.0001)
    assert p_value > 0.05, "the log reports this as not established"
    assert PairedTests.seeds_for_80_percent_power(differences=differential) == pytest.approx(
        49, abs=1
    )


def test_every_arm_is_measured_on_an_identical_transition_grid():
    """What the post-hoc area-under-curve comparison rests on: an unweighted mean
    over checkpoints is only comparable across arms if checkpoint i sits at the
    same transition count in every arm.

    Equal *totals* would not be enough -- a run whose period ended early and whose
    later periods made the steps up could land on 2500 while sampling the curve at
    different x -- so the whole grid is checked, not just its endpoint."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    grids = {
        tuple(transitions for transitions, _solved, _total in triples)
        for seeds in arms.values()
        for record in seeds.values()
        for triples in record["families"].values()
    }
    assert grids == {tuple(range(0, 2600, 100))}


def test_the_arms_are_progress_matched_which_is_what_pr_39_could_not_achieve():
    """The methodological headline, pinned: unlike the reset-*frequency* design,
    these arms end at the same competence, so a cross-arm gap difference would have
    been attributable to reset frequency."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for family in ("TRASH", "RECYCLING"):
        differences = ResetIntervalReport.family_differences(
            arms=arms, family=family, from_arm="armA", to_arm="armD"
        )
        assert abs(statistics.mean(differences)) < 5.0
        assert PairedTests.wilcoxon_signed_rank(differences=differences).p_value > 0.05
    assert (
        ResetIntervalReport.family_differences(
            arms=arms, family="EMPTY", from_arm="armA", to_arm="armD"
        )
        == [0.0] * 20
    )


def test_the_final_gap_null_is_measured_at_a_ceiling_not_at_precision():
    """The log's most important caveat, pinned: every arm's observed gap sd is
    BELOW the binomial noise floor, which is saturation rather than precision --
    most seeds have both throw families at 100% and a gap of exactly zero."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for arm in _ARM_INTERVAL:
        gaps = ResetIntervalReport.gaps(arms=arms, arm=arm)
        floor = ResetIntervalReport.predicted_gap_noise(arms=arms, arm=arm)
        assert statistics.stdev(gaps) < floor
        assert sum(1 for gap in gaps if gap == 0.0) >= 13
