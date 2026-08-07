"""Tests for the Tossing Room reset-*interval* analysis.

Pin the analysis to the *real* committed aggregate (the usual job here -- see
`test_ballring_sampler_iters.py`), so no number quoted in the experiment log or the PR
can drift away from the data that produced it without a test failing.

`PairedTests`, which supplies every exact p-value below, is tested on its own in
`test_paired_tests.py`; it moved out of this file's module when the second copy of it
was deduplicated.
"""

import importlib
import statistics
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.paired_tests import PairedTests
from analysis.practice_makes_perfect.tossingroom_reset_interval import (
    _ARM_INTERVAL,
    ResetIntervalReport,
    expected_denominators,
)

_ARMS_JSON = (
    Path(__file__).parents[3] / "docs/experiment-logs/2026-08-04-tossingroom-reset-interval.json"
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


def test_every_final_success_count_quoted_in_the_log_comes_from_the_committed_data():
    """Every success rate on this project is reported as `solved/attempted`, and
    these are the exact counts the log and the PR quote.

    Pinned as counts rather than as percentages on purpose: the counts are what
    `Metrics.breakdowns` recorded, and a percentage is a lossy rendering of them.
    `EMPTY` at 40/40 against 280 tasks per throw family is the case this makes
    visible -- as `100.0` next to `99.3` the two look like comparable evidence."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    assert {
        arm: {
            family: ResetIntervalReport.pooled_counts(arms=arms, arm=arm, family=family)
            for family in ("TRASH", "RECYCLING", "EMPTY")
        }
        for arm in _ARM_INTERVAL
    } == {
        "armA": {"TRASH": (278, 280), "RECYCLING": (273, 280), "EMPTY": (40, 40)},
        "armB": {"TRASH": (268, 280), "RECYCLING": (266, 280), "EMPTY": (40, 40)},
        "armC": {"TRASH": (274, 280), "RECYCLING": (273, 280), "EMPTY": (40, 40)},
        "armD": {"TRASH": (273, 280), "RECYCLING": (267, 280), "EMPTY": (40, 40)},
    }


def test_the_percentage_rendering_never_disagrees_with_the_count():
    """`pooled_rate` exists only as a rendering of `pooled_counts`. If the two ever
    came from different code paths, the log could quote a count and a percentage
    that do not describe the same thing."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for arm in _ARM_INTERVAL:
        for family in ("TRASH", "RECYCLING", "EMPTY"):
            solved, total = ResetIntervalReport.pooled_counts(arms=arms, arm=arm, family=family)
            assert ResetIntervalReport.pooled_rate(
                arms=arms, arm=arm, family=family
            ) == pytest.approx(100.0 * solved / total)


def test_the_area_under_the_curve_counts_agree_with_the_per_seed_means():
    """The AUC is quoted as a count (tasks solved over 26 checkpoints x 20 seeds).
    That is only the same quantity as the per-seed mean rate the paired tests use
    because every checkpoint has the same denominator -- so the two are checked
    against each other rather than one being assumed to summarise the other."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    expected = {
        "armA": {"RECYCLING": (5528, 7280), "TRASH": (5763, 7280), "EMPTY": (1040, 1040)},
        "armB": {"RECYCLING": (4967, 7280), "TRASH": (5654, 7280), "EMPTY": (1040, 1040)},
        "armC": {"RECYCLING": (4423, 7280), "TRASH": (5427, 7280), "EMPTY": (1040, 1040)},
        "armD": {"RECYCLING": (4187, 7280), "TRASH": (4957, 7280), "EMPTY": (1040, 1040)},
    }
    for arm, families in expected.items():
        for family, counts in families.items():
            solved, total = ResetIntervalReport.curve_counts(arms=arms, arm=arm, family=family)
            assert (solved, total) == counts
            per_seed = ResetIntervalReport.mean_rate_over_training(
                arms=arms, arm=arm, family=family
            )
            assert statistics.mean(per_seed) == pytest.approx(100.0 * solved / total, abs=1e-9)


def test_the_untrained_baseline_is_the_same_in_every_arm_and_is_far_from_the_ceiling():
    """Checkpoint 0 is taken before any arm has acted, so all four arms evaluate the
    same untrained policy -- which is what makes it an honest in-panel floor.

    It is also the answer to "are these tasks just easy?": an untrained EES solves
    58/280 TRASH and 55/280 RECYCLING, so the ~278/280 the arms end at is learned,
    not given. Pinned because the log and the PR both quote it as the reason no
    separate random-skills arm was run here."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    for family, counts in (("TRASH", (58, 280)), ("RECYCLING", (55, 280)), ("EMPTY", (40, 40))):
        for arm in _ARM_INTERVAL:
            assert ResetIntervalReport.untrained_counts(arms=arms, arm=arm, family=family) == counts


def test_the_presaturation_checkpoint_is_chosen_by_the_stated_resolution_rule():
    """The rule is "the last checkpoint before at least half the runs have both
    throw families at their ceiling", and nothing about any outcome enters it.

    Pinned by re-deriving the crossing from the data rather than by asserting 1600
    alone, so the test fails if the *rule* stops matching the *checkpoint*."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    index = ResetIntervalReport.presaturation_index(arms=arms)
    assert ResetIntervalReport.checkpoint_transitions(arms=arms, index=index) == 1600
    assert ResetIntervalReport.saturated_fraction(arms=arms, index=index) < 0.5
    assert ResetIntervalReport.saturated_fraction(arms=arms, index=index + 1) >= 0.5
    for earlier in range(index):
        assert ResetIntervalReport.saturated_fraction(arms=arms, index=earlier) < 0.5


def test_the_presaturation_view_is_confounded_by_progress_and_cannot_be_promoted():
    """The reason the pre-saturation section is supporting analysis and not the
    headline: at 1600 transitions the arms are NOT equally trained.

    `RECYCLING` differs by ~29pp between armA and armD there (p < 0.05), which is
    PR #39's confound exactly. This pins the caveat so a later edit cannot quietly
    promote the pre-saturation gap result to a claim about reset frequency."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    index = ResetIntervalReport.presaturation_index(arms=arms)
    differences = ResetIntervalReport.family_differences(
        arms=arms, family="RECYCLING", from_arm="armA", to_arm="armD", index=index
    )
    assert statistics.mean(differences) == pytest.approx(-28.9, abs=0.05)
    assert PairedTests.wilcoxon_signed_rank(differences=differences).p_value < 0.05
    # And the gap contrast it produces is smaller than the design's own MDE there,
    # so even its p < 0.05 is not a robust finding.
    extremes = [
        d - a
        for a, d in zip(
            ResetIntervalReport.gaps(arms=arms, arm="armA", index=index),
            ResetIntervalReport.gaps(arms=arms, arm="armD", index=index),
            strict=True,
        )
    ]
    assert statistics.mean(extremes) == pytest.approx(19.29, abs=0.05)
    assert PairedTests.wilcoxon_signed_rank(differences=extremes).p_value == pytest.approx(
        0.0401, abs=0.0001
    )
    assert PairedTests.minimum_detectable_effect(differences=extremes) > statistics.mean(extremes)
    slopes = ResetIntervalReport.trend_slopes(arms=arms, index=index)
    assert statistics.mean(slopes) == pytest.approx(6.16, abs=0.01)
    assert PairedTests.wilcoxon_signed_rank(differences=slopes).p_value == pytest.approx(
        0.0209, abs=0.0001
    )


def test_the_presaturation_counts_quoted_in_the_log_come_from_the_committed_data():
    """The 1600-transition table, pinned as counts. The 189/280 against 270/280 on
    RECYCLING is the whole point of the section: mid-curve the arms are 81 tasks
    apart, and by 2500 they are 6 apart."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    index = ResetIntervalReport.presaturation_index(arms=arms)
    assert {
        arm: {
            family: ResetIntervalReport.pooled_counts(
                arms=arms, arm=arm, family=family, index=index
            )
            for family in ("TRASH", "RECYCLING", "EMPTY")
        }
        for arm in _ARM_INTERVAL
    } == {
        "armA": {"TRASH": (274, 280), "RECYCLING": (270, 280), "EMPTY": (40, 40)},
        "armB": {"TRASH": (272, 280), "RECYCLING": (270, 280), "EMPTY": (40, 40)},
        "armC": {"TRASH": (263, 280), "RECYCLING": (228, 280), "EMPTY": (40, 40)},
        "armD": {"TRASH": (247, 280), "RECYCLING": (189, 280), "EMPTY": (40, 40)},
    }


def test_the_per_family_curves_are_steps_rather_than_ramps():
    """The descriptive finding the log reports: a family's 14 tasks succeed and fail
    together, so most checkpoints sit within one task of 0/14 or 14/14.

    It matters beyond being interesting -- 14 tasks per seed are not 14 independent
    observations, so the binomial noise floor computed elsewhere in this file is a
    lower bound on the real per-seed noise, not an estimate of it."""
    arms = ResetIntervalReport.load_arms(json_path=_ARMS_JSON)
    assert ResetIntervalReport.extreme_checkpoints(arms=arms, family="RECYCLING") == (1296, 2080)
    assert ResetIntervalReport.extreme_checkpoints(arms=arms, family="TRASH") == (1338, 2080)
    assert ResetIntervalReport.single_step_runs(arms=arms, family="RECYCLING") == (26, 80)
    assert ResetIntervalReport.single_step_runs(arms=arms, family="TRASH") == (33, 80)
    assert ResetIntervalReport.ceiling_collapses(arms=arms, family="RECYCLING") == 4
    assert ResetIntervalReport.ceiling_collapses(arms=arms, family="TRASH") == 2


def test_neither_the_docstring_nor_the_log_claims_a_single_terminal_failure() -> None:
    """Tossing Room has TWO terminal failure families, not one. Besides the missed
    `RECYCLING` throw, the `EMPTY` family is an ordering trap: its recycling button
    sits behind the one-way ledge, so pressing that one first puts the trash button
    out of reach for the rest of the period. Pinned behaviourally by
    tests/environments/tossingroom/test_environment.py's
    TestTheEmptyFamilyHasItsOwnTerminalFailure.

    Both this analysis module's docstring and the committed log asserted "exactly one
    genuinely terminal failure". This keeps the retraction from being quietly undone by
    a later copy-edit."""
    module = importlib.import_module("analysis.practice_makes_perfect.tossingroom_reset_interval")
    sources = {
        "module docstring": module.__doc__ or "",
        "experiment log": (
            _ARMS_JSON.parent / "2026-08-04-tossingroom-reset-interval.md"
        ).read_text(),
    }
    for name, text in sources.items():
        assert "exactly one genuinely terminal failure" not in text, name
        assert "PressRecycling" in text, name
