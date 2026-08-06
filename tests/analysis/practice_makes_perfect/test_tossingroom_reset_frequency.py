"""Tests for the Tossing Room reset-frequency analysis.

Two jobs here, and the first matters more than usual on this project.

`PairedTests` computes its own exact p-values rather than quoting values produced
by scipy elsewhere (scipy is not a dependency here). That buys reproducibility but
it means the *statistics themselves* are now project code that can be wrong, and
two results have already been retracted on this project for being overclaimed. So
the tests below pin every exact test against hand-computable ground truth: an
all-positive sample of size n has exactly one of 2**n sign assignments at least as
extreme in each direction, so its two-sided p is exactly 2 / 2**n, no table lookup
needed. `test_wilcoxon_matches_a_published_example` additionally reproduces a
worked textbook case where the answer does not follow from that shortcut.

The second job is the usual one on this project (see
`test_ballring_sampler_iters.py`): pin the analysis to the *real* committed
aggregate, so no number quoted in the experiment log or the PR can drift away from
the data that produced it without a test failing.
"""

import importlib
import math
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossingroom_reset_frequency import (
    PairedTests,
    ResetFrequencyReport,
)

_LOGS = Path(__file__).parents[3] / "docs/experiment-logs"

# The live aggregate: the four arms re-run on the fixed 14/14/2 evaluation set that
# PR #41 introduced. Every number quoted in the experiment log's live tables comes
# from this file.
_ARMS_JSON = _LOGS / "2026-08-04-tossingroom-reset-freq.json"

# The superseded aggregate, kept so the log's "previously" columns re-derive from a
# committed file rather than from prose. It was measured on the *sampled*
# composition, so it deliberately fails the 14/14/2 composition check below -- that
# failure is the whole reason the arms were re-run.
_SUPERSEDED_ARMS_JSON = _LOGS / "2026-08-03-tossingroom-reset-freq.json"


def test_wilcoxon_on_an_all_positive_sample_is_exactly_two_over_two_to_the_n():
    """The one case whose exact p needs no table: every difference positive means
    exactly one of the 2**n sign assignments is this extreme in each direction."""
    for n in (5, 8, 10):
        result = PairedTests.wilcoxon_signed_rank(differences=[float(i) for i in range(1, n + 1)])
        assert result.p_value == pytest.approx(2 / 2**n)
        assert result.num_zero_differences == 0


def test_sign_flip_on_an_all_positive_sample_is_exactly_two_over_two_to_the_n():
    for n in (5, 8, 10):
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
    """The gap metric lands on multiples of ~8pp, so exact ties are common. A test
    that quietly dropped them would report a small n as if it were the full ten."""
    result = PairedTests.wilcoxon_signed_rank(differences=[0.0, 0.0, 1.0, 2.0, 3.0])
    assert result.num_zero_differences == 2
    assert result.p_value == pytest.approx(2 / 2**3)


def test_an_all_zero_sample_is_p_one_and_needs_infinitely_many_seeds():
    assert PairedTests.wilcoxon_signed_rank(differences=[0.0] * 10).p_value == 1.0
    assert PairedTests.sign_flip(differences=[0.0] * 10).p_value == 1.0
    assert math.isinf(PairedTests.seeds_for_80_percent_power(differences=[0.0] * 10))


def test_tied_absolute_differences_get_average_ranks():
    """Exactness under ties depends on averaging the ranks and enumerating against
    those, so the averaging itself is pinned."""
    assert PairedTests._average_ranks(values=[3.0, 1.0, 1.0, 2.0]) == [4.0, 1.5, 1.5, 3.0]


def test_power_calculation_matches_the_closed_form():
    """n = (z_0.975 + z_0.80)^2 * (sd/mean)^2 -- ~7.849 seeds for a one-sd effect."""
    differences = [1.0, -1.0, 1.0, -1.0]  # mean 0 handled separately; use a shifted set
    del differences
    needed = PairedTests.seeds_for_80_percent_power(differences=[0.0, 1.0, 2.0, 3.0, 4.0])
    # mean 2.0, sd 1.5811 -> 7.849 * (1.5811/2)^2 = 4.905
    assert needed == pytest.approx(4.905, abs=0.01)


def test_family_labels_cover_every_goal_the_committed_aggregate_contains():
    """Every arm/seed/sweep must carry all three families with a non-zero
    denominator -- an unrecognised goal raises during aggregation, but a family
    that silently vanished from the test draw would not."""
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    for arm, seeds in arms.items():
        for seed, curves in seeds.items():
            assert set(curves) == {"RECYCLING", "TRASH", "EMPTY"}
            for family, triples in curves.items():
                for _transitions, solved, total in triples:
                    assert total > 0, f"{arm}/{seed} has no {family} tasks"
                    assert 0 <= solved <= total


def test_every_arm_shares_the_same_seeds_so_pairing_is_valid():
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    seeds = ResetFrequencyReport.seeds(arms=arms)
    assert seeds == [str(seed) for seed in range(10)]
    for arm in arms:
        assert sorted(arms[arm], key=int) == seeds


def test_every_arm_draws_the_same_test_set_so_denominators_match_across_arms():
    """Pairing across arms is only meaningful if seed s sees the same 30 tasks in
    every arm. It should -- the test RNG stream is derived from `seed +
    test_env_seed_offset`, independent of `--num-cycles` -- but the whole
    cross-arm comparison rests on it, so it is checked against the data rather
    than argued from the code."""
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    for seed in ResetFrequencyReport.seeds(arms=arms):
        denominators = {
            arm: str({family: triples[0][2] for family, triples in arms[arm][seed].items()})
            for arm in arms
        }
        assert len(set(denominators.values())) == 1, (
            f"seed {seed} drew different task families per arm: {denominators}"
        )


def test_the_evaluation_set_really_was_the_fixed_14_14_2_composition():
    """The manipulation check for the *evaluation* side, asserted against the data
    rather than assumed from the flags.

    This experiment's own headline was that a design failed to isolate what it
    claimed. Taking the test-set composition on trust would repeat that one level
    down -- and it is precisely the assumption that went stale underneath the
    original run, whose numbers were measured on a sampled composition that the
    code no longer produces.
    """
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    assert ResetFrequencyReport.expected_composition() == {
        "TRASH": 14,
        "RECYCLING": 14,
        "EMPTY": 2,
    }
    assert ResetFrequencyReport.composition_violations(arms=arms) == []


def test_every_arm_actually_spent_the_2500_transition_budget():
    """The manipulation check for the *training* side. The arms are comparable only
    because the experience budget is identical by construction, so a shortfall is a
    residual confound and is measured rather than argued from the loop's
    arithmetic."""
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    assert ResetFrequencyReport.transition_violations(arms=arms) == []


def test_the_superseded_aggregate_is_kept_and_fails_the_composition_check():
    """The re-run's own justification, pinned. The 2026-08-03 aggregate is retained
    so the log's "previously" columns re-derive from a committed file, and it must
    keep failing the 14/14/2 check -- if it ever passed, the two files would be
    measuring the same evaluation set and the re-run would have been unnecessary."""
    superseded = ResetFrequencyReport.load_arms(json_path=_SUPERSEDED_ARMS_JSON)
    assert ResetFrequencyReport.composition_violations(arms=superseded) != []


def test_the_noise_floor_follows_from_the_fixed_composition():
    """14 tasks in each throw family puts the gap's binomial noise floor at
    100 * sqrt(0.25/14 + 0.25/14) = 18.9pp, the same in every arm because the
    composition no longer varies by seed. Quoted in the log beside every observed
    sd, so it is pinned here."""
    arms = ResetFrequencyReport.load_arms(json_path=_ARMS_JSON)
    for arm in ("armA", "armB", "armC", "armD"):
        assert ResetFrequencyReport.predicted_gap_noise(arms=arms, arm=arm) == pytest.approx(
            18.90, abs=0.01
        )


def test_neither_the_docstring_nor_the_log_claims_a_single_terminal_failure():
    """Tossing Room has TWO terminal failure families, not one. Besides the missed
    `RECYCLING` throw, the `EMPTY` family is an ordering trap: its recycling button
    sits behind the one-way ledge, so pressing that one first puts the trash button
    out of reach for the rest of the period. Pinned behaviourally by
    tests/environments/tossingroom/test_environment.py's
    TestTheEmptyFamilyHasItsOwnTerminalFailure.

    Both this analysis module's docstring and the committed log asserted "exactly one
    genuinely terminal failure". This keeps the retraction from being quietly undone by
    a later copy-edit."""
    module = importlib.import_module("analysis.practice_makes_perfect.tossingroom_reset_frequency")
    sources = {
        "module docstring": module.__doc__ or "",
        "experiment log": (_LOGS / "2026-08-03-tossingroom-reset-frequency.md").read_text(),
    }
    for name, text in sources.items():
        assert "exactly one genuinely terminal failure" not in text, name
        assert "PressRecycling" in text, name
