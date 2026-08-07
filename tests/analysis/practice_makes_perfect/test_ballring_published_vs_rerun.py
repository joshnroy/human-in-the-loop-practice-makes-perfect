"""Tests for the published-vs-re-run Ball-Ring comparison.

The load-bearing one is `test_the_arms_diverge_at_the_first_practice_checkpoint`. The
whole reading of the 99/100-vs-91/100 gap turns on *when* the two arms part company: a
divergence that only appears near the end is consistent with noise in a converged tail,
while one at the first checkpoint after any practice at all means the two runs were never
the same computation. If a future re-aggregation moves that onset later, the prose in
`2026-08-03-ballring-iters.md`'s staleness note becomes wrong and this fails.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.ballring_published_vs_rerun import BallRingPublishedVsRerun

_DOCS = Path(__file__).parents[3] / "docs/experiment-logs"
_PUBLISHED = _DOCS / "2026-08-03-ballring-arms.json"
_RERUN = _DOCS / "2026-08-06-ballring-placeballontable"


def _pair():
    return BallRingPublishedVsRerun.load_pair(published_json=_PUBLISHED, rerun_root=_RERUN)


def test_both_arms_carry_the_same_ten_seeds_and_checkpoints():
    published, rerun = _pair()
    assert sorted(published) == sorted(rerun) == list(range(10))
    for seed in range(10):
        assert len(published[seed]) == len(rerun[seed]) == 26
        # Same transition grid, so the checkpoints are genuinely comparable.
        assert [p[0] for p in published[seed]] == [r[0] for r in rerun[seed]]


def test_endpoints_match_the_two_published_totals():
    """99/100 and 91/100 -- read from the artifacts, never restated by hand."""
    published, rerun = _pair()
    assert BallRingPublishedVsRerun.endpoint_total(arm=published) == (99, 100)
    assert BallRingPublishedVsRerun.endpoint_total(arm=rerun) == (91, 100)


def test_the_arms_diverge_at_the_first_practice_checkpoint():
    """7/10 seeds differ already at 100 transitions; 10/10 by 200.

    Checkpoint 0 is before any practice and is 0/10 in every seed of both arms, so it
    carries no information -- the first *informative* checkpoint is index 1.
    """
    published, rerun = _pair()
    onsets = BallRingPublishedVsRerun.divergence_onset(published=published, rerun=rerun)
    assert all(onset is not None for onset in onsets.values())
    assert sum(1 for onset in onsets.values() if onset == 1) == 7
    assert all(onset <= 2 for onset in onsets.values())


def test_the_whole_curve_is_better_powered_and_still_finds_nothing():
    """The endpoint test *cannot* reach 0.05; the curve-area test can, and does not.

    Endpoint: 5/10 pairs tied, so the exact two-sided permutation p sits exactly at its
    floor of 2 x 2^-5 = 0.0625 -- that number describes the design, not the world.
    Curve area: 10/10 pairs non-tied, so the floor drops to 2 x 2^-10 = 0.00195 and the
    test genuinely could have resolved a consistent shift. It does not: the re-run is
    lower in only 6/10 seeds and p = 0.109.

    Pinned because both halves are load-bearing and pull opposite ways. The endpoint's
    5/5 one-directional split invites "a regression we cannot quite prove"; the
    better-powered statistic says the systematic-regression reading is *not* supported.
    What remains certain is `test_the_arms_diverge_at_the_first_practice_checkpoint` --
    the arms are different computations -- which is a determinism fact, not a statistical
    one.
    """
    published, rerun = _pair()
    end_diffs = BallRingPublishedVsRerun.endpoint_diffs(published=published, rerun=rerun)
    end_p = BallRingPublishedVsRerun.paired_permutation_p(diffs=end_diffs)
    assert sum(1 for d in end_diffs if d == 0) == 5
    assert end_p == pytest.approx(0.0625, abs=1e-9)

    area_diffs = BallRingPublishedVsRerun.curve_area_diffs(published=published, rerun=rerun)
    area_p = BallRingPublishedVsRerun.paired_permutation_p(diffs=area_diffs)
    assert all(d != 0 for d in area_diffs)
    assert sum(1 for d in area_diffs if d < 0) == 6
    assert area_p == pytest.approx(0.109375, abs=1e-9)
    assert area_p > 0.05
