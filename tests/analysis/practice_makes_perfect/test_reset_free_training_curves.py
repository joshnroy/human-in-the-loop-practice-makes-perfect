"""Tests for the cross-variant training-curve figure: `scheduled` against `never` in
each of the four Tossing Room variants this stack measured.

The figure is drawn from the 60 `stats.json` files committed under
`docs/experiment-logs/`, so the extraction is the only thing between the committed data
and a published shape. Four things can be wrong here while the picture still looks
entirely plausible:

1. **Reading the wrong checkpoint as "final".** `Metrics.evaluations` is
   `(transitions, solved, total)` and its *first* entry is the pre-practice evaluation
   at 0 transitions. Taking `[0]` instead of `[-1]` yields a lower number for every arm
   and a monotone story that reads fine. Every arm's per-seed finals are pinned to the
   values the aggregate tables already publish, so an off-by-one indexes to a
   disagreement rather than to a plausible alternative.
2. **Silently short seed sets.** A glob that finds 8 of 10 seeds still means and plots
   without complaint, and a mean over 8 seeds is not the published 300-task denominator.
   `load_arm` raises instead.
3. **Unaligned x-grids.** The four variants do not share an evaluation horizon
   (`--two-way-ledge` drops EMPTY's shortest solve, so its horizon differs), so a mean
   taken across seeds whose checkpoint grids differ silently averages different amounts
   of practice together. Alignment is asserted within each arm.
4. **The bimodality being an artefact of ordering.** `pickup-weight / never` splits
   4 seeds high / 6 seeds low, and that split is the clearest argument in the write-up
   for per-seed plotting over means. It is pinned here so a change to extraction or to
   seed ordering cannot quietly dissolve it.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.reset_free_training_curves import (
    _NUM_SEEDS,
    _NUM_TEST_TASKS,
    ArmSpec,
    ResetFreeTrainingCurves,
)

_LOGS = Path(__file__).resolve().parents[3] / "docs" / "experiment-logs"

# The per-seed final scores every arm is published with, in seed order 0..9. These are
# the numbers the stack's aggregate tables already report; pinning them here is what
# makes this figure a re-reading of banked data rather than a fresh measurement.
_PUBLISHED_FINALS = {
    ("one-way", "scheduled"): [8, 16, 18, 18, 16, 16, 18, 17, 6, 18],
    ("one-way", "never"): [8, 10, 6, 5, 7, 11, 8, 12, 6, 12],
    ("two-way", "scheduled"): [29, 19, 30, 30, 30, 27, 24, 30, 27, 30],
    ("two-way", "never"): [16, 10, 17, 13, 14, 19, 10, 15, 16, 14],
    ("pickup-weight", "scheduled"): [27, 18, 16, 18, 15, 18, 17, 19, 18, 17],
    ("pickup-weight", "never"): [18, 16, 5, 6, 7, 6, 21, 20, 7, 6],
    # The fourth cell (this stack's own new run), pinned the same way the banked six are.
    ("pickup-weight-two-way", "scheduled"): [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    ("pickup-weight-two-way", "never"): [30, 30, 17, 30, 30, 30, 30, 30, 30, 30],
}

_PUBLISHED_TOTALS = {
    ("one-way", "scheduled"): 151,
    ("one-way", "never"): 85,
    ("two-way", "scheduled"): 276,
    ("two-way", "never"): 144,
    ("pickup-weight", "scheduled"): 183,
    ("pickup-weight", "never"): 112,
    ("pickup-weight-two-way", "scheduled"): 300,
    ("pickup-weight-two-way", "never"): 287,
}


def _arm(*, panel: str, policy: str) -> ArmSpec:
    """The declared spec for one (panel, policy) cell, so a test never re-spells a path."""
    return ResetFreeTrainingCurves.arm(panel=panel, policy=policy)


@pytest.mark.parametrize(("panel", "policy"), sorted(_PUBLISHED_FINALS))
def test_per_seed_finals_reproduce_the_published_values(*, panel: str, policy: str) -> None:
    """The whole point of the figure is that it re-reads banked data. If any arm's
    per-seed finals disagree with what the tables published, the figure is describing a
    different run and nothing drawn from it can be trusted."""
    runs = ResetFreeTrainingCurves.load_arm(logs_root=_LOGS, arm=_arm(panel=panel, policy=policy))
    finals = ResetFreeTrainingCurves.per_seed_finals(runs=runs)
    assert finals == _PUBLISHED_FINALS[(panel, policy)]


@pytest.mark.parametrize(("panel", "policy"), sorted(_PUBLISHED_TOTALS))
def test_arm_totals_match_the_published_counts(*, panel: str, policy: str) -> None:
    """Reported as x/y: the denominator is 10 seeds x 30 test tasks, and it is asserted
    rather than assumed so an arm that lost a seed cannot report a plausible x/300."""
    runs = ResetFreeTrainingCurves.load_arm(logs_root=_LOGS, arm=_arm(panel=panel, policy=policy))
    solved, total = ResetFreeTrainingCurves.arm_total(runs=runs)
    assert (solved, total) == (_PUBLISHED_TOTALS[(panel, policy)], _NUM_SEEDS * _NUM_TEST_TASKS)


@pytest.mark.parametrize(("panel", "policy"), sorted(_PUBLISHED_FINALS))
def test_every_seed_in_an_arm_shares_one_checkpoint_grid(*, panel: str, policy: str) -> None:
    """A mean across seeds is only meaningful if the seeds were evaluated at the same
    transition counts. Within an arm they must agree exactly; `checkpoints` raises if
    not, and returns the single shared grid if so."""
    runs = ResetFreeTrainingCurves.load_arm(logs_root=_LOGS, arm=_arm(panel=panel, policy=policy))
    grid = ResetFreeTrainingCurves.checkpoints(runs=runs)
    assert grid[0] == 0, "the first evaluation is the pre-practice one, at 0 transitions"
    assert grid == sorted(grid), "checkpoints must be increasing"
    assert len(grid) == len(runs[0].evaluations)


def test_a_missing_seed_raises_rather_than_plotting_a_short_arm(*, tmp_path: Path) -> None:
    """Nine seeds still produce a mean and a picture. The denominator would silently stop
    being 300, so this is an error rather than a degraded plot."""
    empty = tmp_path / "logs"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ResetFreeTrainingCurves.load_arm(
            logs_root=empty, arm=_arm(panel="one-way", policy="scheduled")
        )


def test_pickup_weight_never_is_bimodal_rather_than_merely_low() -> None:
    """Four seeds track the one-way arm and six collapse -- the stranding split showing
    up directly in task outcomes. This is the figure's strongest argument for per-seed
    lines over a mean, so it is pinned: the mean alone (11.2/30) describes no seed."""
    runs = ResetFreeTrainingCurves.load_arm(
        logs_root=_LOGS, arm=_arm(panel="pickup-weight", policy="never")
    )
    finals = ResetFreeTrainingCurves.per_seed_finals(runs=runs)
    high = [f for f in finals if f >= 15]
    low = [f for f in finals if f <= 7]
    assert len(high) == 4, f"expected 4 seeds tracking the one-way arm, got {high}"
    assert len(low) == 6, f"expected 6 collapsed seeds, got {low}"
    assert not [f for f in finals if 7 < f < 15], "the split is a gap, not a gradient"


def test_the_two_way_ledge_lifts_the_scheduled_arm_too() -> None:
    """The honest frame for the headline, and the one the outcome tables do not show:
    `never` did not get worse in absolute terms when the ledge opened (85/300 ->
    144/300, it improved); `scheduled` simply improved far more (151/300 -> 276/300).
    Asserted so the write-up's reading cannot drift from the data."""
    totals = {
        (panel, policy): ResetFreeTrainingCurves.arm_total(
            runs=ResetFreeTrainingCurves.load_arm(
                logs_root=_LOGS, arm=_arm(panel=panel, policy=policy)
            )
        )[0]
        for panel, policy in _PUBLISHED_TOTALS
    }
    assert totals[("two-way", "never")] > totals[("one-way", "never")]
    scheduled_gain = totals[("two-way", "scheduled")] - totals[("one-way", "scheduled")]
    never_gain = totals[("two-way", "never")] - totals[("one-way", "never")]
    assert scheduled_gain > never_gain


def _totals() -> dict[tuple[str, str], int]:
    return {
        (panel, policy): ResetFreeTrainingCurves.arm_total(
            runs=ResetFreeTrainingCurves.load_arm(
                logs_root=_LOGS, arm=_arm(panel=panel, policy=policy)
            )
        )[0]
        for panel, policy in _PUBLISHED_TOTALS
    }


def test_the_fourth_cells_bimodality_is_gone() -> None:
    """The falsifiable prediction the fourth cell's pre-registration committed *before*
    the sweep ran: removing stranding should remove the low mode. The one-way arm splits
    6/10 low and 4/10 high with an empty band between 7 and 15; the two-way arm must not.

    Asserted as "no cluster in the low band", not as "unimodal" -- one seed does finish
    17/30, and calling a single tail a mode would be the same mistake in reverse."""
    finals = ResetFreeTrainingCurves.per_seed_finals(
        runs=ResetFreeTrainingCurves.load_arm(
            logs_root=_LOGS, arm=_arm(panel="pickup-weight-two-way", policy="never")
        )
    )
    assert [f for f in finals if f <= 7] == [], f"the collapsed mode should be gone: {finals}"
    assert len([f for f in finals if f == 30]) == 9, f"expected 9 seeds at ceiling: {finals}"


def test_removing_both_mechanisms_closes_the_gap() -> None:
    """The fourth cell's headline, pinned so the write-up cannot drift from the data.

    The within-world gap collapses to 13/300, an order below every other panel's -- and
    the interaction runs in OPPOSITE directions in the two variants, which is the finding
    the outcome tables state but no single number carries: the two-way ledge widens the
    gap on `tossingroomsplit` and collapses it on the pickup-weight fork."""
    totals = _totals()
    gap = {
        panel: totals[(panel, "scheduled")] - totals[(panel, "never")]
        for panel in ("one-way", "two-way", "pickup-weight", "pickup-weight-two-way")
    }
    assert gap["pickup-weight-two-way"] == 13
    assert gap["pickup-weight-two-way"] < min(gap["one-way"], gap["two-way"], gap["pickup-weight"])
    # Opposite-signed interactions: +66 on tossingroomsplit, -58 on the pickup-weight fork.
    assert gap["two-way"] - gap["one-way"] > 0
    assert gap["pickup-weight-two-way"] - gap["pickup-weight"] < 0


def test_render_writes_a_four_panel_figure(*, tmp_path: Path) -> None:
    """The artifact itself: four panels -- the completed 2x2 -- and a PNG on disk with
    real bytes in it."""
    out = tmp_path / "curves.png"
    figure = ResetFreeTrainingCurves.render(logs_root=_LOGS, output=out)
    assert len(figure.axes) == 4
    assert out.exists()
    assert out.stat().st_size > 10_000
