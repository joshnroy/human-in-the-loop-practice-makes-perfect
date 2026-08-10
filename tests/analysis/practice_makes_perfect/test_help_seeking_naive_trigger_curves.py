"""Tests for the `on-no-applicable-skill` vs `no-human` training-curve figure.

The figure's whole claim is that the two curves are not merely close but the *same*
curve, drawn twice -- so the sharpest thing to get wrong here is silently comparing two
different things. Three failure modes matter:

1. **Reading the wrong checkpoint as "final".** Same `[-1]` vs `[0]` trap
   `reset_free_training_curves.py` guards against.
2. **A family-classification bug that hides a real difference.** If TRASH/RECYCLING
   were computed inconsistently between the two arms, the panels could show "identical"
   for the wrong reason. Pinned against the published per-seed finals so any drift is
   caught.
3. **The two arms silently compared on different transition grids.** `render` raises
   if they disagree rather than plotting a distorted overlay.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.help_seeking_naive_trigger_curves import (
    _ARM_COLOR,
    _NO_HUMAN,
    _NUM_SEEDS,
    _NUM_TEST_TASKS,
    _ON_NO_APPLICABLE_SKILL,
    HelpSeekingNaiveTriggerCurves,
)

_LOGS = Path(__file__).resolve().parents[3] / "docs" / "experiment-logs"

# The published per-seed finals (docs/experiment-logs/2026-08-10-help-seeking-naive-
# trigger.md and its PR): both arms identical seed-for-seed.
_PUBLISHED_OVERALL_FINALS = [18, 16, 5, 6, 7, 6, 21, 20, 7, 6]
_PUBLISHED_TRASH_FINALS = [14, 12, 2, 2, 3, 2, 14, 14, 4, 3]
_PUBLISHED_RECYCLING_FINALS = [2, 2, 1, 2, 2, 2, 5, 4, 1, 1]

_PUBLISHED_FINALS = {
    "OVERALL": _PUBLISHED_OVERALL_FINALS,
    "TRASH": _PUBLISHED_TRASH_FINALS,
    "RECYCLING": _PUBLISHED_RECYCLING_FINALS,
}
_PUBLISHED_TOTALS = {"OVERALL": 112, "TRASH": 70, "RECYCLING": 22}
_PUBLISHED_DENOMINATORS = {"OVERALL": 300, "TRASH": 140, "RECYCLING": 140}


@pytest.mark.parametrize("arm", ["no-human", "on-no-applicable-skill"])
def test_all_ten_seeds_load(*, arm: str) -> None:
    runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm=arm)
    assert len(runs) == _NUM_SEEDS


@pytest.mark.parametrize("arm", ["no-human", "on-no-applicable-skill"])
@pytest.mark.parametrize("family", ["OVERALL", "TRASH", "RECYCLING"])
def test_per_seed_finals_reproduce_the_published_values(*, arm: str, family: str) -> None:
    """Both arms must reproduce the SAME published finals -- that identity is the
    entire point of the figure, not merely each arm being internally plausible."""
    runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm=arm)
    curves = HelpSeekingNaiveTriggerCurves.per_seed_curve(runs=runs, family=family)
    finals = [curve[-1] for curve in curves]
    assert finals == _PUBLISHED_FINALS[family]


@pytest.mark.parametrize("arm", ["no-human", "on-no-applicable-skill"])
@pytest.mark.parametrize("family", ["OVERALL", "TRASH", "RECYCLING"])
def test_arm_totals_match_the_published_counts(*, arm: str, family: str) -> None:
    """Reported as x/y, matching the log's own table."""
    runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm=arm)
    total = HelpSeekingNaiveTriggerCurves.arm_total(runs=runs, family=family)
    denom = HelpSeekingNaiveTriggerCurves.family_denominator(runs=runs, family=family)
    assert (total, denom) == (_PUBLISHED_TOTALS[family], _PUBLISHED_DENOMINATORS[family])


def test_overall_denominator_is_ten_seeds_times_thirty_tasks() -> None:
    runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm="no-human")
    denom = HelpSeekingNaiveTriggerCurves.family_denominator(runs=runs, family="OVERALL")
    assert denom == _NUM_SEEDS * _NUM_TEST_TASKS


@pytest.mark.parametrize("arm", ["no-human", "on-no-applicable-skill"])
def test_the_two_arms_share_one_checkpoint_grid(*, arm: str) -> None:
    runs = HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm=arm)
    grid = HelpSeekingNaiveTriggerCurves.checkpoints(runs=runs)
    assert grid[0] == 0, "the first evaluation is the pre-practice one, at 0 transitions"
    assert grid == sorted(grid)
    assert grid == list(range(0, 1501, 150)), "expected 11 checkpoints, 150 transitions apart"


def test_a_missing_seed_raises_rather_than_plotting_a_short_arm(*, tmp_path: Path) -> None:
    empty = tmp_path / "logs"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        HelpSeekingNaiveTriggerCurves.load_arm(logs_root=empty, arm="no-human")


def test_the_two_arms_are_identical_seed_for_seed_on_every_family() -> None:
    """The figure's actual claim, checked directly rather than only through two
    separate published-value comparisons: for every family, the treated arm's full
    curve equals the control's full curve, seed for seed and checkpoint for
    checkpoint -- not just at the final evaluation."""
    for family in ("OVERALL", "TRASH", "RECYCLING"):
        no_human = HelpSeekingNaiveTriggerCurves.per_seed_curve(
            runs=HelpSeekingNaiveTriggerCurves.load_arm(logs_root=_LOGS, arm="no-human"),
            family=family,
        )
        treated = HelpSeekingNaiveTriggerCurves.per_seed_curve(
            runs=HelpSeekingNaiveTriggerCurves.load_arm(
                logs_root=_LOGS, arm="on-no-applicable-skill"
            ),
            family=family,
        )
        assert treated == no_human, f"{family} curves diverge between the two arms"


def test_render_writes_a_three_panel_figure(*, tmp_path: Path) -> None:
    out = tmp_path / "curves.png"
    figure = HelpSeekingNaiveTriggerCurves.render(logs_root=_LOGS, output=out)
    assert len(figure.axes) == 3
    assert out.exists()
    assert out.stat().st_size > 10_000


def test_colour_carries_the_intervention_availability_role() -> None:
    """CLAUDE.md's training-curve-style section (#190): orange is always the arm
    nothing helps (`no-human`); blue is always the arm with an assistance mechanism
    available, whether or not it fires. `on-no-applicable-skill` measured 0/10 seeds
    ever asking but still carries blue, since the finding is that the mechanism
    existed and did nothing -- orange would visually erase that."""
    assert _ARM_COLOR[_NO_HUMAN] == "#D55E00"
    assert _ARM_COLOR[_ON_NO_APPLICABLE_SKILL] == "#0072B2"


def test_every_legend_entry_carries_n_and_the_seed_count(*, tmp_path: Path) -> None:
    """CLAUDE.md's training-curve-style section requires `n=` in every legend entry
    (e.g. `env resets -- mean, n=10`), so a reader can check `n` sums to the seed
    total without re-deriving it from the plot."""
    out = tmp_path / "curves.png"
    figure = HelpSeekingNaiveTriggerCurves.render(logs_root=_LOGS, output=out)
    for axis in figure.axes:
        handles, labels = axis.get_legend_handles_labels()
        named = [label for label in labels if label]
        assert named, "panel has no legend entries"
        for label in named:
            assert f"n={_NUM_SEEDS}" in label, f"legend entry missing seed count: {label!r}"


def test_panel_titles_carry_the_denominator_as_of_n_not_x_over_n(*, tmp_path: Path) -> None:
    """CLAUDE.md's training-curve-style section: the axis label stays bare and the
    denominator goes in the panel's own title, phrased `(of N)` (e.g.
    `TRASH tasks (of 14)`), not `(x/N)`."""
    out = tmp_path / "curves.png"
    figure = HelpSeekingNaiveTriggerCurves.render(logs_root=_LOGS, output=out)
    # `figure.axes` also holds the per-panel `secondary_xaxis` (cycle) twins, which
    # carry no title of their own -- only the three main panels do. `render` sets the
    # title with `loc="left"`, so it must be read back the same way -- the default
    # `loc="center"` is a different (empty) string.
    titles = [axis.get_title(loc="left") for axis in figure.axes if axis.get_title(loc="left")]
    assert len(titles) == 3
    for title in titles:
        assert "(of " in title, f"panel title missing '(of N)' denominator: {title!r}"
        assert "/" not in title, f"panel title still uses x/N phrasing: {title!r}"
