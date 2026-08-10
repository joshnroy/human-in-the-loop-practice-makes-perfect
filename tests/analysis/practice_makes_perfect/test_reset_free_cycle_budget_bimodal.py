"""`ResetFreeCycleBudgetBimodal` redraws the 10x-budget cells of
`docs/experiment-logs/2026-08-07-pickup-weight-cycle-budget-10x-runs/` with the one-way
`never` arm's bold mean split into its two structurally distinct subpopulations (see
`ResetFreeCycleBudget`'s own "Addendum: the one-way `never` cell is a mixture of two
populations") rather than pooled into a single line that describes neither.

What is pinned here is the arithmetic, not matplotlib:

- **the stuck/non-stuck split reproduces from the real committed data**, against the
  exact seed sets the addendum already published (`[2, 3, 4, 5, 8, 9]` stuck,
  `[0, 1, 6, 7]` non-stuck) -- read through `ResetFreeCycleBudget.load_cell`, not a
  hand-copied fixture, so a change to that loader's shape cannot silently desync this
  figure from the numbers it is supposed to illustrate;
- **the "never gains anything at all" edge case** collapses into the same "stuck" bucket
  as "last gain was at checkpoint 1", because `max(..., default=0)` cannot distinguish
  them -- both describe a robot that got zero *additional* effective practice after the
  first checkpoint, so the shared bucket is the right call, not an oversight;
- **the subgroup mean is a plain per-seed average**, not a pooled/summed x-y pair, and
  the two agree here only because `ResetFreeCycleBudget.load_cell` already asserts every
  seed shares the same per-checkpoint denominator;
- **the legend's final count is pooled (summed) over the subgroup's own seeds**, read
  from each seed's own last-checkpoint `series(...)[-1]`, not derived by rescaling the
  mean curve.
"""

import json
from pathlib import Path

from analysis.practice_makes_perfect.reset_free_cycle_budget import ResetFreeCycleBudget
from analysis.practice_makes_perfect.reset_free_cycle_budget_bimodal import (
    ResetFreeCycleBudgetBimodal,
)

_RUNS = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "experiment-logs"
    / "2026-08-07-pickup-weight-cycle-budget-10x-runs"
)

_TRASH = "TrashInBin(trash, trash_bin)"
_RECYCLING = "RecyclingInBin(recycling, recycling_bin)"
_EMPTY = "RecyclingBinEmpty(recycling_bin) & TrashBinEmpty(trash_bin)"


# ------------------------------------------------------------------ real committed data


def test_the_stuck_split_reproduces_from_the_real_committed_one_way_never_cell() -> None:
    """The addendum in `2026-08-07-pickup-weight-cycle-budget.md` already published this
    split by wall clock and by stranding cycle, agreeing on 10/10 seeds. This asserts the
    same split falls out of `ResetFreeCycleBudget.load_cell`'s own `effective_attempts`,
    read straight from the committed `stats.json` files -- not a value hand-copied from
    the markdown."""
    cell = ResetFreeCycleBudget.load_cell(
        directory=_RUNS / "oneway-never", budget="10x", policy="never"
    )
    stuck, non_stuck = ResetFreeCycleBudgetBimodal.stuck_split(cell=cell)
    assert stuck == [2, 3, 4, 5, 8, 9]
    assert non_stuck == [0, 1, 6, 7]


def test_the_two_way_never_cell_has_no_split_because_stranding_is_impossible_there() -> None:
    """`--two-way-ledge` removes the domain's only irreversible action, so every seed
    keeps gaining effective attempts throughout -- the split must come back empty on one
    side, which is what licenses drawing a single undivided line for this arm."""
    cell = ResetFreeCycleBudget.load_cell(
        directory=_RUNS / "twoway-never", budget="10x", policy="never"
    )
    stuck, _non_stuck = ResetFreeCycleBudgetBimodal.stuck_split(cell=cell)
    assert stuck == []


# ------------------------------------------------------------------ the default=0 edge case


def test_a_seed_that_never_gains_a_single_effective_attempt_is_also_classified_stuck() -> None:
    """`max(..., default=0)` cannot tell "the last gain was at checkpoint 1" apart from
    "there was never a gain at all" -- both hit the `default=0` branch. That collapse is
    deliberate rather than an oversight: both describe a robot that picked up zero
    *additional* effective practice after the first checkpoint, which is exactly what
    "stuck" means for this figure, so they belong in the same bucket."""
    cell = {
        0: {"effective_attempts": [0, 0, 0, 0]},  # never gains anything at all
        1: {"effective_attempts": [0, 2, 2, 2]},  # last gain at checkpoint 1 -- same bucket
        2: {"effective_attempts": [0, 2, 2, 40]},  # last gain at checkpoint 3 -- non-stuck
    }
    stuck, non_stuck = ResetFreeCycleBudgetBimodal.stuck_split(cell=cell)
    assert stuck == [0, 1]
    assert non_stuck == [2]


# ------------------------------------------------------------------ subgroup arithmetic


def _sweep(*, trash_solved: int, recycling_solved: int, empty_solved: int) -> list[dict]:
    """One evaluation sweep with the domain's real 14 / 14 / 2 composition."""
    outcomes = []
    for index in range(14):
        outcomes.append({"task_index": index, "goal": _TRASH, "solved": index < trash_solved})
    for index in range(14):
        outcomes.append({
            "task_index": 14 + index,
            "goal": _RECYCLING,
            "solved": index < recycling_solved,
        })
    for index in range(2):
        outcomes.append({"task_index": 28 + index, "goal": _EMPTY, "solved": index < empty_solved})
    return outcomes


def _write_run(*, directory: Path, seed: int, levels: list[tuple[int, int, int]]) -> None:
    """A 10x (100-cycle) run flat at a possibly-changing level per checkpoint, with a
    `MoveRoom`-only trailing practice window so `effective_attempts` stays at 0 -- these
    fixtures test subgroup arithmetic, not the split, so no seed here needs to look
    non-stuck."""
    run = directory / str(seed)
    run.mkdir(parents=True)
    breakdowns = [
        {
            "num_online_transitions": index * 150,
            "outcomes": _sweep(trash_solved=t, recycling_solved=r, empty_solved=e),
        }
        for index, (t, r, e) in enumerate(levels)
    ]
    windows = [{"MoveRoom": {"num_attempts": 100, "num_successes": 100}} for _ in range(100)]
    windows.append({"MoveRoom": {"num_attempts": 0, "num_successes": 0}})
    run.joinpath("stats.json").write_text(
        json.dumps({
            "breakdowns": breakdowns,
            "num_practice_resets": 0,
            "practice_outcomes_per_cycle": windows,
            "task_name": "default",
        })
    )


def _cell(*, root: Path, levels_by_seed: dict[int, tuple[int, int, int]]) -> dict:
    directory = root / "oneway-never"
    for seed, level in levels_by_seed.items():
        _write_run(directory=directory, seed=seed, levels=[level] * 101)
    return ResetFreeCycleBudget.load_cell(directory=directory, budget="10x", policy="never")


def test_subgroup_mean_curve_is_a_plain_per_seed_average(*, tmp_path: Path) -> None:
    """Two seeds flat at 12 and 4 TRASH average to 8 at every checkpoint -- a plain mean,
    not solved-summed-over-total-summed (which would also read 16/28 = 8/14 here, so the
    fixture uses TRASH alone to keep the arithmetic checkable by hand)."""
    cell = _cell(root=tmp_path, levels_by_seed={0: (12, 3, 2), 1: (4, 3, 2)})
    mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(cell=cell, seeds=[0, 1], family="TRASH")
    assert mean[0] == 8.0
    assert mean[-1] == 8.0


def test_subgroup_pooled_final_sums_rather_than_scales_the_mean(*, tmp_path: Path) -> None:
    """The legend count is the pooled (summed) solved/total over the subgroup's own
    seeds at the final checkpoint -- 12 + 4 = 16 solved out of 14 + 14 = 28, not derived
    by scaling the mean (8.0) back up by a seed count."""
    cell = _cell(root=tmp_path, levels_by_seed={0: (12, 3, 2), 1: (4, 3, 2)})
    solved, total = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
        cell=cell, seeds=[0, 1], family="TRASH"
    )
    assert (solved, total) == (16, 28)


def test_subgroup_functions_accept_a_strict_subset_of_the_cells_seeds(*, tmp_path: Path) -> None:
    """The whole point of a subgroup split is to average over fewer than all the cell's
    seeds -- with three seeds present, asking for just two must not silently pull in the
    third."""
    cell = _cell(root=tmp_path, levels_by_seed={0: (12, 3, 2), 1: (4, 3, 2), 2: (14, 3, 2)})
    mean = ResetFreeCycleBudgetBimodal.subgroup_mean_curve(cell=cell, seeds=[0, 1], family="TRASH")
    assert mean[0] == 8.0
    solved, total = ResetFreeCycleBudgetBimodal.subgroup_pooled_final(
        cell=cell, seeds=[0, 1], family="TRASH"
    )
    assert (solved, total) == (16, 28)


# ------------------------------------------------------------------ the figure's shape


def test_render_writes_a_six_panel_figure_with_the_denominator_off_the_axis(
    *,
    tmp_path: Path,
) -> None:
    """Three rows (OVERALL / TRASH / RECYCLING) x two columns (one-way / two-way): the
    denominator lives in each panel's title, per CLAUDE.md's training-curve-style
    section, so the y-axis label itself must carry no digit or slash."""
    cells = {
        ("one-way", "scheduled"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "oneway-scheduled", budget="10x", policy="scheduled"
        ),
        ("one-way", "never"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "oneway-never", budget="10x", policy="never"
        ),
        ("two-way", "scheduled"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "twoway-scheduled", budget="10x", policy="scheduled"
        ),
        ("two-way", "never"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "twoway-never", budget="10x", policy="never"
        ),
    }
    figure = ResetFreeCycleBudgetBimodal.render(
        cells=cells, output=tmp_path / "curves.png", title="test"
    )
    axes = list(figure.axes)
    assert len(axes) == 6
    for axis in axes:
        title = axis.get_title()
        assert "of " in title
        ylabel = axis.get_ylabel()
        assert not any(character.isdigit() for character in ylabel)
        assert "/" not in ylabel
    assert (tmp_path / "curves.png").is_file()


def test_legend_entries_carry_both_n_and_the_final_count(*, tmp_path: Path) -> None:
    """CLAUDE.md's training-curve-style section requires `n=` in every legend entry; this
    figure additionally restores the pooled final `x/y` the prototype's legend dropped."""
    cells = {
        ("one-way", "scheduled"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "oneway-scheduled", budget="10x", policy="scheduled"
        ),
        ("one-way", "never"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "oneway-never", budget="10x", policy="never"
        ),
        ("two-way", "scheduled"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "twoway-scheduled", budget="10x", policy="scheduled"
        ),
        ("two-way", "never"): ResetFreeCycleBudget.load_cell(
            directory=_RUNS / "twoway-never", budget="10x", policy="never"
        ),
    }
    figure = ResetFreeCycleBudgetBimodal.render(
        cells=cells, output=tmp_path / "curves.png", title="test"
    )
    one_way_never_panel = figure.axes[0]
    labels = [line.get_label() for line in one_way_never_panel.get_lines()]
    named = [label for label in labels if not label.startswith("_")]
    assert any("n=" in label for label in named)
    assert any("final" in label and "/" in label for label in named)
    # The one-way `never` arm is the split arm: both subgroup legend entries must appear.
    assert any("stuck" in label for label in named)
    assert any("non-stuck" in label for label in named)
    # The two-way `never` panel has no split and must say so explicitly rather than
    # silently drawing one line where the sibling panel draws two.
    two_way_never_labels = [line.get_label() for line in figure.axes[1].get_lines()]
    assert any("no stranding here" in label for label in two_way_never_labels)
