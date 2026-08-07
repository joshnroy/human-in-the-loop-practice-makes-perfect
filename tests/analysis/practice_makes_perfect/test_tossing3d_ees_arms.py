"""Tests for the three-arm Tossing3D EES analysis.

The module under test applies the decision rule pre-registered in
`docs/experiment-logs/2026-08-06-tossing3d-ees-first-real.md`, so what is worth pinning
is that each cell fires on data built to land in it and — the part that actually matters
— that it *declines* on data that misses. A rule which quietly widens to fit whatever was
measured is not a pre-registration.

Four of these guard specific ways this particular experiment could go wrong rather than
merely be imprecise:

1. **The MDE is a real gate, not decoration.** A gap that is large but under the MDE on
   its own two denominators must not be reported as learning. Pinned by two runs whose
   only difference is the denominator.
2. **`undecided` is reachable.** If nothing lands there the rule cannot decline.
3. **The verdict never reads task success.** Tossing3D has a measured same-seed swing of
   at least 10 pp and an underpowered evaluation set, so a verdict that moved when
   episode counts moved would be worthless here. Pinned by changing nothing but
   `evaluations`.
4. **The exact tests are exact.** Fisher's two-sided p is checked against textbook
   values, because it is hand-rolled — scipy is not a dependency of this project.
"""

import math
from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_ees_arms import (
    Tossing3DEesArms,
    UnpairedTests,
)
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics

_TARGET = "MoveToThrowPose"


def _tally(
    *, attempts: int, successes: int, informed: int = 0, informed_successes: int = 0
) -> SkillPracticeTally:
    return SkillPracticeTally(
        num_attempts=attempts,
        num_successes=successes,
        num_informed_attempts=informed,
        num_informed_successes=informed_successes,
    )


def _run(*, tally: SkillPracticeTally, solved: int = 2, total: int = 10) -> Metrics:
    """One seed's Metrics carrying `tally` in a single practice window.

    One window rather than several on purpose: splitting a tally across windows needs
    integer division, which silently moves the very success *rates* these tests exist to
    pin. The module under test sums windows before it decides anything, so one window
    exercises the same path.
    """
    metrics = Metrics()
    for index in range(2):
        metrics.record_evaluation(
            num_online_transitions=index * 3, num_solved=solved, num_total=total
        )
    metrics.record_practice_outcomes(outcomes={_TARGET: tally})
    return metrics


def _curve(*, solved: list[int], transitions: list[int], total: int = 10) -> Metrics:
    """One seed's Metrics carrying a full evaluation curve on an explicit transition grid.

    `transitions` is given per seed rather than derived, because the defect these tests
    pin is precisely that Tossing3D's seeds do *not* share a transition grid: `Toss`
    deletes `Reachable`, so a practice period ends after one throw and each seed spends a
    different number of transitions reaching the same cycle.
    """
    metrics = Metrics()
    for online, num_solved in zip(transitions, solved, strict=True):
        metrics.record_evaluation(
            num_online_transitions=online, num_solved=num_solved, num_total=total
        )
    return metrics


class TestExactTests:
    def test_fisher_matches_the_tea_tasting_table(self) -> None:
        """Fisher's own 2x2. The two-sided p is 0.4857142857..., exactly 17/35."""
        assert UnpairedTests.fisher_exact(table=((3, 1), (1, 3))) == pytest.approx(
            17 / 35, abs=1e-12
        )

    def test_fisher_matches_a_strongly_associated_table(self) -> None:
        assert UnpairedTests.fisher_exact(table=((1, 9), (11, 3))) == pytest.approx(
            0.0027594, abs=1e-6
        )

    def test_fisher_of_an_independent_table_is_one(self) -> None:
        assert UnpairedTests.fisher_exact(table=((5, 5), (5, 5))) == pytest.approx(1.0)

    def test_fisher_is_symmetric_under_transpose(self) -> None:
        straight = UnpairedTests.fisher_exact(table=((2, 8), (7, 3)))
        transposed = UnpairedTests.fisher_exact(table=((2, 7), (8, 3)))
        assert straight == pytest.approx(transposed)

    def test_mde_uses_the_pre_registered_constant_and_both_denominators(self) -> None:
        """2.801585 * sqrt(pbar (1 - pbar) (1/n1 + 1/n2)) -- the pre-registered form."""
        expected = 2.801585 * math.sqrt(0.25 * 0.75 * (1 / 100 + 1 / 100))
        assert Tossing3DEesArms.minimum_detectable_effect(
            p_bar=0.25, n_1=100, n_2=100
        ) == pytest.approx(expected)

    def test_mde_shrinks_as_the_denominators_grow(self) -> None:
        small = Tossing3DEesArms.minimum_detectable_effect(p_bar=0.3, n_1=100, n_2=100)
        large = Tossing3DEesArms.minimum_detectable_effect(p_bar=0.3, n_1=400, n_2=400)
        assert large < small


class TestVerdict:
    def test_zero_informed_draws_is_the_regressed_cell(self) -> None:
        """The pre-flight gate, re-checked on the full sweep: no informed draw at all
        means the #123 fix is not doing what its own probe measured."""
        ees = [_run(tally=_tally(attempts=40, successes=40, informed=0))] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "regressed"

    def test_informed_draws_beating_the_arms_own_uniform_draws_is_learning(self) -> None:
        """Amendment 1's reference: EES's own non-informed draws, not the
        `random-skills` arm, which records no practice outcomes at all."""
        ees = [
            _run(tally=_tally(attempts=40, successes=26, informed=20, informed_successes=18))
        ] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "learns the constant"

    def test_a_modest_gap_clears_the_mde_only_on_the_larger_denominator(self) -> None:
        """`180/500` informed against `120/500` uniform is the same 0.36-vs-0.24 as
        `18/50` against `12/50`. The *gap* is identical; the MDE on those denominators is
        not. This is the test that stops a rate difference being reported as an effect the
        data is too thin to support -- the failure mode this experiment exists to avoid.
        """
        one_run = _run(tally=_tally(attempts=100, successes=30, informed=50, informed_successes=18))
        assert Tossing3DEesArms.verdict(ees_runs=[one_run] * 10)[0] == "learns the constant"
        assert (
            Tossing3DEesArms.verdict(ees_runs=[one_run])[0]
            == "consulted but no better than uniform"
        )

    def test_consulted_but_no_better_than_uniform_is_its_own_cell(self) -> None:
        """The sampler discriminates in quantity and its belief buys nothing: its
        informed draws land at exactly the rate its uniform draws do."""
        ees = [
            _run(tally=_tally(attempts=40, successes=10, informed=20, informed_successes=5))
        ] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "consulted but no better than uniform"

    def test_occasional_informed_draws_are_the_starved_cell(self) -> None:
        ees = [
            _run(tally=_tally(attempts=100, successes=30, informed=5, informed_successes=3))
        ] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "starved"

    def test_undecided_is_reachable(self) -> None:
        """A rule with no way to decline forces every run into a conclusion."""
        ees = [_run(tally=_tally(attempts=0, successes=0))] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "undecided"

    def test_all_draws_informed_leaves_no_reference_and_is_undecided(self) -> None:
        """With no non-informed draw there is nothing to compare the informed ones to.
        Reporting that as learning would be comparing a rate against itself."""
        ees = [
            _run(tally=_tally(attempts=40, successes=30, informed=40, informed_successes=30))
        ] * 10
        label, _ = Tossing3DEesArms.verdict(ees_runs=ees)
        assert label == "undecided"

    def test_the_verdict_does_not_read_task_success(self) -> None:
        """Same practice record, opposite episode counts, identical verdict. Tossing3D's
        task-success axis is provisional twice over -- a measured same-seed swing of at
        least 10 pp, and an MDE of about 17 pp on the evaluation set -- so a verdict that
        moved with it would carry that provisionality into a structural claim."""
        tally = _tally(attempts=40, successes=26, informed=20, informed_successes=18)
        never = Tossing3DEesArms.verdict(ees_runs=[_run(tally=tally, solved=0, total=10)] * 10)
        always = Tossing3DEesArms.verdict(ees_runs=[_run(tally=tally, solved=10, total=10)] * 10)
        assert never == always


class TestReporting:
    def test_pooled_counts_keep_their_denominator(self) -> None:
        runs = [
            _run(tally=_tally(attempts=40, successes=32, informed=20, informed_successes=18))
        ] * 3
        pooled = Tossing3DEesArms.pooled_tally(runs=runs, skill_name=_TARGET)
        assert (pooled.num_successes, pooled.num_attempts) == (96, 120)

    def test_per_seed_success_is_reported_per_seed_not_only_pooled(self) -> None:
        """Ten seeds spanning 0/14 to 14/14 on Tossing Room produced a pooled number
        that described no seed that was run. Per-seed spread is not optional here."""
        runs = [
            _run(tally=_tally(attempts=4, successes=1), solved=solved, total=10)
            for solved in (0, 3, 9)
        ]
        assert Tossing3DEesArms.per_seed_success(runs=runs, index=-1) == [
            (0, 10),
            (3, 10),
            (9, 10),
        ]

    def test_pre_practice_and_end_of_training_are_different_checkpoints(self) -> None:
        metrics = Metrics()
        metrics.record_evaluation(num_online_transitions=0, num_solved=1, num_total=10)
        metrics.record_evaluation(num_online_transitions=50, num_solved=7, num_total=10)
        assert Tossing3DEesArms.pooled_success(runs=[metrics], index=0) == (1, 10)
        assert Tossing3DEesArms.pooled_success(runs=[metrics], index=-1) == (7, 10)

    def test_plot_writes_all_three_figures(self, *, tmp_path: Path) -> None:
        """Three separate files, not three panels of one. Each of the two learning-curve
        axes is its own graph; task success is the third."""
        cycles = tmp_path / "cycles.png"
        transitions = tmp_path / "transitions.png"
        task_success = tmp_path / "task-success.png"
        Tossing3DEesArms.plot(
            arms=_two_arm_curves(),
            cycles_path=cycles,
            transitions_path=transitions,
            task_success_path=task_success,
        )
        for output in (cycles, transitions, task_success):
            assert output.exists() and output.stat().st_size > 0, f"{output.name} not written"


def _two_arm_curves() -> dict[str, list[Metrics]]:
    """Two learning arms whose cycle grids match and whose transition grids do not.

    This is the shape of the real sweep in miniature: both arms ran `--num-cycles 20`
    and so have the same number of checkpoints, but `ees` reaches the last one in far
    fewer transitions than `random-skills`. `skill-oracle` has a single evaluation and
    no practice at all, which is why it is a ceiling line rather than a curve.
    """
    return {
        "ees": [
            _curve(solved=[1, 4, 8], transitions=[0, 40, 80]),
            _curve(solved=[0, 3, 7], transitions=[0, 45, 85]),
        ],
        "random-skills": [
            _curve(solved=[1, 2, 2], transitions=[0, 70, 140]),
            _curve(solved=[0, 1, 3], transitions=[0, 80, 170]),
        ],
        "skill-oracle": [_curve(solved=[10], transitions=[0])] * 2,
    }


class TestLearningCurveAxis:
    """The cycles figure plots against *cycles*, the controlled variable.

    Both arms run the same 21 evaluation checkpoints, but `Toss` deletes `Reachable`, so a
    practice period ends after one throw: `ees` plans `Pick -> MoveToThrowPose -> Toss` in
    about 3-4 transitions per cycle where `random-skills` flails through about 7. On the
    cycle axis the arms therefore align and are compared like with like.

    The transitions axis is not a defect to be removed but a second view, drawn as its own
    graph -- see `TestTransitionsAxis`. Both are kept because on *this* domain they genuinely
    differ; on Tossing Room every run charged exactly 150.0 transitions per cycle, so there
    the two axes were the same curve with different tick labels and only one was drawn.
    """

    def test_the_cycle_grid_is_the_checkpoint_index(self) -> None:
        runs = [
            _curve(solved=[1, 4, 8], transitions=[0, 40, 80]),
            _curve(solved=[0, 3, 7], transitions=[0, 999, 1234]),
        ]
        assert Tossing3DEesArms.cycle_grid(runs=runs) == [0, 1, 2]

    def test_seeds_with_different_checkpoint_counts_raise(self) -> None:
        """The replaced code took `min(len(m.evaluations) for m in runs)`, which silently
        shortened the pooled curve to the shortest seed. Every seed has 21 here so it never
        bit, but a mean that quietly drops a seed's tail is invisible in the drawn line.
        Raising makes an unequal sweep loud instead."""
        runs = [
            _curve(solved=[1, 4, 8], transitions=[0, 40, 80]),
            _curve(solved=[0, 3], transitions=[0, 45]),
        ]
        with pytest.raises(ValueError, match="different numbers of evaluation checkpoints"):
            Tossing3DEesArms.cycle_grid(runs=runs)

    def test_both_arms_span_the_same_x_axis(self) -> None:
        """The defect, stated as an assertion. On a transitions axis the two arms' mean
        lines ended at about 84 and about 144; on the cycle axis they must end together."""
        figure = Tossing3DEesArms.figure_cycles(arms=_two_arm_curves())
        axis = figure.axes[0]
        # The bold means are the only labelled *curves*; per-seed lines are unlabelled
        # (matplotlib gives those a `_child`-prefixed label) and the oracle is a flat
        # ceiling drawn by `axhline`, which is excluded here by name.
        means = [
            line
            for line in axis.lines
            if str(line.get_label()).startswith(("ees", "random-skills"))
        ]
        assert len(means) == 2, f"expected one bold mean per learning arm, got {len(means)}"
        spans = {(line.get_xdata()[0], line.get_xdata()[-1]) for line in means}
        assert len(spans) == 1, f"arms span different x ranges: {spans}"
        assert spans == {(0, 2)}

    def test_the_x_axis_is_labelled_as_cycles_not_transitions(self) -> None:
        figure = Tossing3DEesArms.figure_cycles(arms=_two_arm_curves())
        label = figure.axes[0].get_xlabel().lower()
        assert "cycle" in label
        assert "transition" not in label

    def test_mean_final_transitions_stay_available(self) -> None:
        """Transitions are no longer the axis, but they are still the efficiency story --
        so they are kept as a per-arm annotation rather than dropped."""
        runs = [
            _curve(solved=[1, 4, 8], transitions=[0, 40, 80]),
            _curve(solved=[0, 3, 7], transitions=[0, 45, 90]),
        ]
        assert Tossing3DEesArms.mean_final_transitions(runs=runs) == pytest.approx(85.0)

    def test_every_curve_legend_entry_carries_a_count(self) -> None:
        """`x/y` never a bare percentage, in the legend as much as in the prose."""
        figure = Tossing3DEesArms.figure_cycles(arms=_two_arm_curves())
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert labels, "the curve panel must carry a legend"
        for label in labels:
            assert "/" in label, f"legend entry without an x/y count: {label!r}"

    def test_the_oracle_ceiling_is_still_drawn(self) -> None:
        figure = Tossing3DEesArms.figure_cycles(arms=_two_arm_curves())
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert any("skill-oracle" in label for label in labels)


class TestTransitionsAxis:
    """The transitions figure is the second view, and it is a *separate graph*.

    On this domain the two axes are not proportional, which is the whole reason both are
    drawn. Measured from the committed `stats.json`: a practice period costs `ees` 4.19
    transitions against `random-skills` 7.20 averaged over 10 seeds, every seed sits on its
    own irregular grid (per-cycle steps run 1..20 within a single seed), and the arms finish
    at 69..101 against 92..174. Contrast Tossing Room, where every run charged exactly 150.0
    transitions per cycle and a cycles graph would have been the transitions graph with
    relabelled ticks.

    So the two axes answer different questions: against cycles the arms align and are
    compared like with like; against transitions EES's line ends earlier because it reached
    the same 21/21 checkpoints for fewer steps.
    """

    def test_the_x_axis_is_labelled_as_transitions(self) -> None:
        figure = Tossing3DEesArms.figure_transitions(arms=_two_arm_curves())
        label = figure.axes[0].get_xlabel().lower()
        assert "transition" in label

    def test_the_arms_end_at_different_x_because_ees_spends_fewer_steps(self) -> None:
        """The contrast that earns this second graph. `ees` means end at 82.5 transitions
        and `random-skills` at 155.0, where on the cycle axis both end at 2."""
        figure = Tossing3DEesArms.figure_transitions(arms=_two_arm_curves())
        means = _mean_lines(axis=figure.axes[0])
        ends = {str(line.get_label()).split(" ")[0]: line.get_xdata()[-1] for line in means}
        assert ends["ees"] == pytest.approx(82.5)
        assert ends["random-skills"] == pytest.approx(155.0)
        assert ends["ees"] < ends["random-skills"]

    def test_the_mean_curve_averages_the_x_positions_too(self) -> None:
        """Seeds do not share a transition grid, so a per-checkpoint mean has to average
        the x positions as well as the y. Pinned because silently reusing one seed's grid
        would draw a mean at transition counts no seed actually reached."""
        figure = Tossing3DEesArms.figure_transitions(arms=_two_arm_curves())
        means = _mean_lines(axis=figure.axes[0])
        ees = next(line for line in means if str(line.get_label()).startswith("ees"))
        # Seed grids are [0, 40, 80] and [0, 45, 85]; the mean grid is their elementwise mean.
        assert list(ees.get_xdata()) == pytest.approx([0.0, 42.5, 82.5])

    def test_mean_transitions_per_cycle_is_reported(self) -> None:
        """The per-period cost is the mechanism behind the two axes differing, so it is a
        number the figure can carry rather than a claim only the prose makes."""
        runs = [
            _curve(solved=[1, 4, 8], transitions=[0, 40, 80]),
            _curve(solved=[0, 3, 7], transitions=[0, 45, 85]),
        ]
        # 80/2 = 40.0 and 85/2 = 42.5, averaged over the two seeds.
        assert Tossing3DEesArms.mean_transitions_per_cycle(runs=runs) == pytest.approx(41.25)

    def test_every_curve_legend_entry_carries_a_count(self) -> None:
        figure = Tossing3DEesArms.figure_transitions(arms=_two_arm_curves())
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert labels, "the transitions graph must carry a legend"
        for label in labels:
            assert "/" in label, f"legend entry without an x/y count: {label!r}"

    def test_the_oracle_ceiling_is_still_drawn(self) -> None:
        figure = Tossing3DEesArms.figure_transitions(arms=_two_arm_curves())
        labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
        assert any("skill-oracle" in label for label in labels)


def _mean_lines(*, axis) -> list:  # type: ignore[no-untyped-def]
    """The bold per-arm mean lines on a curve axes.

    Per-seed lines are unlabelled (matplotlib gives those a `_child`-prefixed label) and the
    oracle ceiling is drawn by `axhline`, so matching on the arm name selects exactly the
    means.
    """
    return [
        line for line in axis.lines if str(line.get_label()).startswith(("ees", "random-skills"))
    ]


class TestPriorVersusBeliefPanelIsGone:
    """The `48/275` uniform against `117/206` informed comparison is still the headline
    result -- it stays in the log and the PR body as a number. It is no longer a chart."""

    def test_the_helper_is_removed(self) -> None:
        assert not hasattr(Tossing3DEesArms, "_plot_skill_rate")

    def test_each_figure_is_a_single_panel(self) -> None:
        """Renamed from `test_the_figure_has_two_panels`: the shape changed again. The
        dropped prior-versus-belief panel stays dropped, and the two remaining learning-curve
        views were split apart into their own graphs rather than sharing one canvas, so every
        figure this module produces is now exactly one axes."""
        arms = _two_arm_curves()
        for build in (
            Tossing3DEesArms.figure_cycles,
            Tossing3DEesArms.figure_transitions,
            Tossing3DEesArms.figure_task_success,
        ):
            assert len(build(arms=arms).axes) == 1, f"{build.__name__} is not a single panel"

    def test_the_verdict_still_reports_both_counts(self) -> None:
        """Dropping the panel must not drop the number it drew."""
        ees = [_run(tally=_tally(attempts=40, successes=26, informed=20, informed_successes=18))]
        _, evidence = Tossing3DEesArms.verdict(ees_runs=ees)
        assert "18/20" in evidence and "8/20" in evidence
