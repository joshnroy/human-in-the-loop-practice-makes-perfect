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

    def test_plot_writes_a_figure(self, *, tmp_path: Path) -> None:
        arms = {
            "ees": [
                _run(tally=_tally(attempts=40, successes=32, informed=20, informed_successes=18))
            ]
            * 3,
            "random-skills": [_run(tally=_tally(attempts=40, successes=10))] * 3,
            "skill-oracle": [_run(tally=_tally(attempts=0, successes=0), solved=10)] * 3,
        }
        output = tmp_path / "figure.png"
        Tossing3DEesArms.plot(arms=arms, output_path=output)
        assert output.exists() and output.stat().st_size > 0
