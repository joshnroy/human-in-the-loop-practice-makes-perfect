"""Tests for the per-window practice/planning diagnostics analysis.

Pinned against hand-built `Metrics` whose answer is known by construction, because the
two things this module can silently get wrong are invisible in the rendered figure:

1. **Where a bucket is dated.** `on_cycle_end` fires *before* each evaluation sweep, so
   bucket `i` describes the practice that ran between `evaluations[i]` and
   `evaluations[i + 1]`. Dating it at `evaluations[i]` shifts a whole practice period
   one checkpoint left, which on a short run is the entire effect.
2. **What a seed that never practiced a skill contributes.** Zeros, not nothing --
   dropping it shrinks the mean's denominator and flatters whichever seeds did.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.practice_diagnostics import PracticeDiagnostics
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics


def _metrics(*, windows: list[dict[str, SkillPracticeTally]], step: int = 100) -> Metrics:
    """A run with one evaluation per window, transitions rising by `step` per cycle --
    the shape `PracticeLoop` produces with `num_cycles = len(windows) - 1`."""
    metrics = Metrics()
    for index in range(len(windows)):
        metrics.record_evaluation(
            num_online_transitions=index * step, num_solved=index, num_total=len(windows)
        )
    for window in windows:
        metrics.record_practice_outcomes(outcomes=window)
        metrics.record_planning_outcomes(num_failures=1, num_attempts=10)
    return metrics


def test_a_window_is_dated_at_the_transition_count_its_practice_ended_at() -> None:
    """Bucket i covers sweep i and the practice after it, so it belongs at
    evaluations[i + 1]. The trailing bucket has no practice in it and stays at the
    final count."""
    metrics = _metrics(windows=[{}, {}, {}])
    assert [entry[0] for entry in metrics.evaluations] == [0, 100, 200]
    assert PracticeDiagnostics.window_transitions(metrics=metrics) == [100, 200, 200]


def test_window_transitions_is_empty_for_a_run_with_no_evaluations() -> None:
    assert PracticeDiagnostics.window_transitions(metrics=Metrics()) == []


def test_the_practice_panels_stop_before_the_trailing_evaluation_only_bucket() -> None:
    """That bucket holds no practice by construction and sits at the same transition
    count as the one before it, so drawing it puts a structural zero at a duplicated x
    -- a cliff at the right-hand edge that reads as a collapse."""
    assert PracticeDiagnostics.practice_window_count(metrics=_metrics(windows=[{}, {}, {}])) == 2


def test_practice_window_count_never_goes_negative_on_an_unrecorded_run() -> None:
    assert PracticeDiagnostics.practice_window_count(metrics=Metrics()) == 0


def test_skill_names_are_the_union_over_seeds() -> None:
    """A skill only some seeds ever reached is exactly what is worth seeing, so it must
    not be dropped because seed 0 never got there."""
    runs = [
        _metrics(windows=[{"Throw": SkillPracticeTally(num_attempts=1)}]),
        _metrics(windows=[{"Move": SkillPracticeTally(num_attempts=1)}]),
    ]
    assert PracticeDiagnostics.skill_names(runs=runs) == ["Move", "Throw"]


def test_a_seed_that_never_practiced_a_skill_contributes_zeros_not_nothing() -> None:
    """Dropping it would shrink the mean's denominator and flatter the seeds that did."""
    runs = [
        _metrics(windows=[{"Throw": SkillPracticeTally(num_attempts=4)}, {}]),
        _metrics(windows=[{}, {}]),
    ]
    series = PracticeDiagnostics.per_seed_series(
        runs=runs, skill_name="Throw", field="num_attempts"
    )
    assert series == [[4, 0], [0, 0]]
    assert PracticeDiagnostics.mean_series(series=series) == [2.0, 0.0]


def test_mean_series_truncates_to_the_shortest_seed() -> None:
    """A partially-finished run must not lengthen or skew the others."""
    assert PracticeDiagnostics.mean_series(series=[[1, 2, 3], [3, 4]]) == [2.0, 3.0]


def test_mean_series_of_nothing_is_empty() -> None:
    assert PracticeDiagnostics.mean_series(series=[]) == []


def test_totals_pool_attempts_across_seeds_rather_than_averaging_rates() -> None:
    """A mean of per-seed rates weights a two-attempt seed like a two-hundred-attempt
    one, which is the same denominator-hiding this project's counts rule forbids."""
    runs = [
        _metrics(
            windows=[
                {
                    "Throw": SkillPracticeTally(
                        num_attempts=2,
                        num_successes=2,
                        num_informed_attempts=2,
                        num_informed_successes=2,
                    )
                }
            ]
        ),
        _metrics(
            windows=[
                {
                    "Throw": SkillPracticeTally(
                        num_attempts=200,
                        num_successes=20,
                        num_informed_attempts=100,
                        num_informed_successes=5,
                    )
                }
            ]
        ),
    ]
    pooled = PracticeDiagnostics.totals(runs=runs)["Throw"]
    assert (pooled.num_successes, pooled.num_attempts) == (22, 202)
    assert (pooled.num_informed_successes, pooled.num_informed_attempts) == (7, 102)


def test_the_printed_table_reports_counts_never_bare_percentages(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = {
        "ees": [
            _metrics(
                windows=[
                    {
                        "Throw": SkillPracticeTally(
                            num_attempts=17,
                            num_successes=3,
                            num_random_attempts=8,
                            num_random_successes=1,
                            num_informed_attempts=6,
                            num_informed_successes=1,
                        )
                    }
                ]
            )
        ]
    }
    PracticeDiagnostics.print_table(summary=summary)
    out = capsys.readouterr().out
    assert "3/17" in out
    assert "1/6" in out
    assert "1/8" in out
    # attempts - random - informed = 3, successes - 1 - 1 = 1
    assert "1/3" in out
    assert "%" not in out


def test_the_table_separates_a_parameter_free_skill_from_an_uninformative_sampler(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Tossing3D pair, as it would now print. `Toss` has no sampler and `MoveToThrow`
    has one that never discriminated; #111 printed both as `0/9` fallback and an operator
    reading `stats.json` had no way to tell that the two need opposite fixes."""
    summary = {
        "ees": [
            _metrics(
                windows=[
                    {
                        "Toss": SkillPracticeTally(num_attempts=9, num_unparameterized_attempts=9),
                        "MoveToThrow": SkillPracticeTally(num_attempts=9),
                    }
                ]
            )
        ]
    }
    PracticeDiagnostics.print_table(summary=summary)
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.strip()}

    assert "no sampler" in lines["skill"] and "uninformative" in lines["skill"]
    # Both are 0/9 overall and 0/9 fallback; only the two new columns tell them apart.
    assert lines["Toss"].endswith("0/9             0/0")
    assert lines["MoveToThrow"].endswith("0/0             0/9")


def test_a_method_that_measures_no_practice_says_so_rather_than_printing_an_empty_table(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "Nothing recorded" and "practiced nothing" are different findings, and a blank
    table reads as the second."""
    PracticeDiagnostics.print_table(summary={"skill-oracle": [_metrics(windows=[{}])]})
    assert "does not measure practice" in capsys.readouterr().out


def test_loading_a_sweep_directory_reads_one_metrics_per_seed(*, tmp_path: Path) -> None:
    method_dir = tmp_path / "ees"
    for seed in (0, 1):
        seed_dir = method_dir / str(seed)
        seed_dir.mkdir(parents=True)
        metrics = _metrics(windows=[{"Throw": SkillPracticeTally(num_attempts=seed + 1)}])
        (seed_dir / "stats.json").write_text(metrics.model_dump_json())
    summary = PracticeDiagnostics.summarize(results_root=tmp_path)
    assert list(summary) == ["ees"]
    assert PracticeDiagnostics.totals(runs=summary["ees"])["Throw"].num_attempts == 3


def test_plotting_writes_a_figure(*, tmp_path: Path) -> None:
    """A PR that produces an artifact should show one, so the artifact has to render."""
    summary = {"ees": [_metrics(windows=[{"Throw": SkillPracticeTally(num_attempts=2)}, {}])]}
    output = tmp_path / "figures" / "diagnostics.png"
    PracticeDiagnostics.plot(summary=summary, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_plotting_an_empty_sweep_is_an_error_rather_than_a_blank_page(*, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to plot"):
        PracticeDiagnostics.plot(summary={}, output_path=tmp_path / "empty.png")
