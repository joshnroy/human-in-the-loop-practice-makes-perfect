"""Tests for the Tossing3D practice diagnosis.

The module under test decides which cell of a **pre-registered** decision rule a run
falls into, so the thing worth pinning is that each cell fires on data constructed to
land in it, and — more importantly — that it does *not* fire on data that misses it. A
rule that quietly widens to fit whatever was measured is not a pre-registration.

Three of these guard specific ways a diagnosis could be wrong rather than merely
imprecise:

1. **The control gate.** Zero informed draws on the target means nothing if the
   instrument cannot see informed draws at all. That must return "no verdict", not H3.
2. **`undecided` is reachable.** If no constructed input lands there, the rule has no
   way to decline, and every run would be forced into a conclusion.
3. **The verdict never reads task success.** Tossing3D's episode counts are provisional
   twice over, so a verdict that moved when they moved would be worthless. Pinned by
   changing nothing but `evaluations` and asserting the verdict is identical.
"""

from pathlib import Path

import pytest

from analysis.practice_makes_perfect.tossing3d_practice_diagnosis import (
    Tossing3DPracticeDiagnosis,
)
from hitl_pmp.core.method.types import SkillPracticeTally
from hitl_pmp.core.metrics.metrics import Metrics

_TARGET = "MoveToThrowPose"
_CONTROL = "Pick"


def _metrics(
    *,
    windows: list[dict[str, SkillPracticeTally]],
    solved: int = 2,
    total: int = 10,
) -> Metrics:
    metrics = Metrics()
    for index in range(len(windows)):
        metrics.record_evaluation(
            num_online_transitions=index * 3, num_solved=solved, num_total=total
        )
    for window in windows:
        metrics.record_practice_outcomes(outcomes=window)
    return metrics


def _informed_control(*, num: int = 5) -> SkillPracticeTally:
    return SkillPracticeTally(
        num_attempts=num,
        num_successes=num,
        num_informed_attempts=num,
        num_informed_successes=num,
    )


def _h3_run() -> Metrics:
    """Every target attempt labelled a success, never an informed draw, and a control
    that does make them — the shape H3 predicts."""
    window = {
        _TARGET: SkillPracticeTally(num_attempts=8, num_successes=8),
        _CONTROL: _informed_control(),
    }
    return _metrics(windows=[window, window, {}])


def test_pooled_sums_a_skill_over_every_window_and_seed() -> None:
    runs = [_h3_run(), _h3_run()]
    pooled = Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=_TARGET)
    assert (pooled.num_successes, pooled.num_attempts) == (32, 32)


def test_a_skill_absent_from_a_run_contributes_nothing_rather_than_raising() -> None:
    runs = [_metrics(windows=[{_CONTROL: _informed_control()}])]
    assert Tossing3DPracticeDiagnosis.pooled(runs=runs, skill_name=_TARGET).num_attempts == 0


def test_the_h3_cell_fires_when_every_label_is_positive_and_nothing_is_informed() -> None:
    cell, reasoning = Tossing3DPracticeDiagnosis.verdict(runs=[_h3_run()])
    assert cell.startswith("H3")
    assert "16/16" in reasoning


def test_a_control_that_is_also_never_informed_is_an_instrument_fault_not_a_finding() -> None:
    """The single most important negative here: without the control, "0/16 informed" is
    equally consistent with a broken instrument, and reporting H3 off it would be a
    conclusion drawn from a measurement that did not work."""
    window = {
        _TARGET: SkillPracticeTally(num_attempts=8, num_successes=8),
        _CONTROL: SkillPracticeTally(num_attempts=5, num_successes=5),
    }
    cell, reasoning = Tossing3DPracticeDiagnosis.verdict(runs=[_metrics(windows=[window, {}])])
    assert cell == "instrument fault, no verdict"
    assert "not evidence" in reasoning


def test_the_never_asked_cell_fires_when_the_skill_was_never_practiced() -> None:
    runs = [_metrics(windows=[{_CONTROL: _informed_control()}, {}])]
    cell, _reasoning = Tossing3DPracticeDiagnosis.verdict(runs=runs)
    assert cell == "never asked"


def test_the_starvation_cell_fires_when_informed_draws_rise_as_labels_accumulate() -> None:
    """Mixed labels, almost never informed, but the informed count climbing — a
    classifier accumulating data and only beginning to express a belief."""
    runs = [
        _metrics(
            windows=[
                {
                    _TARGET: SkillPracticeTally(num_attempts=40, num_successes=8),
                    _CONTROL: _informed_control(),
                },
                {
                    _TARGET: SkillPracticeTally(
                        num_attempts=40, num_successes=10, num_informed_attempts=2
                    ),
                    _CONTROL: _informed_control(),
                },
                {},
            ]
        )
    ]
    cell, _reasoning = Tossing3DPracticeDiagnosis.verdict(runs=runs)
    assert cell == "starvation"


def test_the_inability_cell_fires_when_informed_draws_land_at_the_uniform_rate() -> None:
    """Consulted in quantity, states a belief, and the belief is worth nothing:
    20/100 informed successes against the 543/2700 uniform-draw rate."""
    runs = [
        _metrics(
            windows=[
                {
                    _TARGET: SkillPracticeTally(
                        num_attempts=100,
                        num_successes=20,
                        num_informed_attempts=100,
                        num_informed_successes=20,
                    ),
                    _CONTROL: _informed_control(),
                },
                {},
            ]
        )
    ]
    cell, reasoning = Tossing3DPracticeDiagnosis.verdict(runs=runs)
    assert cell == "inability"
    assert "543/2700" in reasoning


def test_informed_draws_well_above_the_uniform_rate_are_not_called_inability() -> None:
    """A classifier that is consulted and *is* better than its prior is neither starved
    nor unable, and must fall through to undecided rather than be labelled a failure."""
    runs = [
        _metrics(
            windows=[
                {
                    _TARGET: SkillPracticeTally(
                        num_attempts=100,
                        num_successes=80,
                        num_informed_attempts=100,
                        num_informed_successes=80,
                    ),
                    _CONTROL: _informed_control(),
                },
                {},
            ]
        )
    ]
    assert Tossing3DPracticeDiagnosis.verdict(runs=runs)[0] == "undecided"


def test_undecided_is_reachable_from_the_gap_between_the_cells() -> None:
    """A rule with no way to decline would force every run into a conclusion."""
    runs = [
        _metrics(
            windows=[
                {
                    _TARGET: SkillPracticeTally(
                        num_attempts=100, num_successes=70, num_informed_attempts=15
                    ),
                    _CONTROL: _informed_control(),
                },
                {},
            ]
        )
    ]
    cell, reasoning = Tossing3DPracticeDiagnosis.verdict(runs=runs)
    assert cell == "undecided"
    assert "no conclusion is supported" in reasoning


def test_one_informed_draw_in_any_window_falsifies_the_flat_at_zero_clause() -> None:
    """H3 claims the classifier *never* discriminates. A single informed draw anywhere
    must break it, or the claim is weaker than it is written to be."""
    runs = [
        _metrics(
            windows=[
                {
                    _TARGET: SkillPracticeTally(num_attempts=100, num_successes=100),
                    _CONTROL: _informed_control(),
                },
                {
                    _TARGET: SkillPracticeTally(
                        num_attempts=100,
                        num_successes=100,
                        num_informed_attempts=1,
                        num_informed_successes=1,
                    ),
                    _CONTROL: _informed_control(),
                },
                {},
            ]
        )
    ]
    assert not Tossing3DPracticeDiagnosis.verdict(runs=runs)[0].startswith("H3")


def test_the_verdict_does_not_read_task_success() -> None:
    """Tossing3D is currently not reproducible from --seed and #102 changed the no-op
    path underneath the existing numbers, so a verdict that moved with the episode
    counts would be overturned by a re-run. Same practice record, opposite scores."""
    window = {
        _TARGET: SkillPracticeTally(num_attempts=8, num_successes=8),
        _CONTROL: _informed_control(),
    }
    dismal = _metrics(windows=[window, window, {}], solved=0, total=10)
    excellent = _metrics(windows=[window, window, {}], solved=10, total=10)
    assert Tossing3DPracticeDiagnosis.verdict(runs=[dismal]) == (
        Tossing3DPracticeDiagnosis.verdict(runs=[excellent])
    )


def test_informed_per_window_sums_across_seeds_rather_than_averaging() -> None:
    """One informed draw in one seed must survive into the series; a mean over ten
    seeds would round it to 0.1 and a threshold could swallow it.

    Two windows in, one out: the trailing evaluation-only bucket holds no practice by
    construction, so including it would make the rule's "rising" clause compare the last
    real cycle against a structural zero."""
    quiet = _metrics(windows=[{_TARGET: SkillPracticeTally(num_attempts=5, num_successes=5)}, {}])
    loud = _metrics(
        windows=[
            {
                _TARGET: SkillPracticeTally(
                    num_attempts=5,
                    num_successes=5,
                    num_informed_attempts=1,
                    num_informed_successes=1,
                )
            },
            {},
        ]
    )
    assert Tossing3DPracticeDiagnosis.informed_per_window(
        runs=[quiet, loud], skill_name=_TARGET
    ) == [1]


def test_the_report_states_task_success_as_context_and_says_so(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    Tossing3DPracticeDiagnosis.print_report(runs=[_h3_run()])
    out = capsys.readouterr().out
    assert "NOT an input to the verdict" in out
    assert "543/2700" in out
    assert "16/16" in out


def test_plotting_writes_a_figure(*, tmp_path: Path) -> None:
    output = tmp_path / "figures" / "diagnosis.png"
    Tossing3DPracticeDiagnosis.plot(runs=[_h3_run()], output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_loading_reads_every_seed_under_every_method_directory(*, tmp_path: Path) -> None:
    method_dir = tmp_path / "ees"
    for seed in (0, 1):
        seed_dir = method_dir / str(seed)
        seed_dir.mkdir(parents=True)
        (seed_dir / "stats.json").write_text(_h3_run().model_dump_json())
    runs = Tossing3DPracticeDiagnosis.load(results_root=tmp_path)
    assert len(runs) == 2
