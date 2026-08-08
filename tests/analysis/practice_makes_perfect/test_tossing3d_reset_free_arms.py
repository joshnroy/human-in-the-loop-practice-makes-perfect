"""Covers the two-arm reset-free comparison rule.

Built from synthetic curves with a *known* shape, because the property under test is
"does this rule reach the right verdict", and only a fabricated curve has a verdict known
independently of the code. A test that read the real sweep would assert whatever the run
happened to do.

The two that matter are `test_two_arms_that_differ_are_not_called_a_null_result` and
`test_two_arms_that_do_not_differ_are_called_a_null_result`: a rule that cannot tell
those apart is the same class of instrument failure this whole PR stack exists to fix --
#178's arms were identical and nothing noticed. Everything else here is plumbing.
"""

import json
import statistics
from pathlib import Path

from analysis.practice_makes_perfect.paired_tests import PairedTests
from analysis.practice_makes_perfect.tossing3d_reset_free_arms import (
    NEVER,
    SCHEDULED,
    WINDOW,
    Tossing3DResetFree,
)

NUM_SWEEPS = 101
NUM_TASKS = 10


def _write_run(
    *,
    run_dir: Path,
    solved_per_sweep: list[int],
    transitions_per_cycle: list[int] | None = None,
) -> None:
    """One run's `stats.json`. `transitions_per_cycle` defaults to a steady 4 per cycle,
    i.e. a robot that never strands; pass zeros to model one that does."""
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = transitions_per_cycle or [4] * (len(solved_per_sweep) - 1)
    cumulative = [0]
    for step in steps:
        cumulative.append(cumulative[-1] + step)
    evaluations = [[cumulative[i], solved, NUM_TASKS] for i, solved in enumerate(solved_per_sweep)]
    (run_dir / "stats.json").write_text(json.dumps({"evaluations": evaluations}))


def _settled(*, level: int, jitter: list[int] | None = None) -> list[int]:
    """A curve that climbs early then sits at `level`, with optional per-sweep noise.

    The noise is not decoration: Tossing3D's real per-seed score moves by several tasks
    between adjacent sweeps with no learning event, and a rule that mistook that for an
    arm difference would be useless."""
    curve = [min(level, i) for i in range(15)] + [level] * (NUM_SWEEPS - 15)
    for offset, delta in enumerate(jitter or []):
        curve[-(offset + 1)] = max(0, min(NUM_TASKS, curve[-(offset + 1)] + delta))
    return curve


def _write_arms(*, root: Path, scheduled_levels: list[int], never_levels: list[int]) -> None:
    for seed, level in enumerate(scheduled_levels):
        _write_run(
            run_dir=root / SCHEDULED / "ees" / str(seed), solved_per_sweep=_settled(level=level)
        )
    for seed, level in enumerate(never_levels):
        _write_run(run_dir=root / NEVER / "ees" / str(seed), solved_per_sweep=_settled(level=level))


def test_load_arms_reads_both_arms_keyed_by_seed(*, tmp_path: Path) -> None:
    _write_arms(root=tmp_path, scheduled_levels=[8, 7], never_levels=[8, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    assert sorted(arms) == sorted([NEVER, SCHEDULED])
    assert sorted(arms[SCHEDULED]) == [0, 1]
    assert len(arms[SCHEDULED][0]) == NUM_SWEEPS


def test_late_scores_average_the_last_window_not_the_final_sweep(*, tmp_path: Path) -> None:
    """The whole rule rests on scoring a window rather than a sweep -- #178 measured a
    final sweep 5 tasks below its own late window on the same data."""
    curve = _settled(level=8, jitter=[-8])
    _write_run(run_dir=tmp_path / SCHEDULED / "ees" / "0", solved_per_sweep=curve)
    _write_run(run_dir=tmp_path / NEVER / "ees" / "0", solved_per_sweep=curve)
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    scores = Tossing3DResetFree.late_scores(curves=arms[SCHEDULED])
    assert curve[-1] == 0
    # Nine sweeps at 8 and one at 0.
    assert scores[0] == statistics.fmean([8] * (WINDOW - 1) + [0])


def test_two_arms_that_do_not_differ_are_called_a_null_result(*, tmp_path: Path) -> None:
    """The honest outcome when the manipulation does nothing -- and the one that must
    still come with an MDE, or it cannot be told apart from no power."""
    levels = [8, 7, 9, 8, 7, 8, 9, 7, 8, 8]
    _write_arms(root=tmp_path, scheduled_levels=levels, never_levels=levels)
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
    seeds = Tossing3DResetFree.shared_seeds(arms=arms)
    differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]

    assert differences == [0.0] * len(seeds)
    assert PairedTests.sign_flip(differences=differences).p_value == 1.0


def test_two_arms_that_differ_are_not_called_a_null_result(*, tmp_path: Path) -> None:
    """The failure this PR stack exists to prevent, in reverse: a real gap on every seed
    must come out significant, or the instrument cannot see the thing it is for."""
    scheduled = [8, 7, 9, 8, 7, 8, 9, 7, 8, 8]
    _write_arms(
        root=tmp_path,
        scheduled_levels=scheduled,
        never_levels=[level - 3 for level in scheduled],
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}
    seeds = Tossing3DResetFree.shared_seeds(arms=arms)
    differences = [late[NEVER][s] - late[SCHEDULED][s] for s in seeds]

    assert differences == [-3.0] * len(seeds)
    assert PairedTests.sign_flip(differences=differences).p_value < 0.05


def test_pairing_uses_only_seeds_present_in_both_arms(*, tmp_path: Path) -> None:
    """A lost run must shrink the pairing, not silently pair a seed against nothing."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 7, 9], never_levels=[8, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    assert Tossing3DResetFree.shared_seeds(arms=arms) == [0, 1]


def test_pooled_reports_a_count_over_the_seeds_denominator(*, tmp_path: Path) -> None:
    """`x/y`, never a bare percentage -- and `x` stays a float because each seed's
    contribution is a window mean, so rounding here would hide that."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 6], never_levels=[8, 6])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    x, y = Tossing3DResetFree.pooled(
        scores=Tossing3DResetFree.late_scores(curves=arms[SCHEDULED]), num_total=NUM_TASKS
    )
    assert (x, y) == (14.0, 20)


def test_report_says_so_rather_than_raising_when_an_arm_is_missing(
    *, tmp_path: Path, capsys
) -> None:
    """A half-finished sweep must not be reported as a result."""
    _write_run(run_dir=tmp_path / SCHEDULED / "ees" / "0", solved_per_sweep=_settled(level=8))
    Tossing3DResetFree.report(results_root=tmp_path)
    assert "No completed runs" in capsys.readouterr().out


def test_every_figure_is_written(*, tmp_path: Path) -> None:
    """A quantitative result needs a figure, so the figures are part of the deliverable
    rather than a manual step."""
    _write_arms(root=tmp_path, scheduled_levels=[8, 7, 9], never_levels=[6, 5, 7])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    outputs = [tmp_path / f"{name}.png" for name in ("curves", "paired", "practice")]

    Tossing3DResetFree.render_curves(arms=arms, output=outputs[0])
    Tossing3DResetFree.render_paired(arms=arms, output=outputs[1])
    Tossing3DResetFree.render_practice(arms=arms, output=outputs[2])

    assert all(output.stat().st_size > 0 for output in outputs)


def test_the_committed_experiment_log_layout_loads_too(*, tmp_path: Path) -> None:
    """`run_sweep` writes `<arm>/<method>/<seed>/`; a committed `docs/experiment-logs/`
    tree is `<arm>/<seed>/`. A loader fixed to one depth finds nothing under the other,
    and `report` then prints "No completed runs" and exits 0 -- a silent wrong answer
    rather than a failure, which is the same shape of defect this stack exists to fix."""
    for arm in (SCHEDULED, NEVER):
        for seed in (0, 1):
            _write_run(run_dir=tmp_path / arm / str(seed), solved_per_sweep=_settled(level=8))

    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)

    assert sorted(arms) == sorted([NEVER, SCHEDULED])
    assert Tossing3DResetFree.shared_seeds(arms=arms) == [0, 1]


def test_seeds_are_not_collided_across_arms(*, tmp_path: Path) -> None:
    """Keying on the containing directory alone would fold `scheduled/0` and `never/0`
    into one entry. That collision is invisible for exactly as long as the two arms
    agree -- the condition that produced this experiment."""
    _write_arms(root=tmp_path, scheduled_levels=[9, 9], never_levels=[3, 3])
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)
    late = {arm: Tossing3DResetFree.late_scores(curves=arms[arm]) for arm in arms}

    assert late[SCHEDULED][0] == 9.0
    assert late[NEVER][0] == 3.0


def test_transitions_per_cycle_is_the_gap_between_consecutive_sweeps(*, tmp_path: Path) -> None:
    """Sweep 0 happens before any practice, so cycle i's cost is the difference between
    consecutive sweeps' `num_online_transitions`."""
    steps = [5, 3, 0, 0]
    _write_run(
        run_dir=tmp_path / SCHEDULED / "ees" / "0",
        solved_per_sweep=[0] * (len(steps) + 1),
        transitions_per_cycle=steps,
    )
    arms = Tossing3DResetFree.load_arms(results_root=tmp_path)

    assert Tossing3DResetFree.transitions_per_cycle(evaluations=arms[SCHEDULED][0]) == steps


def test_a_robot_that_stops_acting_is_reported_as_stranded() -> None:
    """The measurement that separates "learned less" from "practised less". Without it,
    an arm that stopped acting after cycle 2 and an arm that practised badly for 100
    cycles produce the same score gap and the same conclusion."""
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 4, 0, 0, 0]) == 2
    assert Tossing3DResetFree.stranding_onset(transitions=[0, 0, 0]) == 0


def test_a_single_idle_cycle_is_not_stranding() -> None:
    """Terminal-from-here, not "the first gap" -- the same definition
    `pickup_weight_stranding.py` uses, so the two experiments read side by side. A run
    that pauses and resumes was never stranded, and calling it stranded would promote
    ordinary exploration noise into the effect being claimed."""
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 0, 4, 4]) is None
    assert Tossing3DResetFree.stranding_onset(transitions=[4, 0, 0, 4]) is None
