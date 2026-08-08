"""Covers the plateau-versus-climb rule.

Built from synthetic curves with a *known* shape, because the property under test is
"does this rule reach the right verdict", and only a fabricated curve has a verdict known
independently of the code. A test that read the real sweep would assert whatever the run
happened to do.

The two that matter are `test_a_flat_curve_is_not_called_a_climb` and
`test_a_climbing_curve_is_called_a_climb`: a rule that cannot tell those apart answers
Josh's question wrongly in one direction or the other, and everything else here is
plumbing.
"""

import json
from pathlib import Path

from analysis.practice_makes_perfect.tossing3d_plateau import (
    REFERENCE_START_CYCLE,
    WINDOW,
    Tossing3DPlateau,
)

NUM_SWEEPS = 101
NUM_TASKS = 10


def _write_run(*, run_dir: Path, solved_per_sweep: list[int]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluations = [[i * 4, solved, NUM_TASKS] for i, solved in enumerate(solved_per_sweep)]
    (run_dir / "stats.json").write_text(json.dumps({"evaluations": evaluations}))


def _flat(*, level: int, jitter: list[int] | None = None) -> list[int]:
    """A settled curve: climbs early, then sits at `level` with optional sweep-to-sweep
    noise. The noise is the point -- Tossing3D's real per-seed score moves by several
    tasks between adjacent sweeps with no learning event, and a rule that mistook that
    for a climb would be useless."""
    curve = [min(level, i) for i in range(15)] + [level] * (NUM_SWEEPS - 15)
    if jitter:
        for offset, delta in enumerate(jitter):
            curve[-(offset + 1)] = max(0, min(NUM_TASKS, curve[-(offset + 1)] + delta))
    return curve


def _climbing(*, start: int, end: int) -> list[int]:
    """Still improving across the whole run, so `LATE` genuinely exceeds `REFERENCE`."""
    step = (end - start) / (NUM_SWEEPS - 1)
    return [round(start + step * i) for i in range(NUM_SWEEPS)]


def test_window_scores_average_the_requested_sweeps(*, tmp_path: Path) -> None:
    """The whole rule rests on scoring a window rather than a sweep, so the window has
    to be the sweeps it claims."""
    curve = list(range(NUM_SWEEPS))
    _write_run(run_dir=tmp_path / "ees" / "0", solved_per_sweep=[min(c, NUM_TASKS) for c in curve])
    curves = Tossing3DPlateau.load_curves(results_root=tmp_path)
    scores = Tossing3DPlateau.window_scores(curves=curves, start=0, width=5)
    # Sweeps 0..4 of the ramp are 0,1,2,3,4.
    assert scores[0] == 2.0


def test_pooled_keeps_the_denominator(*, tmp_path: Path) -> None:
    """`x/y` everywhere: the denominator is tasks x seeds, and losing it is exactly how
    a percentage sneaks in."""
    for seed in range(3):
        _write_run(run_dir=tmp_path / "ees" / str(seed), solved_per_sweep=_flat(level=8))
    curves = Tossing3DPlateau.load_curves(results_root=tmp_path)
    scores = Tossing3DPlateau.window_scores(curves=curves, start=NUM_SWEEPS - WINDOW, width=WINDOW)
    x, y = Tossing3DPlateau.pooled(scores=scores, num_total=NUM_TASKS)
    assert (x, y) == (24.0, 30)


def test_a_flat_curve_is_not_called_a_climb(*, tmp_path: Path) -> None:
    """Ten seeds that plateaued at 8/10 early and then only jitter. The rule must return
    a null result on climbing -- if this fails, a plateau gets reported as continued
    learning."""
    jitters = [
        [1, -1, 0, 1, -2, 0, 1, -1, 0, 1],
        [-1, 1, 0, -1, 2, 0, -1, 1, 0, -1],
    ]
    for seed in range(10):
        _write_run(
            run_dir=tmp_path / "ees" / str(seed),
            solved_per_sweep=_flat(level=8, jitter=jitters[seed % 2]),
        )
    curves = Tossing3DPlateau.load_curves(results_root=tmp_path)
    late = Tossing3DPlateau.window_scores(curves=curves, start=NUM_SWEEPS - WINDOW, width=WINDOW)
    reference = Tossing3DPlateau.window_scores(
        curves=curves, start=REFERENCE_START_CYCLE, width=WINDOW
    )
    differences = [late[s] - reference[s] for s in sorted(curves)]
    from analysis.practice_makes_perfect.paired_tests import PairedTests

    assert PairedTests.sign_flip(differences=differences).p_value > 0.05


def test_a_climbing_curve_is_called_a_climb(*, tmp_path: Path) -> None:
    """The opposite guard, and the one that stops the rule being vacuously
    conservative: ten seeds still improving must come out significant."""
    for seed in range(10):
        _write_run(
            run_dir=tmp_path / "ees" / str(seed), solved_per_sweep=_climbing(start=2, end=10)
        )
    curves = Tossing3DPlateau.load_curves(results_root=tmp_path)
    late = Tossing3DPlateau.window_scores(curves=curves, start=NUM_SWEEPS - WINDOW, width=WINDOW)
    reference = Tossing3DPlateau.window_scores(
        curves=curves, start=REFERENCE_START_CYCLE, width=WINDOW
    )
    differences = [late[s] - reference[s] for s in sorted(curves)]
    from analysis.practice_makes_perfect.paired_tests import PairedTests

    assert all(d > 0 for d in differences)
    assert PairedTests.sign_flip(differences=differences).p_value < 0.05


def test_an_unfinished_run_is_omitted_rather_than_truncating_a_window(*, tmp_path: Path) -> None:
    """A run still in flight has no `stats.json`. Including a short curve would make
    `num_sweeps` the minimum across seeds and silently move both windows for every
    seed -- so the whole comparison would shift because one run had not finished."""
    _write_run(run_dir=tmp_path / "ees" / "0", solved_per_sweep=_flat(level=9))
    inflight = tmp_path / "ees" / "1"
    inflight.mkdir(parents=True)
    (inflight / "progress.jsonl").write_text('{"sweeps_completed": 12}\n')

    curves = Tossing3DPlateau.load_curves(results_root=tmp_path)
    assert set(curves) == {0}
