"""Covers the per-cycle progress file, which exists so a long run can be watched
while it runs rather than only explained after it finishes.

The failure it removes: `stats.json` is written once, at the end. A 100-cycle
Tossing3D run is hours, so until now a hung run and a merely slow one produced the
same evidence -- an output directory with nothing in it -- and the only way to tell
them apart was to wait. `progress.jsonl` is appended and flushed per sweep, so both
"how far along is it" and "is it still moving" are answerable from the file itself.

Like `sampler_draws.jsonl`, this is a **sibling** of `stats.json` and never a field
in it: `stats.json`'s byte-stability is what verifies a change did not alter results,
and it carries timestamps, so putting them inside would break that on every run.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Light Switch: pure numpy, milliseconds per cycle, and it exercises the same
# MethodRunner cadence every domain shares. Nothing here is domain-specific.
LIGHTSWITCH_ARGS = ("--env", "lightswitch", "--grid-size", "2", "--num-test-tasks", "3")


def _run(*, output_dir: Path, num_cycles: int, extra: tuple[str, ...] = ()) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "hitl_pmp.cli",
            *LIGHTSWITCH_ARGS,
            "--method",
            "ees",
            "--num-cycles",
            str(num_cycles),
            "--max-steps-per-interaction",
            "4",
            "--seed",
            "3",
            "--output-dir",
            str(output_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
    )
    return output_dir


def _progress(*, output_dir: Path) -> list[dict[str, object]]:
    text = (output_dir / "progress.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_one_line_per_evaluation_sweep(*, tmp_path: Path) -> None:
    """N cycles means N+1 sweeps -- one before any practice, then one per cycle --
    so a reader can compute a fraction complete without being told the total
    separately."""
    out = _run(output_dir=tmp_path / "run", num_cycles=3)
    lines = _progress(output_dir=out)
    assert [line["sweeps_completed"] for line in lines] == [1, 2, 3, 4]
    assert all(line["sweeps_total"] == 4 for line in lines)


def test_every_line_carries_what_an_eta_needs(*, tmp_path: Path) -> None:
    """Elapsed wall-clock and a fraction complete are the whole requirement: an ETA
    is elapsed / completed * remaining, computable from any single line."""
    out = _run(output_dir=tmp_path / "run", num_cycles=2)
    for line in _progress(output_dir=out):
        assert set(line) == {
            "sweeps_completed",
            "sweeps_total",
            "transitions",
            "num_solved",
            "num_total",
            "elapsed_seconds",
            "timestamp",
        }
        assert isinstance(line["elapsed_seconds"], float)
        assert line["elapsed_seconds"] >= 0.0


def test_elapsed_seconds_is_monotonic(*, tmp_path: Path) -> None:
    """A stalled last line is how a watcher tells a hung run from a slow one, so the
    clock has to actually advance across lines."""
    out = _run(output_dir=tmp_path / "run", num_cycles=3)
    elapsed = [float(line["elapsed_seconds"]) for line in _progress(output_dir=out)]
    assert elapsed == sorted(elapsed)


def test_the_last_line_agrees_with_stats_json(*, tmp_path: Path) -> None:
    """The progress file and stats.json are two views of the same run. If the final
    line disagreed with the file written beside it, neither could be trusted."""
    out = _run(output_dir=tmp_path / "run", num_cycles=3)
    last = _progress(output_dir=out)[-1]
    stats = json.loads((out / "stats.json").read_text())
    transitions, num_solved, num_total = stats["evaluations"][-1]
    assert (last["transitions"], last["num_solved"], last["num_total"]) == (
        transitions,
        num_solved,
        num_total,
    )
    assert last["sweeps_completed"] == len(stats["evaluations"])


def test_progress_does_not_change_the_run(*, tmp_path: Path) -> None:
    """Always on, so unlike --record-sampler-draws there is no off switch to compare
    against -- instead this pins that the file is a pure addition: the same seed still
    produces a byte-identical stats.json, which is the property PR #146 and
    tests/scripts/test_reproducibility.py both rest on."""
    first = _run(output_dir=tmp_path / "a", num_cycles=3)
    second = _run(output_dir=tmp_path / "b", num_cycles=3)
    assert (first / "stats.json").read_bytes() == (second / "stats.json").read_bytes()


def test_a_zero_cycle_run_still_reports_its_single_sweep(*, tmp_path: Path) -> None:
    """Every non-learning baseline is num_cycles=0. One sweep happens, so one line
    is written -- a watcher polling the file must not have to special-case them."""
    out = _run(output_dir=tmp_path / "run", num_cycles=0)
    assert [line["sweeps_completed"] for line in _progress(output_dir=out)] == [1]
