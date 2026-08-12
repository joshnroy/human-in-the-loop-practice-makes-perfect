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

**Why the runs are module-scoped fixtures.** `--seed` fully determines a run, so the
four tests that each wanted a 3-cycle run were paying for four byte-identical ones.
They share one now; every assertion is unchanged. The contract that makes it safe is
that these directories are **read-only** to the tests below. The determinism test is
the deliberate exception: it needs two genuinely independent invocations, so it takes
the shared run as one arm and runs a second itself.

**Since this writer became a `ResultsWriter`**, the file's own bytes are pinned here
too (`PINNED_LINES`), captured from a run that predates the change. That is what makes
the adoption a refactor rather than a rewrite: the writer moved into
`results_writer/`, gained the ABC and a `close`, and every byte it emits stayed put.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.results_writer.registry import RESULTS_WRITERS
from hitl_pmp.results_writer.results_writer import ResultsWriter
from hitl_pmp.results_writer.run_progress import RunProgressWriter

# Light Switch: pure numpy, milliseconds per cycle, and it exercises the same
# MethodRunner cadence every domain shares. Nothing here is domain-specific.
LIGHTSWITCH_ARGS = ("--env", "lightswitch", "--grid-size", "2", "--num-test-tasks", "3")

# The exact lines a `--method ees --num-cycles 3 --seed 3` run wrote *before*
# `RunProgressWriter` became a `ResultsWriter`, with only the two wall-clock fields
# masked -- see `_masked_lines`. Every other byte is determined by the seed, so this
# pins field names, field order, JSON separators, the line count and every value the
# refactor could have moved.
PINNED_LINES = (
    '{"sweeps_completed":1,"sweeps_total":4,"transitions":0,"num_solved":1,'
    '"num_total":3,"elapsed_seconds":<elapsed>,"timestamp":"<timestamp>"}',
    '{"sweeps_completed":2,"sweeps_total":4,"transitions":4,"num_solved":1,'
    '"num_total":3,"elapsed_seconds":<elapsed>,"timestamp":"<timestamp>"}',
    '{"sweeps_completed":3,"sweeps_total":4,"transitions":8,"num_solved":0,'
    '"num_total":3,"elapsed_seconds":<elapsed>,"timestamp":"<timestamp>"}',
    '{"sweeps_completed":4,"sweeps_total":4,"transitions":12,"num_solved":1,'
    '"num_total":3,"elapsed_seconds":<elapsed>,"timestamp":"<timestamp>"}',
)

# `elapsed_seconds` and `timestamp` are wall-clock by design -- they are the whole
# reason this file exists -- so they are the two fields no two runs can share. Masking
# exactly them, and nothing else, is what keeps the pin above a genuine byte
# comparison rather than a reparse-and-compare-some-fields.
_WALL_CLOCK = re.compile(r'"elapsed_seconds":[^,]+,"timestamp":"[^"]+"')


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


def _masked_lines(*, output_dir: Path) -> tuple[str, ...]:
    """The file's literal text with only `elapsed_seconds` and `timestamp` masked."""
    text = (output_dir / "progress.jsonl").read_text(encoding="utf-8")
    masked = _WALL_CLOCK.sub('"elapsed_seconds":<elapsed>,"timestamp":"<timestamp>"', text)
    assert masked.endswith("\n"), "every line is newline-terminated, including the last"
    return tuple(masked.split("\n")[:-1])


@pytest.fixture(scope="module")
def three_cycle_run(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The ordinary case: enough cycles that a sequence of sweeps exists to assert on."""
    return _run(output_dir=tmp_path_factory.mktemp("three_cycle"), num_cycles=3)


@pytest.fixture(scope="module")
def zero_cycle_run(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The non-learning-baseline shape, which has no practice cycle at all."""
    return _run(output_dir=tmp_path_factory.mktemp("zero_cycle"), num_cycles=0)


@pytest.fixture(scope="module")
def skill_oracle_run(*, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The `--method skill-oracle` path specifically, because it is the one whose
    `args` namespace has **no `num_cycles` attribute at all** -- the method-CLI passes
    `num_cycles=0` to `MethodRunner.run` as a literal. See
    `test_sweeps_total_comes_from_the_run_not_from_args`."""
    output_dir = tmp_path_factory.mktemp("skill_oracle")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "hitl_pmp.cli",
            *LIGHTSWITCH_ARGS,
            "--method",
            "skill-oracle",
            "--seed",
            "3",
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
    )
    return output_dir


def test_one_line_per_evaluation_sweep(*, three_cycle_run: Path) -> None:
    """N cycles means N+1 sweeps -- one before any practice, then one per cycle --
    so a reader can compute a fraction complete without being told the total
    separately."""
    lines = _progress(output_dir=three_cycle_run)
    assert [line["sweeps_completed"] for line in lines] == [1, 2, 3, 4]
    assert all(line["sweeps_total"] == 4 for line in lines)


def test_every_line_carries_what_an_eta_needs(*, three_cycle_run: Path) -> None:
    """Elapsed wall-clock and a fraction complete are the whole requirement: an ETA
    is elapsed / completed * remaining, computable from any single line.

    Asserted over the shared 3-cycle run rather than a 2-cycle one of its own: the
    claim is about every line of any run, so more lines is strictly more evidence."""
    for line in _progress(output_dir=three_cycle_run):
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


def test_elapsed_seconds_is_monotonic(*, three_cycle_run: Path) -> None:
    """A stalled last line is how a watcher tells a hung run from a slow one, so the
    clock has to actually advance across lines."""
    elapsed = [float(line["elapsed_seconds"]) for line in _progress(output_dir=three_cycle_run)]
    assert elapsed == sorted(elapsed)


def test_the_last_line_agrees_with_stats_json(*, three_cycle_run: Path) -> None:
    """The progress file and stats.json are two views of the same run. If the final
    line disagreed with the file written beside it, neither could be trusted."""
    last = _progress(output_dir=three_cycle_run)[-1]
    stats = json.loads((three_cycle_run / "stats.json").read_text())
    transitions, num_solved, num_total = stats["evaluations"][-1]
    assert (last["transitions"], last["num_solved"], last["num_total"]) == (
        transitions,
        num_solved,
        num_total,
    )
    assert last["sweeps_completed"] == len(stats["evaluations"])


def test_progress_does_not_change_the_run(*, three_cycle_run: Path, tmp_path: Path) -> None:
    """Always on, so unlike --record-sampler-draws there is no off switch to compare
    against -- instead this pins that the file is a pure addition: the same seed still
    produces a byte-identical stats.json, which is the property PR #146 and
    tests/scripts/test_reproducibility.py both rest on.

    Two genuinely independent subprocess invocations, which is what the claim needs:
    the shared fixture run is one arm, and the second is run here."""
    second = _run(output_dir=tmp_path / "b", num_cycles=3)
    assert (three_cycle_run / "stats.json").read_bytes() == (second / "stats.json").read_bytes()


def test_a_zero_cycle_run_still_reports_its_single_sweep(*, zero_cycle_run: Path) -> None:
    """Every non-learning baseline is num_cycles=0. One sweep happens, so one line
    is written -- a watcher polling the file must not have to special-case them."""
    assert [line["sweeps_completed"] for line in _progress(output_dir=zero_cycle_run)] == [1]


def test_the_bytes_are_unchanged_by_becoming_a_results_writer(*, three_cycle_run: Path) -> None:
    """The load-bearing test of that adoption: `progress.jsonl` is consumed by
    `analysis/` and by whoever is watching a live sweep, so a refactor that moved a
    field, reordered two, or dropped the trailing newline would break readers while
    every behavioural assertion above still passed.

    `PINNED_LINES` was captured from a run made before the change, at the same seed,
    so this is a genuine before/after comparison rather than a self-consistency
    check."""
    assert _masked_lines(output_dir=three_cycle_run) == PINNED_LINES


def test_it_is_a_registered_results_writer() -> None:
    """`progress.jsonl` fires at exactly `record_checkpoint`'s boundary, so it is
    driven by the same list every other observer is, rather than by its own hand-wired
    call in `method_runner.py`."""
    assert issubclass(RunProgressWriter, ResultsWriter)
    assert RunProgressWriter in RESULTS_WRITERS


def test_it_is_always_on_whenever_there_is_somewhere_to_write(*, tmp_path: Path) -> None:
    """No flag, and deliberately none: instrumentation you have to remember to switch
    on is not available for the run you did not expect to need it for. `--output-dir`
    is the whole condition, exactly as it was before this became a `ResultsWriter` --
    so `open_if_requested` admits an always-on writer rather than an opt-in flag being
    invented to fit its name."""
    assert (
        RunProgressWriter.open_if_requested(
            args=argparse.Namespace(output_dir=tmp_path), num_cycles=3
        )
        is not None
    )
    assert (
        RunProgressWriter.open_if_requested(args=argparse.Namespace(output_dir=None), num_cycles=3)
        is None
    )


def test_sweeps_total_comes_from_the_run_not_from_args(*, tmp_path: Path) -> None:
    """`num_cycles` is a **method-CLI decision, not a flag**: `SkillOracleCli` passes
    the literal `0` to `MethodRunner.run` and its `args` namespace carries no
    `num_cycles` at all. So a writer that re-derived the denominator from `args` would
    silently write a wrong `sweeps_total` for any method that decides its own cycle
    count -- which is why `open_if_requested` takes it as an argument.

    The namespace here deliberately has no `num_cycles` attribute, reproducing exactly
    that shape."""
    writer = RunProgressWriter.open_if_requested(
        args=argparse.Namespace(output_dir=tmp_path), num_cycles=3
    )
    assert writer is not None
    assert writer.sweeps_total == 4


def test_the_skill_oracle_path_still_reports_its_single_sweep(*, skill_oracle_run: Path) -> None:
    """End to end through the real CLI on the one method whose `args` has no
    `num_cycles`, so the claim above is checked against the actual wiring and not only
    against a hand-built namespace."""
    lines = _progress(output_dir=skill_oracle_run)
    assert [line["sweeps_completed"] for line in lines] == [1]
    assert [line["sweeps_total"] for line in lines] == [1]


def test_close_releases_the_handle(*, tmp_path: Path) -> None:
    """The second half of adopting the interface. Before, the handle stayed open until
    the interpreter collected the writer; `close` fires from `method_runner.py`'s
    `finally`, so the file is released even when the run crashes."""
    writer = RunProgressWriter.open_if_requested(
        args=argparse.Namespace(output_dir=tmp_path), num_cycles=0
    )
    assert writer is not None
    metrics = Metrics()
    metrics.record_evaluation(num_online_transitions=0, num_solved=1, num_total=3)
    writer.record_checkpoint(metrics=metrics)
    writer.close(metrics=metrics)
    assert len(_progress(output_dir=tmp_path)) == 1
    # Idempotent: `close` runs on every path out of a run, including one that never
    # opened a handle at all.
    writer.close(metrics=metrics)
