"""`ResultsWriter`: the interface an *observer* of a run's results implements, and the
one hook set `method_runner.py` drives for every writer in `registry.RESULTS_WRITERS`.

## What this abstracts, and what it deliberately does not

Before this, every extra thing a run emitted was wired by hand: `method_runner.py`
named `RunProgressWriter`, `CompetenceLogRecorder` and `EpisodeTraceRecorder`
individually, called each one's `open_if_requested` individually, and fired each one's
hook individually inside `on_sweep_end`. Adding a sixth output meant editing the
harness. `ResultsWriter` turns that into a list: a new observer is a class plus one
entry in `registry.RESULTS_WRITERS`, and the harness does not change.

It abstracts **observers of a run's results at the two boundaries the harness owns** --
an evaluation sweep finishing, and the run ending. It does *not* abstract:

- **`stats.json` and `config_snapshot.json`.** These are the run's *product*, not an
  observation of it, and `stats.json`'s byte-stability is what proves a change did not
  alter results. Nothing that can be added to a list should be able to move that write.
- **`timing.json`.** It is written by `scripts/run_sweep.py`, in the *parent* process,
  after the child exits -- so no in-run hook can produce it, and a writer here could
  not see the wall-clock it measures.
- **`sampler_draws.jsonl`.** Its event is a sampler consultation deep inside a
  `Method`; the harness never sees one, so there is no hook here that could fire.
- **`episode_traces.jsonl`.** Its event is one evaluation *episode*, inside
  `PracticeLoop._evaluate`, one level below the sweep boundary these hooks sit on.

That list is the honest boundary of the mechanism rather than a to-do: three of the
four are structurally out of reach of a harness-level hook, and the fourth must not
move. See this folder's README.

## A real pydantic instance, not a static-method container

`core/README.md`'s dividing line is whether a class carries genuine per-run state. A
`ResultsWriter` does: `WandbResultsWriter` holds a live W&B run handle, and any
file-backed writer would hold an open handle -- exactly what makes
`CompetenceLogRecorder`, `EpisodeTraceRecorder` and `RunProgressWriter` real instances
rather than the static containers `HumanOracle` and `Renderer` are. So `ResultsWriter`
is `BaseModel, abc.ABC` with ordinary `self` methods.

## Why it is not a `core/` interface

`core/` holds the fixed abstract interfaces of the *problem* -- and it never takes a
stateful recorder: `episode_traces.py` states the rule outright ("a recorder carries
real per-run state, and `core/README.md`'s existing precedent -- `recording.LoopRecorder`,
kept out of `core` for the identical reason -- is that a stateful recorder never crosses
into `core`, only plain data does"). `ResultsWriter` is a stateful recorder, so it sits
in `practice_loop.py`'s layer, below `method_runner.py` and above `core`, and the
import-linter contract in `pyproject.toml` pins it there.

## Pure observer

Nothing here draws randomness, and no hook returns a value any caller branches on --
the same contract every existing recorder holds. A run with a writer on takes exactly
the actions it would have taken with it off and writes a byte-identical `stats.json`,
asserted end-to-end through the real CLI in `tests/results_writer/test_wandb_writer.py`
rather than argued from inspection.
"""

import abc
import argparse

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics


class ResultsWriter(BaseModel, abc.ABC):
    """An observer the harness offers every run, which decides for itself whether this
    run asked for it.

    Exactly one method is abstract -- `open_if_requested`, the one thing that cannot
    have a meaningful default, since interpreting the flag is the whole of what makes
    one writer different from another. Both recording hooks are concrete no-ops, for
    the same reason `Method.get_practice_policy` is: a writer that cares about only one
    boundary should implement only that one and carry no boilerplate for the other."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    @abc.abstractmethod
    def open_if_requested(*, args: argparse.Namespace) -> "ResultsWriter | None":
        """This run's writer, or None -- the one place this writer's flag is
        interpreted, so no caller has to re-derive it.

        Called on every registered writer at the start of every run, which is what
        makes the registry safe to grow: a writer whose flag was not passed returns
        None and is never driven again. A writer whose flag *was* passed but cannot
        work (no `--output-dir`, an optional dependency missing) must raise **here**,
        before the run starts -- a multi-hour run that produced no instrumentation
        because of a missing second flag is the expensive way to find out, and
        `config_snapshot.py`'s never-raises policy is right for provenance and wrong
        for a thing whose only job is to record."""

    def record_checkpoint(self, *, metrics: Metrics) -> None:
        """One evaluation sweep just finished and was recorded.

        Fired after `PracticeLoop._evaluate` has appended, so `metrics.evaluations[-1]`
        is that sweep -- the same boundary `progress.jsonl` and `competence_log.jsonl`
        already key on, and the same learning-curve x-axis. Deliberately per *sweep*,
        never per environment step: a step-level hook would be thousands of calls per
        run for no analytic gain."""

    def close(self, *, metrics: Metrics) -> None:
        """The run is over; write any summary and release anything held.

        Fired from a `finally`, matching `recording.LoopRecorder.close`'s own placement:
        a crashed run is exactly when someone wants whatever was recorded up to the
        crash, and a W&B run left unfinished would never be flushed. `metrics` is
        therefore whatever the run got to -- possibly no evaluations at all -- rather
        than a guarantee of a completed run."""
