"""`ResultsWriter`: the interface an *observer* of a run's results implements, and the one
hook set `method_runner.py` drives for every writer in `registry.RESULTS_WRITERS`.

It abstracts observers **at the two boundaries the harness owns** -- an evaluation sweep
finishing, and the run ending. A new observer is a class plus one entry in the registry;
the harness does not change. What it deliberately does *not* abstract, and why:

- **`stats.json` and `config_snapshot.json`** are the run's *product*, not an observation
  of it, and `stats.json`'s byte-stability is what proves a change did not alter results.
  Nothing addable to a list should be able to move that write.
- **`timing.json`** is written by `scripts/run_sweep.py` in the *parent* process after the
  child exits, so no in-run hook could produce it.
- **`sampler_draws.jsonl`**'s event is a sampler consultation deep inside a `Method`,
  which the harness never sees.
- **`episode_traces.jsonl`**'s event is one evaluation *episode*, one level below the
  sweep boundary these hooks sit on.

**A real pydantic instance, not a static-method container** (CLAUDE.md's dividing line):
`WandbResultsWriter` holds a live W&B run handle, and any file-backed writer holds an open
handle. **Not a `core/` interface**, for the same reason: `core/` never takes a stateful
recorder, only plain data crosses that boundary. So this sits in `practice_loop.py`'s
layer, and the import-linter contract in `pyproject.toml` pins it there.

**Pure observer.** Nothing here draws randomness and no hook returns a value any caller
branches on. A run with a writer on writes a byte-identical `stats.json` to one with it
off, asserted end-to-end through the real CLI in
`tests/results_writer/test_wandb_writer.py`."""

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
    def open_if_requested(*, args: argparse.Namespace, num_cycles: int) -> "ResultsWriter | None":
        """This run's writer, or None -- the one place this writer's own condition is
        interpreted, so no caller has to re-derive it.

        Called on every registered writer at the start of every run, which is what
        makes the registry safe to grow: a writer that declines returns None and is
        never driven again. A writer that *was* asked for but cannot work (no
        `--output-dir`, an optional dependency missing) must raise **here**, before the
        run starts -- a multi-hour run that produced no instrumentation because of a
        missing second flag is the expensive way to find out, and
        `config_snapshot.py`'s never-raises policy is right for provenance and wrong
        for a thing whose only job is to record.

        **The condition need not be a flag.** `WandbResultsWriter` keys on
        `--record-wandb`; `RunProgressWriter` is always on and keys on `--output-dir`
        alone, because instrumentation you have to remember to switch on is not
        available for the run you did not expect to need it for. "Always, when I can
        write at all" is a legitimate answer to "does this run want you", so an
        always-on writer needs no flag invented for it to fit this method's name.

        `num_cycles` is passed rather than read off `args` because it is **not a
        flag**: it is a method-CLI decision handed to `MethodRunner.run`, and
        `SkillOracleCli` passes a literal `0` with no `num_cycles` in its namespace at
        all. A writer that re-derived it from `args` would be right for today's two
        methods by coincidence. It is the one piece of run *shape*, as opposed to run
        *configuration*, a writer cannot otherwise see; a writer that does not need it
        ignores it."""

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
