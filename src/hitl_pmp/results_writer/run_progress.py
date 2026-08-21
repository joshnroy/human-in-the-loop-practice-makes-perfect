"""Per-sweep progress, written while a run is still going.

`stats.json` is written once, after the last cycle -- so on a multi-hour Tossing3D run a
**hung run and a slow run produced identical evidence**, and no ETA was computable.
`progress.jsonl` is appended and flushed after every evaluation sweep, answering both
from the file alone: `sweeps_completed` / `sweeps_total` gives the ETA, and the last
line's wall-clock `timestamp` distinguishes a hang (stopped advancing) from a slow cycle
(merely old).

**Always on**, with no flag: `--output-dir` is the whole condition. Instrumentation you
have to remember to switch on is not available for the run you did not expect to need it
for, and this costs a few hundred bytes and one `write` per sweep.

**`open_if_requested` takes `num_cycles` rather than reading it off `args`**, because
`num_cycles` is a method-CLI decision, not a flag: `SkillOracleCli` passes the literal
`0` and its namespace has no such attribute, while `PracticeCycleCli` declares one. A
`getattr(args, "num_cycles", 0)` fallback would be right for both of today's methods by
coincidence and silently wrong for the first method that computes its own cycle count.

**A sibling of `stats.json`**, for the same reason as `timing.json`,
`config_snapshot.json`, `sampler_draws.jsonl` and `competence_log.jsonl`: it carries
timestamps, and `stats.json`'s byte-stability is how this repo verifies a change did not
alter results (`tests/scripts/test_reproducibility.py`).

**One line per sweep, not per cycle.** `num_cycles = N` produces `N + 1` sweeps, so a
`num_cycles=0` run (every non-learning baseline) still reports its single sweep instead
of writing nothing, and `sweeps_completed / sweeps_total` is a true fraction of the work."""

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.results_writer.results_writer import ResultsWriter

# The sibling `--output-dir` file this writes, named after its content the same way
# `stats.json`, `timing.json` and `config_snapshot.json` are.
RUN_PROGRESS_FILENAME = "progress.jsonl"


class RunProgressWriter(ResultsWriter):
    """Appends one `RunProgress` line per completed evaluation sweep.

    A real pydantic instance rather than a static-method container, because it carries
    genuine per-run state -- the open handle and the monotonic clock origin -- which is
    this project's stated dividing line between the two.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path
    sweeps_total: int

    _handle: TextIO | None = PrivateAttr(default=None)
    _has_written: bool = PrivateAttr(default=False)
    _started_monotonic: float = PrivateAttr(default_factory=time.monotonic)

    @staticmethod
    def open_if_requested(
        *, args: argparse.Namespace, num_cycles: int
    ) -> "RunProgressWriter | None":
        """The writer for this run, or None when there is no `--output-dir` to write
        into -- the always-on condition, with no flag of its own.

        Constructed at the *start* of the run, so `elapsed_seconds` measures the run
        rather than the time since the first sweep happened to finish. That is a
        property of where `method_runner.py` opens the registry, which is before
        `PracticeLoop.run` for exactly this reason."""
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            return None
        return RunProgressWriter(
            output_path=Path(output_dir) / RUN_PROGRESS_FILENAME,
            # N cycles is N + 1 sweeps; see the module docstring.
            sweeps_total=num_cycles + 1,
        )

    def record_checkpoint(self, *, metrics: Metrics, video_path: Path | None = None) -> None:
        """Append the state of the run as of the sweep that just finished.

        Reads `metrics.evaluations[-1]`, so this must be called after the sweep has
        been recorded -- which is exactly where `PracticeLoop` fires its hook. Doing
        nothing when there are no evaluations yet keeps the hook safe to call
        unconditionally.

        `video_path` is ignored: this writer's product is `progress.jsonl`, a small
        text file meant to be tailed, and a checkpoint clip's path is not progress.
        """
        del video_path
        if not metrics.evaluations:
            return
        transitions, num_solved, num_total = metrics.evaluations[-1]
        progress = RunProgress(
            sweeps_completed=len(metrics.evaluations),
            sweeps_total=self.sweeps_total,
            transitions=transitions,
            num_solved=num_solved,
            num_total=num_total,
            # Monotonic, so a clock adjustment mid-run cannot make elapsed time go
            # backwards -- the same choice run_sweep.py's timing.json makes.
            elapsed_seconds=time.monotonic() - self._started_monotonic,
            timestamp=datetime.now().astimezone().isoformat(),
        )
        handle = self._open()
        handle.write(progress.model_dump_json() + "\n")
        # Per line. The whole point is to be readable *during* the run, and an
        # unflushed buffer is indistinguishable from a hang -- which is precisely the
        # confusion this file exists to end.
        handle.flush()

    def close(self, *, metrics: Metrics) -> None:
        """Release the handle. Nothing is written here: every line was already flushed
        as it happened, which is the whole point of the file, so there is no tail to
        emit and a run that crashed mid-sweep loses nothing it had recorded.

        `metrics` is unused for that reason, and kept only because it is the hook's
        signature. Fired from `method_runner.py`'s `finally`, so it also runs on a
        crash; a writer that never opened a handle closes nothing."""
        del metrics
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _open(self) -> TextIO:
        handle = self._handle
        if handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncating on the *first* open only, so a re-used output directory starts
            # clean; appending afterwards, so a `close` followed by a further checkpoint
            # cannot silently discard the lines already written. Nothing does that
            # today -- `close` fires from a `finally` after the last sweep -- and that
            # is exactly why the failure would be invisible if it ever started to.
            handle = (
                self.output_path.open("a", encoding="utf-8")
                if self._has_written
                else self.output_path.open("w", encoding="utf-8")
            )
            self._handle = handle
            self._has_written = True
        return handle


class RunProgress(BaseModel):
    """One evaluation sweep's worth of progress.

    Frozen: a record of something that already happened. `num_solved`/`num_total` are
    kept as a pair rather than a rate, so every reader reports `x/y` and no denominator
    can go missing between here and a status line.
    """

    model_config = ConfigDict(frozen=True)

    sweeps_completed: int
    sweeps_total: int
    transitions: int
    num_solved: int
    num_total: int
    elapsed_seconds: float
    timestamp: str
