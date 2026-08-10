"""Per-sweep progress, written while a run is still going.

## The failure this removes

`stats.json` is written once, after the last cycle. On Light Switch that is seconds
and nobody notices; on Tossing3D a 100-cycle run is hours, and until now the output
directory stayed empty for all of them. So a **hung run and a slow run produced
identical evidence**, and the only way to tell them apart was to wait for one of them
to finish. There was no way to compute an ETA either, which is what turns "it is
still going" into a decision about whether to keep waiting.

`progress.jsonl` is appended and flushed after every evaluation sweep. Two questions
become answerable from the file alone:

- **How far along is it** -- `sweeps_completed` out of `sweeps_total`, so an ETA is
  `elapsed_seconds / sweeps_completed * (sweeps_total - sweeps_completed)` off any
  single line.
- **Is it still moving** -- the last line's `timestamp`. A wall-clock stamp that
  stopped advancing is a hang; one that is merely old is a slow cycle. A file that
  only records counts cannot distinguish those.

## Always on, unlike --record-sampler-draws

There is no flag. Instrumentation you have to remember to switch on is not available
for the run you did not expect to need it for, and this one costs a few hundred bytes
and one `write` per sweep. `--record-sampler-draws` is opt-in because it is per-draw
and domain-specific; this is per-sweep and universal.

## A sibling of stats.json, for the same reason as the others

It carries timestamps and elapsed wall-clock, and `stats.json`'s **byte-stability is
load-bearing** -- it is how this repo verifies a change did not alter results (PR #146
used exactly that property; `tests/scripts/test_reproducibility.py` rests on it for
three domains). A timestamp inside `stats.json` would break that on every single run.
`timing.json`, `config_snapshot.json` and `sampler_draws.jsonl` are all separate for
this reason; this is the fourth.

Nothing here is ever an input to a reproducibility comparison, and a run's actions do
not depend on it: the same seed still writes a byte-identical `stats.json` with this
file being written beside it.

## One line per sweep, not per cycle

`num_cycles = N` produces `N + 1` evaluation sweeps -- one before any practice, then
one after each cycle -- so counting sweeps rather than cycles means a `num_cycles=0`
run (every non-learning baseline) still reports its single sweep instead of writing
nothing at all. It also means `sweeps_completed / sweeps_total` is a true fraction of
the work, which a cycle count off by one would not be.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, PrivateAttr

from hitl_pmp.core.metrics.metrics import Metrics

# The sibling `--output-dir` file this writes, named after its content the same way
# `stats.json`, `timing.json` and `config_snapshot.json` are.
RUN_PROGRESS_FILENAME = "progress.jsonl"


class RunProgressWriter(BaseModel):
    """Appends one `RunProgress` line per completed evaluation sweep.

    A real pydantic instance rather than a static-method container, because it carries
    genuine per-run state -- the open handle and the monotonic clock origin -- which is
    this project's stated dividing line between the two.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_path: Path
    sweeps_total: int

    _handle: TextIO | None = PrivateAttr(default=None)
    _started_monotonic: float = PrivateAttr(default_factory=time.monotonic)

    @staticmethod
    def for_run(*, output_dir: Path | None, num_cycles: int) -> "RunProgressWriter | None":
        """The writer for this run, or None when there is no `--output-dir` to write
        into. Constructed at the *start* of the run, so `elapsed_seconds` measures the
        run rather than the time since the first sweep happened to finish."""
        if output_dir is None:
            return None
        return RunProgressWriter(
            output_path=Path(output_dir) / RUN_PROGRESS_FILENAME,
            # N cycles is N + 1 sweeps; see the module docstring.
            sweeps_total=num_cycles + 1,
        )

    def record(self, *, metrics: Metrics) -> None:
        """Append the state of the run as of the sweep that just finished.

        Reads `metrics.evaluations[-1]`, so this must be called after the sweep has
        been recorded -- which is exactly where `PracticeLoop` fires its hook. Doing
        nothing when there are no evaluations yet keeps the hook safe to call
        unconditionally.
        """
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

    def _open(self) -> TextIO:
        if self._handle is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.output_path.open("w", encoding="utf-8")
        return self._handle


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
