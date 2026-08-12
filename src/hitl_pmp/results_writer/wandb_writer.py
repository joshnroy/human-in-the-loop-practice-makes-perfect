"""`--record-wandb`: mirror a run's evaluation sweeps into Weights & Biases.

## What W&B adds that this repo does not already have

Very little of the usual experiment-tracking pitch applies here: `stats.json` already
holds the results, `config_snapshot.json` already holds richer provenance than W&B
collects, `timing.json` already holds wall-clock and concurrency, and
`progress.jsonl` already answers "is this run alive". Three things are genuinely
missing, and they are the whole reason for this file:

1. **Cross-run comparison without writing a script.** Every comparison today is a
   bespoke `analysis/` script against one `--results-root`.
2. **Watching a long sweep from somewhere other than this machine.**
3. **A durable index of runs.** `results/` is gitignored, local, and on one box.

W&B is an **index and a viewer, never the system of record.** `docs/experiment-logs/`
stays the durable, reviewed, committed record; a W&B run page is not citable six months
out the way a committed log entry is, and `scripts/check_doc_links.sh` already bans a
URL in a committed doc.

## Offline by default

If `WANDB_MODE` is unset this passes `mode="offline"`, which writes to local disk and
syncs later (`wandb sync <output-dir>/wandb/offline-run-*`). That default is set here
rather than left to the environment for three independent reasons: no run ever blocks
on the network; a machine with no credential (CI, a fresh worktree) still works; and a
sweep of ~22 concurrent runs opens no sockets. An explicit `WANDB_MODE=online` wins, so
watching a single long run live is a launch-time choice, not a code change.

`scripts/run_sweep.py` already forwards `os.environ` to every child, so `WANDB_MODE`,
`WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_RUN_GROUP` propagate across a whole grid with
no code change and no extra flags -- which is why this file adds exactly one flag and
reads the rest from W&B's own environment variables rather than inventing parallel ones.

## `wandb sweep` is deliberately not adopted

`scripts/run_sweep.py` already owns fixed seeds, the `<results-root>/<method>/<seed>/`
layout `analysis/` globs for, spawn-retry, `timing.json` and cross-agent concurrency
budgeting. W&B's sweep agent **draws its own hyperparameter values**, which is directly
at odds with this project's "fixed seeds, never randomly drawn" discipline. W&B's
*grouping* gives what its sweeps would have given us, via `WANDB_RUN_GROUP`.

## Where the offline data lands

`dir=<output-dir>`, so a run's W&B directory sits beside the `stats.json` it describes,
inside the gitignored `results/` tree. W&B's own default is `./wandb/`, relative to the
**current working directory** -- which for a run launched from the repo would be the
repo root. `wandb/` is gitignored anyway, belt and braces, since a failed `init` can
still write there.

## Pure observer

Nothing here draws randomness and no method returns a value any caller branches on.
A run with `--record-wandb` writes a byte-identical `stats.json` to one without it,
asserted end-to-end through the real CLI in `tests/results_writer/test_wandb_writer.py`.
The one guarantee that is **weaker** and should not be overclaimed: any W&B call costs
wall-clock, so `timing.json` from a `--record-wandb` run is not strictly comparable to
one without. Offline mode makes that small but it does not make it zero, and it has not
been measured.
"""

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, PrivateAttr

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.results_writer.results_writer import ResultsWriter
from hitl_pmp.results_writer.types import CheckpointScalars, RunSummaryScalars

# One project per repo, not one per experiment: cross-run comparison is the entire
# point and it does not work across projects. Overridable by W&B's own WANDB_PROJECT.
DEFAULT_PROJECT = "hitl-pmp"

# W&B's own environment variables, read rather than re-declared as flags -- run_sweep.py
# already forwards os.environ to every child, so these configure a whole grid for free.
PROJECT_ENV_VAR = "WANDB_PROJECT"
MODE_ENV_VAR = "WANDB_MODE"


class WandbResultsWriter(ResultsWriter):
    """Logs one point per evaluation sweep, plus a run summary, to Weights & Biases.

    A real pydantic instance rather than a static-method container, because it carries
    genuine per-run state -- the live W&B run handle -- which is this project's stated
    dividing line between the two.

    The handle is opened **lazily**, on the first checkpoint rather than at
    construction, matching every existing recorder's lazy open: `wandb.init()` performs
    real work and spawns a background service process, and a run that never reaches a
    checkpoint should not pay for it or leave a stray empty W&B run behind. Whether the
    optional dependency is present is still checked eagerly, in `open_if_requested`, so
    a missing `wandb` fails before the run starts rather than at the first checkpoint."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_dir: Path
    # The *resolved* argparse namespace, stringified -- so defaulted flags land too,
    # the same reason `config_snapshot.py` records `vars(args)` rather than sys.argv.
    config: dict[str, str]
    run_name: str
    job_type: str
    tags: tuple[str, ...]

    _run: Any = PrivateAttr(default=None)

    @staticmethod
    def open_if_requested(*, args: argparse.Namespace) -> "WandbResultsWriter | None":
        """This run's W&B writer, or None. Raises up front on both ways the flag can be
        unusable, rather than discovering either mid-run."""
        if not getattr(args, "record_wandb", False):
            return None
        output_dir = getattr(args, "output_dir", None)
        if output_dir is None:
            raise ValueError(
                "--record-wandb needs --output-dir: W&B's offline data is written to "
                "<output-dir>/wandb/ so it sits beside the stats.json it describes, and "
                "without an output directory there is nowhere to put it."
            )
        if importlib.util.find_spec("wandb") is None:
            # Loudly, and here. A writer that swallowed this would produce a run that
            # looks logged and is not -- the single most expensive failure mode for
            # instrumentation, since it is only discovered when the data is wanted.
            raise ValueError(
                "--record-wandb needs the optional wandb dependency, which is not "
                'installed: pip install -e ".[wandb]"'
            )
        method = str(getattr(args, "method", "unknown"))
        environment = str(getattr(args, "env", "unknown"))
        seed = getattr(args, "seed", 0)
        return WandbResultsWriter(
            output_dir=Path(output_dir),
            config={
                key: WandbResultsWriter._as_scalar(value=value) for key, value in vars(args).items()
            },
            # Deterministic, so re-running a seed updates a recognisable run rather
            # than minting an unrelated one with a random adjective-animal name.
            run_name=f"{method}-seed{seed}",
            job_type=method,
            tags=(environment, method),
        )

    def record_checkpoint(self, *, metrics: Metrics) -> None:
        scalars = CheckpointScalars.from_metrics(metrics=metrics)
        if scalars is None:
            return
        self._handle().log(scalars.model_dump())

    def close(self, *, metrics: Metrics) -> None:
        """Write the run summary and finish the W&B run.

        A run that never opened a handle stays unopened: `close` fires from a `finally`
        on every run, and initialising W&B here purely to record that nothing happened
        would leave an empty run behind for every crash before the first sweep."""
        if self._run is None:
            return
        self._run.summary.update(RunSummaryScalars.from_metrics(metrics=metrics).model_dump())
        self._run.finish()
        self._run = None

    def _handle(self) -> Any:
        if self._run is None:
            # Lazy, inside the one method that needs it -- the same discipline
            # environments/tossing3d/kinder_backend.py holds for KINDER. This package
            # must import, typecheck and test on a machine with no wandb at all.
            import wandb

            # wandb.init(dir=...) requires the directory to exist; every other writer
            # here creates its own parent for the same reason.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._run = wandb.init(
                project=os.environ.get(PROJECT_ENV_VAR) or DEFAULT_PROJECT,
                # Offline unless the environment says otherwise -- see the module
                # docstring. W&B reads entity and group from its own env vars.
                mode=os.environ.get(MODE_ENV_VAR) or "offline",
                dir=str(self.output_dir),
                name=self.run_name,
                job_type=self.job_type,
                tags=list(self.tags),
                config=self.config,
            )
            # The learning-curve x-axis every other sidecar already keys on, declared
            # so W&B's charts use it instead of its own internal step counter.
            self._run.define_metric("num_online_transitions")
            self._run.define_metric("*", step_metric="num_online_transitions")
        return self._run

    @staticmethod
    def _as_scalar(*, value: object) -> str:
        """Everything stringified, the same shape `ConfigSnapshot.args` uses. A resolved
        namespace holds `Path`s and enums, which are not natively loggable, and a config
        that is a uniform `dict[str, str]` cannot half-serialize depending on which
        flags a domain happens to declare."""
        return str(value)
