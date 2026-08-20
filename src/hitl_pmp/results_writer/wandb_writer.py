"""`--record-wandb`: mirror a run's evaluation sweeps into Weights & Biases.

W&B is an **index and a viewer, never the system of record** -- `docs/experiment-logs/`
stays the durable committed record. What it adds over `stats.json` +
`config_snapshot.json` + `timing.json` + `progress.jsonl` is cross-run comparison
without writing a script, watching a sweep from another machine, and a durable index.

**Offline by default.** If `WANDB_MODE` is unset this passes `mode="offline"`: no run
blocks on the network, a machine with no credential still works, and a sweep of ~22
concurrent runs opens no sockets. Sync later with
`wandb sync <output-dir>/wandb/offline-run-*`. Only W&B's own four modes are accepted;
anything else raises before the run starts, since a silent downgrade is a run the
launcher believes is syncing and is not. `dir=<output-dir>`, so a run's W&B directory
sits beside the `stats.json` it describes rather than in W&B's cwd-relative default.

**`wandb sweep` is deliberately not adopted**: its agent draws its own hyperparameter
values, which is directly at odds with this project's fixed-seeds discipline.
`WANDB_RUN_GROUP` gives the grouping without that.

**One canonical run per experiment.** Names come from `run_naming.RunNamer`; before the
run starts this asks W&B whether the name is taken and **compares configurations**. A
same-name run with a different config means the namer is missing an axis -- a bug here --
and it fails loudly, naming the field. See `run_collision.py`. The check needs the
network, so it runs only when the run itself is online; offline runs print one line
saying it was skipped, and their collisions surface at `wandb sync`. It also cannot catch
two runs launched *simultaneously* under one name, since each queries before either
exists.

**Pure observer.** A run with `--record-wandb` writes a byte-identical `stats.json` to
one without, asserted end-to-end in `tests/results_writer/test_wandb_writer.py`. The one
weaker guarantee: any W&B call costs wall-clock, so `timing.json` from a `--record-wandb`
run is not strictly comparable to one without. Unmeasured."""

import argparse
import importlib.util
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, PrivateAttr

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.results_writer.results_writer import ResultsWriter
from hitl_pmp.results_writer.run_collision import RunNameCollisionCheck
from hitl_pmp.results_writer.run_naming import RunNamer
from hitl_pmp.results_writer.types import CheckpointScalars, ExistingRun, RunSummaryScalars

# One project per repo, not one per experiment: cross-run comparison is the entire
# point and it does not work across projects. Overridable by W&B's own WANDB_PROJECT.
DEFAULT_PROJECT = "hitl-pmp"

# W&B's own environment variables, read rather than re-declared as flags -- run_sweep.py
# already forwards os.environ to every child, so these configure a whole grid for free.
PROJECT_ENV_VAR = "WANDB_PROJECT"
MODE_ENV_VAR = "WANDB_MODE"
ENTITY_ENV_VAR = "WANDB_ENTITY"

# The only mode with an API to ask. Every other mode ("offline", "disabled", "dryrun")
# skips the collision check rather than opening a socket a sweep did not ask for.
ONLINE_MODE = "online"

# Bounds on the one query the check makes, so a slow or wedged API costs seconds rather
# than a run. The filter is on the exact name, so the realistic result size is 0 or 1;
# the cap only stops a pathological project from paginating forever.
API_TIMEOUT_SECONDS = 15
MAX_EXISTING_RUNS_INSPECTED = 25

# `wandb.init`'s `mode` parameter is a `Literal`, not a `str`, so an environment variable
# cannot reach it unvalidated. Spelled out here rather than imported from wandb, because
# this module must import and typecheck on a machine that does not have wandb at all --
# which is also why mypy on CI, where the extra is not installed, cannot see this
# mismatch: with no package to read, `wandb.init` is `Any` and every argument type-checks.
WandbMode = Literal["online", "offline", "disabled", "shared"]
VALID_MODES: tuple[WandbMode, ...] = ("online", "offline", "disabled", "shared")
DEFAULT_MODE: WandbMode = "offline"


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
    # Resolved once at open time rather than re-read at each checkpoint, so the mode a
    # run used is a readable property of the writer and a bad value fails before the run.
    mode: WandbMode

    _run: Any = PrivateAttr(default=None)

    @staticmethod
    def resolve_mode() -> WandbMode:
        """`WANDB_MODE`, narrowed to the literal `wandb.init` accepts.

        A real check rather than a cast: `os.environ.get` returns `str`, and an
        unrecognised value has to be rejected somewhere. It is rejected here, loudly,
        for the same reason `open_if_requested` refuses a missing dependency -- a typo'd
        `WANDB_MODE=onlien` that quietly fell back to offline would produce a run the
        launcher believes is syncing and is not, discovered only when the data is
        wanted. Unset and empty both mean the offline default."""
        requested = os.environ.get(MODE_ENV_VAR)
        if not requested:
            return DEFAULT_MODE
        for mode in VALID_MODES:
            if requested == mode:
                return mode
        raise ValueError(
            f"{MODE_ENV_VAR}={requested!r} is not one of {VALID_MODES}. W&B accepts only "
            "those four modes; leave it unset for this project's offline default."
        )

    @staticmethod
    def open_if_requested(
        *, args: argparse.Namespace, num_cycles: int
    ) -> "WandbResultsWriter | None":
        """This run's W&B writer, or None. Raises up front on every way this run can be
        unrecordable -- no `--output-dir`, no `wandb`, an unrecognised `WANDB_MODE`, a
        namespace the namer cannot name, or a name that is already some other
        experiment's -- rather than discovering any of them mid-run.

        `num_cycles` is ignored: W&B's own config already carries the resolved
        namespace, and the cycle count is only a *denominator*, which is
        `RunProgressWriter`'s concern rather than a dashboard's."""
        del num_cycles
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
        config = {
            key: WandbResultsWriter._as_scalar(value=value) for key, value in vars(args).items()
        }
        # Built from the resolved namespace by the shared namer, not assembled here:
        # the name is a convention every future tracker backend shares, and burying it
        # in one backend is how a second one ends up with a second convention.
        run_name = RunNamer.name(args=args)
        WandbResultsWriter._check_name_is_free(args=args, run_name=run_name, config=config)
        return WandbResultsWriter(
            output_dir=Path(output_dir),
            config=config,
            run_name=run_name,
            # `job_type`/`tags` stay the coarse facets W&B groups a run list by; the
            # name is what identifies it. Both read the namespace directly, and neither
            # is optional on any method or domain. `--wandb-tag` (repeatable) is
            # appended on top rather than replacing these two, so a whole experiment's
            # runs can be tag-filtered without losing the env/method grouping every
            # other run is filed under. `getattr`, not `args.wandb_tag`, since a
            # hand-built Namespace predating the flag has no such attribute.
            job_type=str(args.method),
            tags=(str(args.env), str(args.method), *(getattr(args, "wandb_tag", None) or [])),
            mode=WandbResultsWriter.resolve_mode(),
        )

    @staticmethod
    def _check_name_is_free(
        *, args: argparse.Namespace, run_name: str, config: dict[str, str]
    ) -> None:
        """Fail before the run starts if `run_name` is already taken in this project.

        Setup time on purpose: the whole value is in not discovering, hours in, that
        this run's results were about to land under a name that already means something
        else."""
        mode = os.environ.get(MODE_ENV_VAR) or "offline"
        if mode != ONLINE_MODE:
            # Not silent: a check that quietly does nothing is worse than no check,
            # because it is trusted. One line, on stderr, beside the run's other
            # launch-time diagnostics.
            print(
                f"[wandb] run name {run_name!r}: collision check skipped, "
                f"WANDB_MODE={mode!r} has no API to ask. A name clash will surface at "
                f"`wandb sync` instead. Run with WANDB_MODE=online to check up front.",
                file=sys.stderr,
            )
            return
        if not hasattr(args, "re_run"):
            raise ValueError(
                "--re-run is a global flag, so a resolved configuration always has it; "
                "this namespace does not, which means it is not one."
            )
        RunNameCollisionCheck.check(
            name=run_name,
            config=config,
            existing=WandbResultsWriter._existing_runs(run_name=run_name),
            re_run=bool(args.re_run),
        )

    @staticmethod
    def _existing_runs(*, run_name: str) -> tuple[ExistingRun, ...]:
        """Whatever W&B already holds under this name, reduced to plain data.

        Reduced at this boundary so `RunNameCollisionCheck` never touches a W&B object:
        that is what lets every case of the check, and the wording of both its errors,
        be tested with no network and no credential."""
        # Lazy, for the same reason `_handle` imports lazily: this package must import,
        # typecheck and test on a machine with no wandb at all.
        import wandb

        project = os.environ.get(PROJECT_ENV_VAR) or DEFAULT_PROJECT
        entity = os.environ.get(ENTITY_ENV_VAR)
        # W&B resolves the default entity itself when the path is a bare project.
        path = f"{entity}/{project}" if entity else project
        try:
            api = wandb.Api(timeout=API_TIMEOUT_SECONDS)
            found = api.runs(path, filters={"display_name": run_name})
            runs = [
                ExistingRun(
                    name=str(run.name),
                    identifier=str(run.id),
                    url=str(run.url),
                    config={key: str(value) for key, value in dict(run.config).items()},
                )
                for run in itertools.islice(found, MAX_EXISTING_RUNS_INSPECTED)
            ]
        except Exception as error:
            # An error, not a shrug. This run asked for WANDB_MODE=online, so it is
            # going to need this same API moments later in `wandb.init`; a run that
            # skipped the check here would simply fail further in, with less to say.
            raise ValueError(
                f"could not check {path!r} for an existing run named {run_name!r}, so "
                f"whether this run is the canonical one for its configuration is "
                f"unknown: {error}. Re-run with WANDB_MODE=offline to record locally "
                f"and sync later, or fix the credential/network first."
            ) from error
        return tuple(runs)

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
                # Offline unless the environment said otherwise -- see the module
                # docstring. W&B reads entity and group from its own env vars.
                mode=self.mode,
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
