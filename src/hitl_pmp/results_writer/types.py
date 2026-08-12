"""The plain data this package passes around: the flat scalar payloads a
`ResultsWriter` reports, and the two backend-agnostic descriptions run naming needs.

The scalar payloads are derived from `Metrics` in one
place so every writer reports the same numbers under the same names.

Flat and scalar on purpose. A `Metrics` is a rich object with computation methods on
it; an experiment-tracking backend wants `{name: number}`. Deriving that once here
rather than inside each writer means two writers cannot disagree about what
"num_solved" meant at a checkpoint, and it keeps the backend-specific class down to
the handful of lines that actually talk to the backend.

`num_solved`/`num_total` are kept as a **pair, never a rate**, so every reader reports
`x/y` and no denominator can go missing between here and a dashboard -- the same choice
`run_progress.RunProgress` makes for the same reason.

`ExistingRun` and `RunNameField` are here for the same reason the payloads are: they
are plain data two modules share (`run_naming.py` renders a `RunNameField`,
`run_collision.py` compares an `ExistingRun`, and `wandb_writer.py` constructs one from
the W&B API), and neither mentions any particular tracker.
"""

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.metrics.metrics import Metrics


class RunSummaryScalars(BaseModel):
    """The whole run, as of whenever `ResultsWriter.close` fired.

    Frozen: a record of something that already happened.

    Unlike `CheckpointScalars` this is always constructible, including for a run that
    crashed before its first evaluation sweep -- `close` fires from a `finally`, so a
    summary that could be None would put a branch in every writer for the case where
    there is least to say. A run with no sweeps reports `num_checkpoints=0` and `0/0`,
    which is exactly true rather than invented."""

    model_config = ConfigDict(frozen=True)

    num_checkpoints: int
    num_online_transitions: int
    num_solved: int
    num_total: int
    num_planning_failures: int
    num_planning_attempts: int
    num_practice_resets: int
    num_human_interventions: int
    summed_human_cost: float

    @staticmethod
    def from_metrics(*, metrics: Metrics) -> "RunSummaryScalars":
        num_failures, num_attempts = metrics.total_planning_outcomes()
        # The LAST evaluation, not the first: for a learning Method the first sweep
        # runs before any practice, so reporting it would summarise the untrained
        # score -- the same reason method_runner.py prints evaluations[-1].
        final = metrics.evaluations[-1] if metrics.evaluations else (0, 0, 0)
        return RunSummaryScalars(
            num_checkpoints=len(metrics.evaluations),
            num_online_transitions=final[0],
            num_solved=final[1],
            num_total=final[2],
            num_planning_failures=num_failures,
            num_planning_attempts=num_attempts,
            num_practice_resets=metrics.num_practice_resets,
            num_human_interventions=metrics.num_human_interventions_recorded,
            summed_human_cost=metrics.summed_human_cost_recorded,
        )


class CheckpointScalars(BaseModel):
    """One evaluation sweep.

    Frozen, for the same reason. `checkpoint` indexes `Metrics.evaluations` (0-based,
    0 being the sweep before any practice), so a reader can join a writer's output back
    to `stats.json` without guessing an offset -- the same key `competence_log.jsonl`
    and `episode_traces.jsonl` already carry."""

    model_config = ConfigDict(frozen=True)

    checkpoint: int
    num_online_transitions: int
    num_solved: int
    num_total: int
    num_practice_resets: int
    num_human_interventions: int
    summed_human_cost: float

    @staticmethod
    def from_metrics(*, metrics: Metrics) -> "CheckpointScalars | None":
        """The sweep `metrics` most recently recorded, or None before there is one.

        None rather than a row of zeros: a reader could not tell an invented zero row
        apart from a genuine 0/0 sweep, and the caller that has nothing to report
        should report nothing -- the same lazy contract that keeps a recorder from
        leaving a stray empty file."""
        if not metrics.evaluations:
            return None
        num_online_transitions, num_solved, num_total = metrics.evaluations[-1]
        return CheckpointScalars(
            checkpoint=len(metrics.evaluations) - 1,
            num_online_transitions=num_online_transitions,
            num_solved=num_solved,
            num_total=num_total,
            num_practice_resets=metrics.num_practice_resets,
            num_human_interventions=metrics.num_human_interventions_recorded,
            summed_human_cost=metrics.summed_human_cost_recorded,
        )


class ExistingRun(BaseModel):
    """A run a tracker already holds, reduced to the three things a collision check
    needs: what it is called, how to compare it, and where a human can go and look.

    Frozen: a record of something that already happened, same as the payloads above.

    Backend-agnostic on purpose. The W&B API object carries dozens of attributes and
    lazily fetches more over the network; reducing it to this at the boundary is what
    lets every case of `RunNameCollisionCheck` be tested without a credential, a
    network, or `wandb` installed at all -- which matters because the check's error
    messages are as much the deliverable as the check is.

    `config` is the run's *resolved* configuration, stringified, exactly the shape
    `ConfigSnapshot.args` and `WandbResultsWriter.config` already use -- so a
    comparison never has to reason about how two backends serialized an enum."""

    model_config = ConfigDict(frozen=True)

    name: str
    # The tracker's own id, which is what a name is deliberately *not*: names are
    # curated and human-readable, ids are unique and opaque. Reported alongside the URL
    # so a message stays useful if the URL scheme ever changes.
    identifier: str
    url: str
    config: dict[str, str]


class RunNameField(BaseModel):
    """One component of a run's name: which resolved-namespace flag it reads, and how
    that flag's value is rendered into a token.

    Frozen, and declarative rather than a callable, so the whole naming convention is
    one readable table in `run_naming.py` instead of logic spread over branches.

    `optional` is the field that earns its keep. A flag can be genuinely absent from a
    resolved namespace -- `--num-cycles` and `--ask-for-help` come from the *method*'s
    own `add_arguments`, and `SkillOracleCli` adds none, so an oracle run has neither
    attribute. Declaring that per field is the point: `getattr(args, "num_cycles", 0)`
    would paper over exactly the omission the collision check exists to detect, and it
    is wrong the moment a method computes its own cycle count instead of passing the
    literal the default happens to match. An optional field is omitted from the name
    when absent, which is only safe because **whether it is absent is determined by a
    field already in the name** (`method` for the two above, `env` for a domain flag);
    any new optional field must hold that same property."""

    model_config = ConfigDict(frozen=True)

    dest: str
    # Rendered before the value, so a bare number reads as what it counts: "seed3",
    # "c100". Empty for a value that is already self-describing ("tossingroom", "never").
    prefix: str = ""
    # (when true, when false) for a store_true flag. A store_true's off state has no
    # word of its own, so naming both states is the only way a run says which ledge it
    # ran on rather than leaving a reader to infer it from a missing token.
    toggle: tuple[str, str] | None = None
    optional: bool = False
