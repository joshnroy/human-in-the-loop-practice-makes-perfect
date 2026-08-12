"""The flat scalar payloads a `ResultsWriter` reports, derived from `Metrics` in one
place so every writer reports the same numbers under the same names.

Flat and scalar on purpose. A `Metrics` is a rich object with computation methods on
it; an experiment-tracking backend wants `{name: number}`. Deriving that once here
rather than inside each writer means two writers cannot disagree about what
"num_solved" meant at a checkpoint, and it keeps the backend-specific class down to
the handful of lines that actually talk to the backend.

`num_solved`/`num_total` are kept as a **pair, never a rate**, so every reader reports
`x/y` and no denominator can go missing between here and a dashboard -- the same choice
`run_progress.RunProgress` makes for the same reason.
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
