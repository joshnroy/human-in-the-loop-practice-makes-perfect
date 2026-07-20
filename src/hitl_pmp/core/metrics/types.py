from typing import Any

from pydantic import BaseModel


class MetricsSnapshot(BaseModel):
    """A full point-in-time snapshot of one concrete Metrics implementation's
    entire query surface -- what MetricsWriter.write serializes to
    --output-dir/stats.json. A real, statically-checked shape (mirrors every
    other structured value in this codebase, e.g. Task/Goal/GroundAtom) rather
    than an untyped dict handed straight to json.dumps."""

    task_training_curve: list[tuple[int, float]]
    # Matches Metrics.task_training_curve_by_subtask's own Any return type --
    # deliberately unconstrained on the ABC itself, since a future concrete
    # Metrics might report subtask curves in a genuinely different shape.
    task_training_curve_by_subtask: Any
    percentage_success_per_task_test: dict[str, float]
    percentage_success_overall_test: float
    percentage_success_per_task_train: dict[str, float]
    percentage_success_overall_train: float
    num_complete_environment_resets: int
    num_human_interventions: tuple[float, int]
    summed_human_cost: float
