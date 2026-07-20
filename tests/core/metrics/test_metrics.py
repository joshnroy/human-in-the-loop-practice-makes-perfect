import json
from pathlib import Path
from typing import Any

from hitl_pmp.core.metrics.metrics import Metrics, MetricsWriter


def test_metrics_declares_expected_abstract_methods() -> None:
    assert Metrics.__abstractmethods__ == frozenset({
        "percentage_success_per_task_test",
        "percentage_success_overall_test",
        "percentage_success_per_task_train",
        "percentage_success_overall_train",
        "num_complete_environment_resets",
        "num_human_interventions",
        "summed_human_cost",
        "record_evaluation",
        "task_training_curve",
        "task_training_curve_by_subtask",
    })


class _FakeMetrics(Metrics):
    @staticmethod
    def percentage_success_per_task_test() -> dict[str, float]:
        return {"light_on": 0.5}

    @staticmethod
    def percentage_success_overall_test() -> float:
        return 0.5

    @staticmethod
    def percentage_success_per_task_train() -> dict[str, float]:
        return {}

    @staticmethod
    def percentage_success_overall_train() -> float:
        return 0.0

    @staticmethod
    def num_complete_environment_resets() -> int:
        return 2

    @staticmethod
    def num_human_interventions() -> tuple[float, int]:
        return (3.5, 1)

    @staticmethod
    def summed_human_cost() -> float:
        return 3.5

    @staticmethod
    def record_evaluation(*, num_online_transitions: int, num_solved: int, num_total: int) -> None:
        raise NotImplementedError

    @staticmethod
    def task_training_curve() -> list[tuple[int, float]]:
        return [(0, 0.0), (10, 0.5)]

    @staticmethod
    def task_training_curve_by_subtask() -> Any:
        return {"light_on": [(0, 0.0), (10, 0.5)]}


def test_metrics_writer_write_serializes_every_read_method(*, tmp_path: Path) -> None:
    output_path = tmp_path / "stats.json"
    MetricsWriter.write(metrics=_FakeMetrics, output_path=output_path)
    snapshot = json.loads(output_path.read_text())
    assert snapshot == {
        "task_training_curve": [[0, 0.0], [10, 0.5]],
        "task_training_curve_by_subtask": {"light_on": [[0, 0.0], [10, 0.5]]},
        "percentage_success_per_task_test": {"light_on": 0.5},
        "percentage_success_overall_test": 0.5,
        "percentage_success_per_task_train": {},
        "percentage_success_overall_train": 0.0,
        "num_complete_environment_resets": 2,
        "num_human_interventions": [3.5, 1],
        "summed_human_cost": 3.5,
    }


def test_metrics_writer_write_creates_missing_parent_directories(*, tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "results" / "stats.json"
    MetricsWriter.write(metrics=_FakeMetrics, output_path=output_path)
    assert output_path.exists()
