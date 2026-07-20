import abc
from pathlib import Path
from typing import Any

from .types import MetricsSnapshot


class Metrics(abc.ABC):
    """Evaluation protocol; a static-method container, never instantiated."""

    @staticmethod
    @abc.abstractmethod
    def percentage_success_per_task_test() -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def percentage_success_overall_test() -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def percentage_success_per_task_train() -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def percentage_success_overall_train() -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def num_complete_environment_resets() -> int:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def num_human_interventions() -> tuple[float, int]:
        """Returns (summed cost, count); should trend down as the agent learns to reset itself."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def summed_human_cost() -> float:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def record_evaluation(*, num_online_transitions: int, num_solved: int, num_total: int) -> None:
        """Records one evaluation checkpoint (e.g. after an online-learning cycle) --
        the building block task_training_curve() reports back out."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def task_training_curve() -> list[tuple[int, float]]:
        """(num_online_transitions, percentage_solved) pairs, in recorded order --
        e.g. Figure 4 of the "Practice Makes Perfect" paper plots exactly this,
        per approach per seed."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def task_training_curve_by_subtask() -> Any:
        raise NotImplementedError


class MetricsWriter:
    """Serializes a concrete Metrics implementation's full protocol to a JSON
    file -- written purely against the abstract Metrics interface (every field
    comes from calling metrics's own nine query methods), so this works
    unchanged for any current or future concrete Metrics/Method/environment
    combination, with no per-combination serialization code needed. Sits
    alongside Metrics in this file the same way core.renderer.VideoWriter sits
    alongside Renderer: a concrete support class for an abstract interface, not
    part of the interface itself. A static-method container, never
    instantiated, same as every other business-logic class in this project."""

    @staticmethod
    def write(*, metrics: type[Metrics], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = MetricsSnapshot(
            task_training_curve=metrics.task_training_curve(),
            task_training_curve_by_subtask=metrics.task_training_curve_by_subtask(),
            percentage_success_per_task_test=metrics.percentage_success_per_task_test(),
            percentage_success_overall_test=metrics.percentage_success_overall_test(),
            percentage_success_per_task_train=metrics.percentage_success_per_task_train(),
            percentage_success_overall_train=metrics.percentage_success_overall_train(),
            num_complete_environment_resets=metrics.num_complete_environment_resets(),
            num_human_interventions=metrics.num_human_interventions(),
            summed_human_cost=metrics.summed_human_cost(),
        )
        output_path.write_text(snapshot.model_dump_json(indent=2))
