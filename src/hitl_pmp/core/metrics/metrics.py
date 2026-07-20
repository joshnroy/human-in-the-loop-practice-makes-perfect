import abc
from typing import Any


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
