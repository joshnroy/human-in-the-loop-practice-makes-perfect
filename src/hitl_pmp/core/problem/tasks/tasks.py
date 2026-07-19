import abc

from .types import Task


class Tasks(abc.ABC):
    """Task/goal generation — a static-method container, never instantiated."""

    @staticmethod
    @abc.abstractmethod
    def get_train_tasks() -> list[Task]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_test_task() -> Task:
        """A randomly sampled test task, not known to the agent ahead of time."""
        raise NotImplementedError
