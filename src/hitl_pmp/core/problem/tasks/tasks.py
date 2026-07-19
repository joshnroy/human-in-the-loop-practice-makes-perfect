import abc

from .types import Task


class Tasks(abc.ABC):
    """Task/goal generation — a static-method container, never instantiated."""

    @staticmethod
    @abc.abstractmethod
    def sample_train_task() -> Task:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def sample_test_task() -> Task:
        """A randomly sampled test task, not known to the agent ahead of time."""
        raise NotImplementedError
