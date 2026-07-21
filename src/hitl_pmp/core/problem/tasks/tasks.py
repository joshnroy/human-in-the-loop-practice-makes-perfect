import abc

from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.problem.environment.environment import Environment

from .types import Task


class Tasks(BaseModel, abc.ABC):
    """Task/goal generation -- a real, constructor-injected instance now (not a
    static-method container): env is required because building an initial State
    (what sample_train_task/sample_test_task ultimately return, wrapped in a Task)
    concretely needs domain knowledge (e.g. LightSwitchTasks calls
    self.env.build_initial_state). arbitrary_types_allowed is set here (not left to
    each concrete subclass) since every concrete Tasks is expected to hold at least
    one RNG stream (e.g. LightSwitchTasks.train_rng/test_rng), the same numpy-array
    reasoning State/Rollout/LabeledAction already apply for their own fields."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment

    @abc.abstractmethod
    def sample_train_task(self) -> Task:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_test_task(self) -> Task:
        """A randomly sampled test task, not known to the agent ahead of time."""
        raise NotImplementedError
