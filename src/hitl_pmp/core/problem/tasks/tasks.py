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

    def sample_train_task_in_place(self) -> Task:
        """A training task for the world the environment is ALREADY in.

        `PracticeLoop` calls this instead of `sample_train_task` under
        `PracticeResetPolicy.NEVER`, where the period must begin from whatever the
        previous one left behind and nothing is allowed to move the environment. The
        two differ only for a domain whose task *sampling* is itself an operation on
        the world -- which the default below assumes it is not, and which
        `PracticeLoop` verifies rather than trusts.

        The default is right wherever building a task is arithmetic (Light Switch,
        Tossing Room, Ball Ring): there a sampled task is a genuinely fresh draw, and
        installing it is a separate act the loop is free to decline. Override this only
        when sampling touches the environment.

        Tossing3D is the domain that does, and the reason this method exists. Its
        `build_task` can only obtain an initial `State` by really rebuilding the MuJoCo
        scene, so `sample_train_task` re-seeds the live simulator as a side effect.
        Under `NEVER` the loop then declined to *install* that task's initial state and
        recorded zero resets, while the scene had already been rebuilt underneath it --
        a reset-free arm that was reset every cycle and reported that it was not. Every
        field of both arms' `stats.json` matched on 10/10 seeds except
        `num_practice_resets` (100 against 0). The invariant was never a property of
        the call site; it held only because every caller happened to call
        `reset_to_task` afterwards, which is exactly what `NEVER` stopped doing.
        """
        return self.sample_train_task()
