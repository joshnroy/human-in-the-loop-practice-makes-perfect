import abc

import numpy as np
from pydantic import BaseModel, ConfigDict

from hitl_pmp.core.method.types import EpisodeTrace, Policy
from hitl_pmp.core.renderer.renderer import Renderer

from .environment.environment import Environment
from .environment.types import Action, State
from .human.human import HumanOracle
from .human.types import CommandGoalDescription, CommandStartStateDescription, Cost
from .tasks.tasks import Tasks
from .tasks.types import Goal, Task


class Problem(BaseModel, abc.ABC):
    """Composition root: Environment + HumanOracle + Tasks. A real,
    constructor-injected instance now (not a static-method container): env/tasks
    are required fields (references to the actual Environment/Tasks *instances*
    this Problem drives), human is optional since not every domain has one
    (LightSwitchProblem never sets it -- no irreversible action exists there).
    Mirrors the design doc's flat Problem(ABC): every method here is a thin
    passthrough to the relevant part, except run_task_episode, which is genuine
    orchestration logic each concrete Problem must implement.

    human stays type[HumanOracle] rather than an instance, unlike env/tasks: it has
    no state of its own to hold (see human.py's own docstring), so there's nothing
    for an instance to carry that the class itself doesn't already provide.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment
    tasks: Tasks
    human: type[HumanOracle] | None = None

    def get_current_state(self) -> State:
        return self.env.get_current_state()

    def take_action(self, *, action: Action) -> State:
        return self.env.take_action(action=action)

    def get_valid_actions(self) -> list[Action]:
        return self.env.get_valid_actions()

    def hard_reset(self) -> None:
        self.env.hard_reset()

    def reset_to_task(self, *, task: Task) -> State:
        """Install a task's initial state as the environment's current state, and
        return it.

        Harness-only, like hard_reset -- an agent never calls this. It exists so
        both places that start an episode from a task (an evaluation episode, and
        each of PracticeLoop's free periods) go through one named operation rather
        than reaching into self.env.set_state, whose documented role is the
        HumanOracle's privileged override."""
        self.env.set_state(state=task.initial_state)
        return self.env.get_current_state()

    def _describe_command(
        self, *, goal: Goal, target_state: State | None = None
    ) -> tuple[CommandStartStateDescription, CommandGoalDescription]:
        return (
            CommandStartStateDescription(state=self.get_current_state()),
            CommandGoalDescription(goal=goal, target_state=target_state),
        )

    def calculate_cost_for_human_command(
        self, *, goal: Goal, target_state: State | None = None
    ) -> Cost:
        """Query what asking the human for this would cost, without actually asking.

        target_state is passed straight through to the HumanOracle, defaulting to None
        so every existing caller is unchanged -- see CommandGoalDescription.target_state
        for what a command carrying one means."""
        assert self.human is not None, "calculate_cost_for_human_command needs self.human set."
        start, end = self._describe_command(goal=goal, target_state=target_state)
        return self.human.calculate_cost_for_human_command(
            command_start_state_description=start, command_goal_description=end
        )

    def execute_human_command(self, *, goal: Goal, target_state: State | None = None) -> None:
        """The only sanctioned reset: let the human work toward goal. No return value —
        query calculate_cost_for_human_command beforehand if the cost is needed; this
        method's only job is to make it happen. self.human is responsible for
        updating self.env (it was handed env directly) to reflect whatever actually
        happened, since only it knows what that was.

        Pass target_state to ask for a *reset* -- "put the world back into exactly this
        configuration" -- rather than for the goal to be brought about. Both descriptions
        reach the oracle together and it decides what it can do with them; this facade
        neither interprets nor validates either one."""
        assert self.human is not None, "execute_human_command needs self.human set."
        start, end = self._describe_command(goal=goal, target_state=target_state)
        self.human.execute_human_command(
            command_start_state_description=start, command_goal_description=end, env=self.env
        )

    def execute_movables_reset(self) -> None:
        """The *partial*-reset sanctioned command: let the human reposition whichever
        of this domain's own non-robot objects it considers movable, leaving the
        robot's own configuration untouched. No `goal`/`target_state` -- unlike
        `execute_human_command`, there is no target description for this facade to
        build and hand over; see `HumanOracle.execute_movables_reset` for why.

        Reached only from `HumanCubeBinResetRequested` (see `PracticeLoop`), which is
        only ever raised by a Method built against a domain whose `SkillProvider.
        human_cube_bin_reset_skill` opted in -- so by the time this runs,
        `self.env.reset_movables()` succeeding is an established contract, not
        something this facade re-checks."""
        assert self.human is not None, "execute_movables_reset needs self.human set."
        self.human.execute_movables_reset(env=self.env)

    def sample_train_task(self) -> Task:
        return self.tasks.sample_train_task()

    def sample_test_task(self) -> Task:
        return self.tasks.sample_test_task()

    def sample_train_task_in_place(self) -> Task:
        return self.tasks.sample_train_task_in_place()

    @abc.abstractmethod
    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray], EpisodeTrace]:
        """Run policy on task until goal reached or timeout; returns
        (succeeded, frames, trace).

        frames is empty unless renderer is given, in which case every run is
        optionally recordable through this one path -- one frame per step (including
        the initial state) via renderer.render_frame, no separate rendering-only
        codepath needed.

        trace is the full (state, labeled action) history of the episode, returned
        unconditionally (it costs one list append per step, not a rendered frame) --
        see EpisodeTrace's own docstring for why it is plain data rather than a
        recorder threaded through this call. A caller that wants it persisted (e.g.
        --record-episode-traces) reads it back and hands it to
        hitl_pmp.episode_traces.EpisodeTraceRecorder itself."""
        raise NotImplementedError
