import numpy as np
import pytest

from hitl_pmp.core.method.types import LabeledAction, Policy
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.human.human import HumanOracle
from hitl_pmp.core.problem.human.types import (
    CommandGoalDescription,
    CommandStartStateDescription,
    Cost,
)
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.tasks import Tasks
from hitl_pmp.core.problem.tasks.types import Goal, Task
from hitl_pmp.core.renderer.renderer import Renderer

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


def _state(*, x: float) -> State:
    return State(data={_OBJ: np.array([x])})


class _Env(Environment):
    def take_action(self, *, action: Action) -> State:
        self.set_state(state=_state(x=float(action[0])))
        return self.get_current_state()

    def get_valid_actions(self) -> list[Action]:
        return [np.array([1.0])]

    def hard_reset(self) -> None:
        self.set_state(state=_state(x=0.0))


class _Human(HumanOracle):
    @staticmethod
    def calculate_cost_for_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
    ) -> Cost:
        return 1.0

    @staticmethod
    def execute_human_command(
        *,
        command_start_state_description: CommandStartStateDescription,
        command_goal_description: CommandGoalDescription,
        env: Environment,
    ) -> None:
        env.set_state(state=_state(x=42.0))


class _Tasks(Tasks):
    def sample_train_task(self) -> Task:
        return Task(initial_state=_state(x=0.0), goal=Goal(atoms=frozenset()))

    def sample_test_task(self) -> Task:
        return Task(initial_state=_state(x=1.0), goal=Goal(atoms=frozenset()))


class _Problem(Problem):
    def run_task_episode(
        self, *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        del renderer
        return True, []


def _build_problem() -> _Problem:
    env = _Env()
    return _Problem(env=env, tasks=_Tasks(env=env), human=_Human)


def test_get_current_state_delegates_to_env() -> None:
    problem = _build_problem()
    problem.env.set_state(state=_state(x=7.0))
    assert problem.get_current_state()[_OBJ].tolist() == [7.0]


def test_take_action_delegates_to_env() -> None:
    problem = _build_problem()
    result = problem.take_action(action=np.array([5.0]))
    assert result[_OBJ].tolist() == [5.0]


def test_get_valid_actions_delegates_to_env() -> None:
    problem = _build_problem()
    actions = problem.get_valid_actions()
    assert len(actions) == 1
    assert actions[0].tolist() == [1.0]


def test_hard_reset_delegates_to_env() -> None:
    problem = _build_problem()
    problem.hard_reset()
    assert problem.get_current_state()[_OBJ].tolist() == [0.0]


def test_reset_to_task_installs_the_tasks_initial_state_and_returns_it() -> None:
    """The one named way to start an episode-like unit from a task -- used by both
    an evaluation episode and each of PracticeLoop's free periods, so neither has
    to reach into env.set_state, whose documented role is the HumanOracle's
    privileged override."""
    problem = _build_problem()
    problem.take_action(action=np.array([5.0]))
    assert problem.get_current_state()[_OBJ].tolist() == [5.0]

    task = Task(initial_state=State(data={_OBJ: np.array([42.0])}), goal=Goal(atoms=frozenset()))
    returned = problem.reset_to_task(task=task)

    assert returned[_OBJ].tolist() == [42.0]
    assert problem.get_current_state()[_OBJ].tolist() == [42.0]


def test_sample_train_and_test_task_delegate_to_tasks() -> None:
    problem = _build_problem()
    assert problem.sample_train_task().initial_state[_OBJ].tolist() == [0.0]
    assert problem.sample_test_task().initial_state[_OBJ].tolist() == [1.0]


def test_calculate_cost_for_human_command_delegates_to_human() -> None:
    problem = _build_problem()
    problem.env.set_state(state=_state(x=0.0))
    cost = problem.calculate_cost_for_human_command(goal=Goal(atoms=frozenset()))
    assert cost == 1.0


def test_calculate_cost_for_human_command_requires_human_to_be_set() -> None:
    env = _Env()
    problem = _Problem(env=env, tasks=_Tasks(env=env))
    with pytest.raises(AssertionError):
        problem.calculate_cost_for_human_command(goal=Goal(atoms=frozenset()))


def test_execute_human_command_lets_human_mutate_env() -> None:
    problem = _build_problem()
    problem.env.set_state(state=_state(x=0.0))
    problem.execute_human_command(goal=Goal(atoms=frozenset()))
    assert problem.get_current_state()[_OBJ].tolist() == [42.0]


def test_execute_human_command_requires_human_to_be_set() -> None:
    env = _Env()
    problem = _Problem(env=env, tasks=_Tasks(env=env))
    with pytest.raises(AssertionError):
        problem.execute_human_command(goal=Goal(atoms=frozenset()))


def test_describe_command_builds_start_and_goal_descriptions() -> None:
    problem = _build_problem()
    problem.env.set_state(state=_state(x=3.0))
    goal = Goal(atoms=frozenset())
    start, end = problem._describe_command(goal=goal)
    assert start.state[_OBJ].tolist() == [3.0]
    assert end.goal is goal


def test_problem_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Problem(env=_Env(), tasks=_Tasks(env=_Env()))  # type: ignore[abstract]


def test_concrete_subclass_implements_run_task_episode() -> None:
    problem = _build_problem()
    task = problem.sample_train_task()
    policy: Policy = lambda state: LabeledAction(  # noqa: E731
        action=np.array([0.0]), label="test"
    )
    solved, frames = problem.run_task_episode(task=task, policy=policy)
    assert solved is True
    assert frames == []


def test_two_problem_instances_are_wired_to_independent_environments() -> None:
    """The whole point of this refactor: no shared Problem.env global anymore."""
    first = _build_problem()
    second = _build_problem()
    first.env.set_state(state=_state(x=1.0))
    second.env.set_state(state=_state(x=2.0))
    assert first.get_current_state()[_OBJ].tolist() == [1.0]
    assert second.get_current_state()[_OBJ].tolist() == [2.0]
