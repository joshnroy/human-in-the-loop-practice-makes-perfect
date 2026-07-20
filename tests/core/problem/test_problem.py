import numpy as np
import pytest

from hitl_pmp.core.method.types import Policy
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
    @staticmethod
    def take_action(*, action: Action) -> State:
        Environment.current_state = _state(x=float(action[0]))
        return Environment.current_state

    @staticmethod
    def get_valid_actions() -> list[Action]:
        return [np.array([1.0])]

    @staticmethod
    def hard_reset() -> None:
        Environment.set_state(state=_state(x=0.0))


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
        env: type[Environment],
    ) -> None:
        env.set_state(state=_state(x=42.0))


class _Tasks(Tasks):
    @staticmethod
    def sample_train_task() -> Task:
        return Task(initial_state=_state(x=0.0), goal=Goal(atoms=frozenset()))

    @staticmethod
    def sample_test_task() -> Task:
        return Task(initial_state=_state(x=1.0), goal=Goal(atoms=frozenset()))


class _Problem(Problem):
    @staticmethod
    def run_task_episode(
        *, task: Task, policy: Policy, renderer: type[Renderer] | None = None
    ) -> tuple[bool, list[np.ndarray]]:
        del renderer
        return True, []


def _wire_problem() -> None:
    Problem.env = _Env
    Problem.human = _Human
    Problem.tasks = _Tasks


def test_singleton_attributes_set_on_base_class_are_visible_via_subclass() -> None:
    _wire_problem()
    assert Problem.env is _Env
    assert _Problem.env is _Env


def test_get_current_state_delegates_to_env() -> None:
    _wire_problem()
    Environment.set_state(state=_state(x=7.0))
    assert Problem.get_current_state()[_OBJ].tolist() == [7.0]


def test_take_action_delegates_to_env() -> None:
    _wire_problem()
    result = Problem.take_action(action=np.array([5.0]))
    assert result[_OBJ].tolist() == [5.0]


def test_get_valid_actions_delegates_to_env() -> None:
    _wire_problem()
    actions = Problem.get_valid_actions()
    assert len(actions) == 1
    assert actions[0].tolist() == [1.0]


def test_hard_reset_delegates_to_env() -> None:
    _wire_problem()
    Problem.hard_reset()
    assert Problem.get_current_state()[_OBJ].tolist() == [0.0]


def test_sample_train_and_test_task_delegate_to_tasks() -> None:
    _wire_problem()
    assert Problem.sample_train_task().initial_state[_OBJ].tolist() == [0.0]
    assert Problem.sample_test_task().initial_state[_OBJ].tolist() == [1.0]


def test_calculate_cost_for_human_command_delegates_to_human() -> None:
    _wire_problem()
    cost = Problem.calculate_cost_for_human_command(goal=Goal(atoms=frozenset()))
    assert cost == 1.0


def test_execute_human_command_lets_human_mutate_env() -> None:
    _wire_problem()
    Environment.set_state(state=_state(x=0.0))
    Problem.execute_human_command(goal=Goal(atoms=frozenset()))
    assert Problem.get_current_state()[_OBJ].tolist() == [42.0]


def test_describe_command_builds_start_and_goal_descriptions() -> None:
    _wire_problem()
    Environment.set_state(state=_state(x=3.0))
    goal = Goal(atoms=frozenset())
    start, end = Problem._describe_command(goal=goal)
    assert start.state[_OBJ].tolist() == [3.0]
    assert end.goal is goal


def test_problem_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Problem()  # type: ignore[abstract]


def test_concrete_subclass_implements_run_task_episode() -> None:
    _wire_problem()
    task = Problem.sample_train_task()
    policy: Policy = lambda state: np.array([0.0])  # noqa: E731
    solved, frames = _Problem.run_task_episode(task=task, policy=policy)
    assert solved is True
    assert frames == []
