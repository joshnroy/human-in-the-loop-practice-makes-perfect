from typing import Any

import numpy as np

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.types import (
    GroundSkill,
    LabeledAction,
    Policy,
    Rollout,
    SetupCommand,
)
from hitl_pmp.core.problem.environment.environment import Environment
from hitl_pmp.core.problem.environment.types import Action, Object, State, Type
from hitl_pmp.core.problem.tasks.types import Goal, Task

_BLOCK = Type(name="block", feature_names=("x",))
_OBJ = Object(name="block1", type=_BLOCK)


class _Env(Environment):
    def take_action(self, *, action: Action) -> State:
        raise NotImplementedError

    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def hard_reset(self) -> None:
        raise NotImplementedError


class _MinimalMethod(Method):
    """Implements only the six genuinely abstract methods -- deliberately does NOT
    override get_practice_policy/end_cycle, so this pins their concrete defaults."""

    def reset_environment(self, *, start_state: State) -> bool:
        raise NotImplementedError

    def get_task_policy(self, *, task: Task) -> Policy:
        del task
        return lambda state: LabeledAction(action=np.array([0.0]), label="minimal")

    def generate_train_task(self, *, tbd_inputs: Any) -> Task:
        raise NotImplementedError

    def execute_setup_command(self, *, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    def execute_skill(self, *, skill: GroundSkill) -> Rollout:
        raise NotImplementedError

    def improve_skill_parameters(self, *, skill: GroundSkill, rollout: Rollout) -> None:
        raise NotImplementedError


def _task() -> Task:
    return Task(initial_state=State(data={_OBJ: np.array([0.0])}), goal=Goal(atoms=frozenset()))


def test_method_declares_expected_abstract_methods() -> None:
    """get_practice_policy/end_cycle are deliberately NOT in this set: they're
    concrete defaults, so a Method that doesn't practice (every baseline built so
    far) needs no boilerplate to satisfy them."""
    assert Method.__abstractmethods__ == frozenset({
        "reset_environment",
        "get_task_policy",
        "generate_train_task",
        "execute_setup_command",
        "execute_skill",
        "improve_skill_parameters",
    })


def test_get_practice_policy_defaults_to_get_task_policy() -> None:
    method = _MinimalMethod(env=_Env())
    task = _task()
    state = task.initial_state
    practice = method.get_practice_policy(task=task)
    evaluation = method.get_task_policy(task=task)
    assert practice(state).label == evaluation(state).label


def test_end_cycle_defaults_to_a_no_op() -> None:
    """A non-learning Method has nothing to retrain, so the default must be safe to
    call unconditionally from PracticeLoop -- it just has to not raise."""
    method = _MinimalMethod(env=_Env())
    method.end_cycle()


def test_a_method_can_override_practice_policy_independently_of_task_policy() -> None:
    """The whole point of the split: a learning Method explores during practice but
    exploits during evaluation, without the two codepaths being conflated."""

    class _SplitMethod(_MinimalMethod):
        def get_practice_policy(self, *, task: Task) -> Policy:
            del task
            return lambda state: LabeledAction(action=np.array([1.0]), label="practicing")

    method = _SplitMethod(env=_Env())
    task = _task()
    state = task.initial_state
    assert method.get_practice_policy(task=task)(state).label == "practicing"
    assert method.get_task_policy(task=task)(state).label == "minimal"
