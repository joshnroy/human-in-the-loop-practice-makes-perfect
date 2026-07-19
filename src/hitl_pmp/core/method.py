from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .structs import Action, State

if TYPE_CHECKING:
    from .problem import Task


class Method(abc.ABC):
    """The agent side: decides what to practice, executes skills, improves them.

    A static-method container, never instantiated; mirrors Problem on the
    environment side.
    """

    @staticmethod
    @abc.abstractmethod
    def reset_environment(*, start_state: State) -> bool:
        """The agent's own attempt to self-navigate to start_state, without human help."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_task_policy(*, task: Task) -> Policy:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def generate_train_task(*, tbd_inputs: Any) -> Task:
        """Decides what to practice next; exact inputs still TBD per the design doc."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def execute_setup_command(*, setup_command: SetupCommand) -> None:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def execute_skill(*, skill: Skill) -> Rollout:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def improve_skill_parameters(*, skill: Skill, rollout: Rollout) -> None:
        raise NotImplementedError


Policy = Callable[[State], Action]
Rollout = Any
Skill = Any
SetupCommand = Any
