import abc
from typing import ClassVar

from gymnasium.spaces import Space

from .types import Action, State


class Environment(abc.ABC):
    """The real-world environment (or the real/ground-truth simulator standing in for
    it) — a static-method container, never instantiated. There is exactly one of it,
    tracked via current_state; it is not a reusable dynamics function for hypothetical
    rollouts. A Method that needs to plan carries its own model for that — it must not
    call Environment with a hypothetical state to explore "what if". Concrete
    subclasses set action_space and any of their own internal state as class
    attributes; all methods are static and keyword-only. The most external/
    foundational module in core/ — nothing else here is imported into it.

    Deliberately no reward function: this is a multi-task environment, and success
    is judged by goal-state reaching (Task.goal.is_satisfied), not a scalar reward —
    a fixed reward wouldn't make sense across tasks the agent invents for itself.
    """

    action_space: ClassVar[Space]
    current_state: ClassVar[State]

    @staticmethod
    def get_current_state() -> State:
        return Environment.current_state

    @staticmethod
    @abc.abstractmethod
    def take_action(*, action: Action) -> State:
        """Advances current_state by one action, via the domain's own underlying
        dynamics (not via set_state, which is a privileged external override), and
        returns the new state."""
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_valid_actions() -> set[Action]:
        raise NotImplementedError

    @staticmethod
    def set_state(*, state: State) -> None:
        """External override: what happens when a human (via HumanOracle, called
        through Problem.send_human_command) physically moves the real state — not
        the environment's own dynamics, and not a semantic reset."""
        Environment.current_state = state

    @staticmethod
    @abc.abstractmethod
    def hard_reset() -> None:
        """Reset to the initial state distribution. Only for the harness to call before
        a run starts — never mid-practice, and never by the agent itself. Concrete
        implementations sample an initial state and call Environment.set_state on it."""
        raise NotImplementedError
