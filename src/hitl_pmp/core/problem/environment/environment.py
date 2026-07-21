import abc
from typing import ClassVar

from gymnasium.spaces import Space
from pydantic import BaseModel

from .types import Action, State


class Environment(BaseModel, abc.ABC):
    """The real-world environment (or the real/ground-truth simulator standing in for
    it) -- a real, constructor-injected instance now (not a static-method container):
    grid_size-style per-run config and current_state are genuine per-instance state,
    not shared globally. It is **not** a reusable, stateless dynamics function that
    other code can call with a hypothetical state to explore "what if" -- a `Method`
    that needs to plan carries its own model for that; it must not borrow
    `Environment` to do it. `take_action(*, action)` advances `current_state` by one
    action via the domain's own underlying dynamics and returns the new state;
    `get_valid_actions()` reads from `current_state` too -- neither takes an explicit
    `state` argument, both operate on this instance's one real state.
    `get_current_state()`/`set_state()` are concrete (shared across every
    `Environment`, not reimplemented per domain) -- `set_state` is a *privileged
    external override* (used by a human, via `HumanOracle`/
    `Problem.execute_human_command`, to force a state -- distinct from
    `take_action`'s normal forward dynamics). `hard_reset()` resets to the initial
    state distribution but is only ever called by the harness before a run starts,
    never by the agent itself.

    Deliberately no reward function: this is a multi-task environment, and success
    is judged by goal-state reaching (Task.goal.is_satisfied), not a scalar reward --
    a fixed reward wouldn't make sense across tasks the agent invents for itself.

    action_space stays a ClassVar: it's a structural constant of the domain (same
    for every instance that will ever exist), not per-run configuration -- unlike
    current_state, nothing about it varies between two LightSwitchEnvironment
    instances constructed with different grid_size/tolerance values.
    """

    action_space: ClassVar[Space]
    current_state: State | None = None

    def get_current_state(self) -> State:
        assert self.current_state is not None, (
            "No current_state yet -- call hard_reset() or set_state() first."
        )
        return self.current_state

    @abc.abstractmethod
    def take_action(self, *, action: Action) -> State:
        """Advances current_state by one action, via the domain's own underlying
        dynamics (not via set_state, which is a privileged external override), and
        returns the new state."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_valid_actions(self) -> list[Action]:
        raise NotImplementedError

    def set_state(self, *, state: State) -> None:
        """External override: what happens when a human (via HumanOracle, called
        through Problem.execute_human_command) physically moves the real state --
        not the environment's own dynamics, and not a semantic reset."""
        self.current_state = state

    @abc.abstractmethod
    def hard_reset(self) -> None:
        """Reset to the initial state distribution. Only for the harness to call
        before a run starts -- never mid-practice, and never by the agent itself.
        Concrete implementations sample an initial state and call self.set_state on
        it."""
        raise NotImplementedError
