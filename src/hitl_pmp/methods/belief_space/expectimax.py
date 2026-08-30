"""Exact finite-horizon expectimax over a fully specified belief MDP.

The solver deliberately knows nothing about robots, skills, or how a belief is
represented.  A caller supplies the terminal utility, applicable actions, and the
complete chance distribution for each action.  A state may therefore already contain
the observed environment state, accumulated cost, and posterior sufficient statistics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT")


class ChanceOutcome(BaseModel, Generic[StateT]):
    """One exactly enumerated successor of an action."""

    model_config = ConfigDict(frozen=True)

    probability: float
    next_state: StateT


class ExpectimaxResult(BaseModel, Generic[ActionT]):
    """Optimal value and first action; ``None`` means stop now."""

    model_config = ConfigDict(frozen=True)

    value: float
    action: ActionT | None


def solve_expectimax(
    *,
    state: StateT,
    horizon: int,
    stop_value: Callable[[StateT], float],
    actions: Callable[[StateT], Iterable[ActionT]],
    outcomes: Callable[[StateT, ActionT], Iterable[ChanceOutcome[StateT]]],
) -> ExpectimaxResult[ActionT]:
    """Return the exact optimal finite-horizon practice decision.

    Stopping is available at every node and wins ties.  Chance outcomes must form a
    finite probability distribution; malformed models fail at the node where they are
    encountered rather than quietly returning a biased expectation.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    memo: dict[tuple[StateT, int], ExpectimaxResult[ActionT]] = {}

    def recurse(  # noqa: PLR0917 (private recursion uses a positional hot path)
        current: StateT, remaining: int
    ) -> ExpectimaxResult[ActionT]:
        key = (current, remaining)
        if key in memo:
            return memo[key]

        best = ExpectimaxResult[ActionT](value=float(stop_value(current)), action=None)
        if not math.isfinite(best.value):
            raise ValueError(f"stop value must be finite, got {best.value}")
        if remaining == 0:
            memo[key] = best
            return best

        for action in actions(current):
            branches = tuple(outcomes(current, action))
            if not branches:
                raise ValueError(f"action {action!r} has no chance outcomes")
            total_probability = 0.0
            action_value = 0.0
            for branch in branches:
                probability = float(branch.probability)
                if not math.isfinite(probability) or probability <= 0.0:
                    raise ValueError(
                        f"chance probability must be finite and positive, got {probability}"
                    )
                total_probability += probability
                child = recurse(branch.next_state, remaining - 1)
                action_value += probability * child.value
            if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    f"chance probabilities for {action!r} sum to {total_probability}, not 1"
                )
            if action_value > best.value:
                best = ExpectimaxResult(value=action_value, action=action)

        memo[key] = best
        return best

    return recurse(state, horizon)
