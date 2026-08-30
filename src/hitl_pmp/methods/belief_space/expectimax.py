"""Exact finite-horizon expectimax over a fully specified belief MDP."""

import math
from collections.abc import Iterable
from functools import cache
from typing import Generic

from .types import (
    ActionT,
    AvailableActions,
    ChanceOutcome,
    ChanceOutcomes,
    ExpectimaxResult,
    StateT,
    StopValue,
)


def solve_expectimax(
    *,
    state: StateT,
    horizon: int,
    stop_value: StopValue[StateT],
    actions: AvailableActions[StateT, ActionT],
    outcomes: ChanceOutcomes[StateT, ActionT],
) -> ExpectimaxResult[ActionT]:
    """Exact-enumeration counterpart of understanding/pomdp_formulation.py.

    Pseudocode and review:
    https://github.com/joshnroy/human-in-the-loop-practice-makes-perfect/pull/290
    Algorithm notes (Google Drive reference from the pseudocode):
    https://drive.google.com/drive/folders/17j47M4NUGQIoKzNOo7yvWIhw13tE7h-a

    Stopping wins ties. Each call creates a solver with its own recursive cache;
    results are never reused across separate searches.
    """
    solver = _ExpectimaxSearch(stop_value=stop_value, actions=actions, outcomes=outcomes)
    return solver.solve(state=state, horizon=horizon)


class _ExpectimaxSearch(Generic[StateT, ActionT]):
    """One search's model callbacks and memoized recursion."""

    def __init__(
        self,
        *,
        stop_value: StopValue[StateT],
        actions: AvailableActions[StateT, ActionT],
        outcomes: ChanceOutcomes[StateT, ActionT],
    ) -> None:
        self.stop_value = stop_value
        self.actions = actions
        self.outcomes = outcomes
        # Decorate the bound method per instance, rather than sharing a class-level cache.
        self.solve = cache(self._solve)

    def _solve(self, *, state: StateT, horizon: int) -> ExpectimaxResult[ActionT]:
        if horizon < 0:
            raise ValueError(f"horizon must be non-negative, got {horizon}")

        current_best_value = self.stop_value(state=state)
        current_best_action: ActionT | None = None
        if not math.isfinite(current_best_value):
            raise ValueError(f"stop value must be finite, got {current_best_value}")

        if horizon == 0:
            return ExpectimaxResult(value=current_best_value, action=current_best_action)

        for practice_action in self.actions(state=state):
            value_of_state = 0.0
            for potential_next_state in _validated_outcomes(
                action=practice_action,
                branches=self.outcomes(state=state, action=practice_action),
            ):
                value_of_next_state = self.solve(
                    state=potential_next_state.next_state,
                    horizon=horizon - 1,
                ).value

                value_of_state += potential_next_state.probability * value_of_next_state

            if current_best_value < value_of_state:
                current_best_value = value_of_state
                current_best_action = practice_action

        return ExpectimaxResult(value=current_best_value, action=current_best_action)


def _validated_outcomes(
    *, action: ActionT, branches: Iterable[ChanceOutcome[StateT]]
) -> tuple[ChanceOutcome[StateT], ...]:
    outcomes = tuple(branches)
    if not outcomes:
        raise ValueError(f"action {action!r} has no chance outcomes")
    for outcome in outcomes:
        probability = outcome.probability
        if not math.isfinite(probability) or probability <= 0.0:
            raise ValueError(f"chance probability must be finite and positive, got {probability}")
    total_probability = sum(outcome.probability for outcome in outcomes)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"chance probabilities for {action!r} sum to {total_probability}, not 1")
    return outcomes
