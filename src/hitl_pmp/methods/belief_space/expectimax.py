"""Exact finite-horizon expectimax over a fully specified belief MDP."""

import math
from collections.abc import Iterable

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
    _memo: dict[tuple[StateT, int], ExpectimaxResult[ActionT]] | None = None,
) -> ExpectimaxResult[ActionT]:
    """Exact-enumeration counterpart of understanding/pomdp_formulation.py.

    Stopping wins ties. The memo is shared only within one recursive search.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")
    if _memo is None:
        _memo = {}
    key = (state, horizon)
    if key in _memo:
        return _memo[key]

    current_best_value = stop_value(state=state)
    current_best_action: ActionT | None = None
    if not math.isfinite(current_best_value):
        raise ValueError(f"stop value must be finite, got {current_best_value}")

    if horizon == 0:
        result = ExpectimaxResult[ActionT](value=current_best_value, action=current_best_action)
        _memo[key] = result
        return result

    for practice_action in actions(state=state):
        value_of_state = 0.0
        for potential_next_state in _validated_outcomes(
            action=practice_action,
            branches=outcomes(state=state, action=practice_action),
        ):
            value_of_next_state = solve_expectimax(
                state=potential_next_state.next_state,
                horizon=horizon - 1,
                stop_value=stop_value,
                actions=actions,
                outcomes=outcomes,
                _memo=_memo,
            ).value

            value_of_state += potential_next_state.probability * value_of_next_state

        if current_best_value < value_of_state:
            current_best_value = value_of_state
            current_best_action = practice_action

    result = ExpectimaxResult(value=current_best_value, action=current_best_action)
    _memo[key] = result
    return result


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
