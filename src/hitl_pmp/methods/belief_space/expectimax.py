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
) -> ExpectimaxResult[ActionT]:
    """Choose between stopping and practicing, looking ahead at most horizon actions.

    Stopping wins ties. Repeated states share a cached solution at the same depth.
    """
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}")

    return _solve_state(
        state=state,
        remaining=horizon,
        stop_value=stop_value,
        actions=actions,
        outcomes=outcomes,
        memo={},
    )


def _solve_state(
    *,
    state: StateT,
    remaining: int,
    stop_value: StopValue[StateT],
    actions: AvailableActions[StateT, ActionT],
    outcomes: ChanceOutcomes[StateT, ActionT],
    memo: dict[tuple[StateT, int], ExpectimaxResult[ActionT]],
) -> ExpectimaxResult[ActionT]:
    key = (state, remaining)
    if key not in memo:
        memo[key] = _choose_best_action(
            state=state,
            remaining=remaining,
            stop_value=stop_value,
            actions=actions,
            outcomes=outcomes,
            memo=memo,
        )
    return memo[key]


def _choose_best_action(
    *,
    state: StateT,
    remaining: int,
    stop_value: StopValue[StateT],
    actions: AvailableActions[StateT, ActionT],
    outcomes: ChanceOutcomes[StateT, ActionT],
    memo: dict[tuple[StateT, int], ExpectimaxResult[ActionT]],
) -> ExpectimaxResult[ActionT]:
    best: ExpectimaxResult[ActionT] = _stop_decision(value=stop_value(state=state))
    if remaining == 0:
        return best
    for action in actions(state=state):
        value = _expected_action_value(
            state=state,
            action=action,
            remaining=remaining,
            stop_value=stop_value,
            actions=actions,
            outcomes=outcomes,
            memo=memo,
        )
        if value > best.value:
            best = ExpectimaxResult(value=value, action=action)
    return best


def _expected_action_value(
    *,
    state: StateT,
    action: ActionT,
    remaining: int,
    stop_value: StopValue[StateT],
    actions: AvailableActions[StateT, ActionT],
    outcomes: ChanceOutcomes[StateT, ActionT],
    memo: dict[tuple[StateT, int], ExpectimaxResult[ActionT]],
) -> float:
    branches = _validated_outcomes(action=action, branches=outcomes(state=state, action=action))
    return sum(
        branch.probability
        * _solve_state(
            state=branch.next_state,
            remaining=remaining - 1,
            stop_value=stop_value,
            actions=actions,
            outcomes=outcomes,
            memo=memo,
        ).value
        for branch in branches
    )


def _stop_decision(*, value: float) -> ExpectimaxResult[ActionT]:
    if not math.isfinite(value):
        raise ValueError(f"stop value must be finite, got {value}")
    return ExpectimaxResult(value=value, action=None)


def _validated_outcomes(
    *, action: ActionT, branches: Iterable[ChanceOutcome[StateT]]
) -> tuple[ChanceOutcome[StateT], ...]:
    outcomes = tuple(branches)
    if not outcomes:
        raise ValueError(f"action {action!r} has no chance outcomes")
    for outcome in outcomes:
        _validate_probability(probability=outcome.probability)
    total_probability = sum(outcome.probability for outcome in outcomes)
    if not math.isclose(total_probability, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"chance probabilities for {action!r} sum to {total_probability}, not 1")
    return outcomes


def _validate_probability(*, probability: float) -> None:
    if not math.isfinite(probability) or probability <= 0.0:
        raise ValueError(f"chance probability must be finite and positive, got {probability}")
