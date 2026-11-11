"""The value of deploying with no human: one finite-horizon value iteration.

Every practice domain prices STOP the same way -- the probability that a held-out task
is solved within the evaluation horizon, as a function of each skill's competence --
and differs only in its transition table. A domain supplies that table; this solves it.
"""

from collections.abc import Callable, Hashable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

# (skill, state after it succeeds, state after it fails).
DeploymentOption = tuple[str, Any, Any]
DeploymentTransitions = Mapping[Any, tuple[DeploymentOption, ...]]


class DeploymentValueIteration:
    """A static-method container."""

    @staticmethod
    def values(
        *,
        transitions: DeploymentTransitions,
        terminal_values: Mapping[Any, float],
        competences: Mapping[str, NDArray[np.float64]],
        horizon: int,
        start: Hashable,
    ) -> NDArray[np.float64]:
        """Batch value at `start` with `horizon` skill attempts left.

        `transitions` holds the states the robot can still act in, `terminal_values`
        the absorbing ones and what each is worth. A state takes its best option; one
        with no option, or with no attempt left, is worth nothing. The competence
        arrays share one shape, which the result keeps.
        """
        shape = np.shape(next(iter(competences.values())))
        terminal = {state: np.full(shape, worth) for state, worth in terminal_values.items()}
        nothing = np.zeros(shape)
        value = {**dict.fromkeys(transitions, nothing), **terminal}
        for _ in range(horizon):
            previous = value
            value = dict(terminal)
            for state, options in transitions.items():
                best = nothing
                for skill, succeeded, failed in options:
                    competence = competences[skill]
                    best = np.maximum(
                        best,
                        competence * previous[succeeded] + (1.0 - competence) * previous[failed],
                    )
                value[state] = best
        return value[start].copy()

    @staticmethod
    def failure_keeps_state(
        *,
        moves: Mapping[Any, tuple[tuple[str, Any], ...]],
        solved: Callable[[Any], bool],
    ) -> tuple[dict[Any, tuple[DeploymentOption, ...]], dict[Any, float]]:
        """The table for a domain whose failed skill leaves the abstract state as it
        was: every `(skill, successor)` move fails back into its own state, and a
        solved state is absorbing and worth 1."""
        transitions = {
            state: tuple((skill, successor, state) for skill, successor in options)
            for state, options in moves.items()
            if not solved(state)
        }
        return transitions, {state: 1.0 for state in moves if solved(state)}
