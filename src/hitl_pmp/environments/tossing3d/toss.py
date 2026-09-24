"""The composed toss's parameter pipeline, shared verbatim by both layouts.

`MoveToTossLocationAndToss` has three learned parameters, `[standoff, speed,
release]`. The fourth controller argument -- the rotation about the bin that decides
where the robot stands -- is chosen here per state by `TossDirectionSelector` and is
never sampled or learned. The barrier layout and the same-side layout both route the
toss through this one class (`SameSideSkills.TOSS is Tossing3DToss`), so their
dimension, bounds, proposal, direction selection, action encoding and classifier row
cannot drift apart.

## The classifier row is `[1, standoff, speed, release]`

It used to also carry the pre-move robot-frame bin displacement (forward, lateral)
and the yaw. The throw's physics depends only on the throw: the robot always ends up
exactly `standoff` from the bin, facing it, and the barrier was measured not to change
the throw's outcome by side. Where the robot happened to be *before* driving says
nothing about the landing, and it differs systematically between practice (same-side
bins) and evaluation (far-side bins), so it made evaluation rows look unlike practice
rows for no physical reason.
"""

from typing import Any, ClassVar

import numpy as np

from hitl_pmp.core.problem.environment.types import Action, State

from .environment import Tossing3DEnvironment
from .toss_direction import TossDirectionChoice, TossDirectionSelector
from .wide_long_range_proposal import (
    WIDE_TOSS_RELEASE_MS_BOUNDS,
    WIDE_TOSS_SPEED_BOUNDS,
    WIDE_TOSS_STANDOFF_BOUNDS,
    WideLongRangeTossProposal,
)

# The name the chosen direction is logged under, beside the three learned parameters.
TOSS_DIRECTION_ANNOTATION = "toss_direction_deg"


class Tossing3DToss:
    """A static-method container, never instantiated."""

    PARAM_BOUNDS: ClassVar[tuple[tuple[float, float], ...]] = (
        WIDE_TOSS_STANDOFF_BOUNDS,
        WIDE_TOSS_SPEED_BOUNDS,
        WIDE_TOSS_RELEASE_MS_BOUNDS,
    )

    @staticmethod
    def sample_params(*, rng: np.random.Generator) -> np.ndarray:
        return WideLongRangeTossProposal.sample(rng=rng)

    @staticmethod
    def choose_direction(*, state: State, params: np.ndarray) -> TossDirectionChoice:
        return TossDirectionSelector.select_for_state(state=state, standoff=float(params[0]))

    @staticmethod
    def rejection_reason(*, state: State, params: np.ndarray) -> str | None:
        """Never a rejection: a standoff with a plannable direction is accepted, and one
        with none raises `NoFeasibleTossDirectionError` from the selector."""
        Tossing3DToss.choose_direction(state=state, params=params)
        return None

    @staticmethod
    def compute_action(*, params: np.ndarray, state: State) -> Action:
        standoff, speed, release = (float(value) for value in params)
        choice = Tossing3DToss.choose_direction(state=state, params=params)
        return np.array(
            [
                Tossing3DEnvironment.move_to_toss_location_and_toss_id,
                standoff,
                choice.rotation,
                speed,
                release,
            ],
            dtype=float,
        )

    @staticmethod
    def feature_row(*, params: np.ndarray) -> list[float]:
        return [1.0, *(float(value) for value in params)]

    @staticmethod
    def action_annotations(*, action: Any) -> dict[str, float]:
        return {TOSS_DIRECTION_ANNOTATION: float(round(float(np.degrees(action[2])), 6))}
