"""The barrier layout's only toss proposal: one draw for every receiver side.

All four parameters -- standoff, yaw, speed, release -- are drawn jointly and
independently over the controller's full ranges, and the per-state geometry gate
keeps the feasible subset. The interface is deliberately identical for both bin
sides: the fixed far-receiver launch pose (standoff 2.5 m, yaw 0) that survived
from the calibrated proposal's development ridge was a point mass that (a) made
far evaluation a 2-parameter problem while robot-side practice explored four,
(b) starved refits of far-side variety, and (c) kept every "far" number tied to
one geometry. Its removal intentionally supersedes the far-receivers-byte-
identical contract two earlier PRs pinned.

History of the support: the calibrated predecessor coupled release to speed
along its development ridge, so an untrained learner started at 10/10 and
practice had nothing to teach it (the calibration survives as `release_ridge`).
The 2026-09-22 far coverage sweep and the 2026-09-23 controller-wide
short-standoff probe together show scoring support across the standoff range on
anti-correlated (speed, release) ridges, alongside a large region of completed
physical misses -- which is what makes learning real on both sides.
"""

import numpy as np

from .skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
)

# The controller's own limits, restated under the proposal's name so the
# declared support and the controller bounds are asserted equal by test rather
# than accidentally coinciding.
WIDE_TOSS_STANDOFF_BOUNDS = TOSS_DISTANCE_BOUNDS
WIDE_TOSS_YAW_BOUNDS = TOSS_ROTATION_BOUNDS
WIDE_TOSS_SPEED_BOUNDS = TOSS_SPEED_BOUNDS
WIDE_TOSS_RELEASE_MS_BOUNDS = TOSS_RELEASE_MS_BOUNDS


class WideLongRangeTossProposal:
    """A state-independent proposal; it neither simulates nor supplies labels."""

    @staticmethod
    def release_ridge(*, speed: float) -> float:
        """The calibrated proposal's release center for a speed, in milliseconds."""
        return 450.0 + (420.0 - speed) / 3.0

    @staticmethod
    def sample(*, rng: np.random.Generator) -> np.ndarray:
        """Return metres, radians, degrees/second, and milliseconds -- all four
        drawn jointly over the controller's full ranges, for either bin side."""
        standoff = float(rng.uniform(*WIDE_TOSS_STANDOFF_BOUNDS))
        yaw = float(rng.uniform(*WIDE_TOSS_YAW_BOUNDS))
        speed = float(rng.uniform(*WIDE_TOSS_SPEED_BOUNDS))
        release = float(rng.uniform(*WIDE_TOSS_RELEASE_MS_BOUNDS))
        return np.array([standoff, yaw, speed, release])

    @staticmethod
    def contains(*, params: np.ndarray) -> bool:
        """Whether a `(standoff, yaw, speed, release)` row lies in the closed support."""
        standoff, yaw, speed, release = (float(value) for value in params)
        return (
            WIDE_TOSS_STANDOFF_BOUNDS[0] <= standoff <= WIDE_TOSS_STANDOFF_BOUNDS[1]
            and WIDE_TOSS_YAW_BOUNDS[0] <= yaw <= WIDE_TOSS_YAW_BOUNDS[1]
            and WIDE_TOSS_SPEED_BOUNDS[0] <= speed <= WIDE_TOSS_SPEED_BOUNDS[1]
            and WIDE_TOSS_RELEASE_MS_BOUNDS[0] <= release <= WIDE_TOSS_RELEASE_MS_BOUNDS[1]
        )
