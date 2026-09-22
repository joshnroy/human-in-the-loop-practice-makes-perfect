"""The barrier layout's only toss proposal: deliberately wide, so learning is real.

Its removed predecessor, the calibrated long-range proposal, coupled release time to
speed along its development ridge, so nearly every candidate it emitted already
scored -- an untrained learner started at 10/10 and practice had nothing to teach it
(the calibration itself survives as `release_ridge`). This proposal
keeps the standoff and yaw that guarantee a legal launch pose for every bin position,
but draws speed and release time *independently* over a band that straddles that
ridge: the support still contains a scoring setting for every location in the bin
region (the 2026-09-22 nine-location coverage sweep's success witnesses) alongside a
large region of completed physical misses (its miss witnesses).
"""

import numpy as np

# Chosen so every witness tuple from the coverage sweep lies inside: its slowest miss
# witness sits at (330 deg/s, 520 ms), outside the calibrated 390-420 deg/s band.
WIDE_TOSS_SPEED_BOUNDS = (330.0, 420.0)
WIDE_TOSS_RELEASE_MS_BOUNDS = (430.0, 520.0)
STANDOFF_M = 2.5
YAW_RAD = 0.0


class WideLongRangeTossProposal:
    """A state-independent proposal; it neither simulates nor supplies labels."""

    @staticmethod
    def release_ridge(*, speed: float) -> float:
        """The calibrated proposal's release center for a speed, in milliseconds."""
        return 450.0 + (420.0 - speed) / 3.0

    @staticmethod
    def sample(*, rng: np.random.Generator) -> np.ndarray:
        """Return metres, radians, degrees/second, and milliseconds."""
        speed = float(rng.uniform(*WIDE_TOSS_SPEED_BOUNDS))
        release = float(rng.uniform(*WIDE_TOSS_RELEASE_MS_BOUNDS))
        return np.array([STANDOFF_M, YAW_RAD, speed, release])

    @staticmethod
    def contains(*, params: np.ndarray) -> bool:
        """Whether a `(standoff, yaw, speed, release)` row lies in the closed support."""
        standoff, yaw, speed, release = (float(value) for value in params)
        return (
            standoff == STANDOFF_M
            and yaw == YAW_RAD
            and WIDE_TOSS_SPEED_BOUNDS[0] <= speed <= WIDE_TOSS_SPEED_BOUNDS[1]
            and WIDE_TOSS_RELEASE_MS_BOUNDS[0] <= release <= WIDE_TOSS_RELEASE_MS_BOUNDS[1]
        )
