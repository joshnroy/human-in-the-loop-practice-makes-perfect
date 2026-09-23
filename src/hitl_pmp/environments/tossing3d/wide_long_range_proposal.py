"""The barrier layout's only toss proposal: deliberately wide, so learning is real.

Its removed predecessor, the calibrated long-range proposal, coupled release time to
speed along its development ridge, so nearly every candidate it emitted already
scored -- an untrained learner started at 10/10 and practice had nothing to teach it
(the calibration itself survives as `release_ridge`). This proposal
keeps the standoff and yaw that guarantee a legal launch pose for every bin position,
but draws speed and release time *independently* over a band that straddles that
ridge: the support still contains a scoring setting for every tested bin location
(the 2026-09-22 coverage sweep's success witnesses at its nine corner/edge/center
locations -- a proof for those nine points, not for the continuum between them)
alongside a large region of completed physical misses (its miss witnesses).

That fixed launch pose only exists for *far* receivers. `sample_robot_side` is the
robot-side variant -- joint (standoff, yaw) draws, see its bounds' comment.
"""

import numpy as np

from .skills import (
    TOSS_DISTANCE_BOUNDS,
    TOSS_RELEASE_MS_BOUNDS,
    TOSS_ROTATION_BOUNDS,
    TOSS_SPEED_BOUNDS,
)

# Chosen so every witness tuple from the coverage sweep lies inside: its slowest miss
# witness sits at (330 deg/s, 520 ms), outside the calibrated 390-420 deg/s band.
WIDE_TOSS_SPEED_BOUNDS = (330.0, 420.0)
WIDE_TOSS_RELEASE_MS_BOUNDS = (430.0, 520.0)
STANDOFF_M = 2.5
YAW_RAD = 0.0

# Robot-side receivers cannot use the fixed far-receiver geometry above: with the
# reset's bin yaw of 180 deg the toss ray points east, so a 2.5 m yaw-0 standoff
# from any bin in the grasp-safe rectangle (x in [-0.9, 0.2]) lands past the
# barrier at x = 1.25 and the geometry gate rejects every draw -- the 2026-09-22
# validation run's 0-of-10,000 pools. Standoff and yaw are therefore drawn
# *jointly* over the controller's full ranges and the per-state geometry gate
# keeps the feasible subset: short standoffs stay west of the barrier near the
# center line, and wide yaws swing the toss location along y where standoff alone
# cannot fit.
ROBOT_SIDE_STANDOFF_BOUNDS = TOSS_DISTANCE_BOUNDS
ROBOT_SIDE_YAW_BOUNDS = TOSS_ROTATION_BOUNDS

# The far band above has no scoring support at robot-side standoffs below 2.0 m:
# the 2026-09-22 verification run's short-standoff practice draws scored 0/31,
# and the 2026-09-23 controller-wide coverage probe (10 speeds x 8 releases at
# standoffs {1.25, 1.5, 1.75} x 3 receiver positions, 720 completed trials)
# found 0/240 scoring cells inside that band at every short standoff. Scoring
# support does exist there, on an anti-correlated (speed, release) ridge running
# from slow-late (115 deg/s, 840 ms) to fast-early (325-420 deg/s, 400 ms) --
# both ends outside the far band. Robot-side speed and release are therefore
# drawn over the controller's full ranges, like standoff and yaw already are;
# far receivers keep the calibrated band and are byte-identical
# (`test_far_receiver_draws_are_byte_identical_across_the_robot_side_widening`).
ROBOT_SIDE_SPEED_BOUNDS = TOSS_SPEED_BOUNDS
ROBOT_SIDE_RELEASE_MS_BOUNDS = TOSS_RELEASE_MS_BOUNDS


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

    @staticmethod
    def sample_robot_side(*, rng: np.random.Generator) -> np.ndarray:
        """The robot-side variant: all four parameters drawn jointly over the
        controller's full ranges -- see the bounds' comments above for why.

        Return metres, radians, degrees/second, and milliseconds, like `sample`.
        """
        standoff = float(rng.uniform(*ROBOT_SIDE_STANDOFF_BOUNDS))
        yaw = float(rng.uniform(*ROBOT_SIDE_YAW_BOUNDS))
        speed = float(rng.uniform(*ROBOT_SIDE_SPEED_BOUNDS))
        release = float(rng.uniform(*ROBOT_SIDE_RELEASE_MS_BOUNDS))
        return np.array([standoff, yaw, speed, release])

    @staticmethod
    def contains_robot_side(*, params: np.ndarray) -> bool:
        """Whether a row lies in the robot-side variant's closed support."""
        standoff, yaw, speed, release = (float(value) for value in params)
        return (
            ROBOT_SIDE_STANDOFF_BOUNDS[0] <= standoff <= ROBOT_SIDE_STANDOFF_BOUNDS[1]
            and ROBOT_SIDE_YAW_BOUNDS[0] <= yaw <= ROBOT_SIDE_YAW_BOUNDS[1]
            and ROBOT_SIDE_SPEED_BOUNDS[0] <= speed <= ROBOT_SIDE_SPEED_BOUNDS[1]
            and ROBOT_SIDE_RELEASE_MS_BOUNDS[0] <= release <= ROBOT_SIDE_RELEASE_MS_BOUNDS[1]
        )
