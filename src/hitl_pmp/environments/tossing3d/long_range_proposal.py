"""Coupled proposals calibrated for KINDER's far-side Tossing3D receivers.

Calibration uses independent development scenes, never learner evaluation labels.
The fixed standoff keeps the base on the near side across the current receiver
support. Joint speed and release time then vary together within a tested band.
The ordinary geometry gate and learned classifier still evaluate each candidate.
"""

import numpy as np


class LongRangeTossProposal:
    """A state-independent proposal; it neither simulates nor supplies labels."""

    @staticmethod
    def sample(*, rng: np.random.Generator) -> np.ndarray:
        """Return metres, radians, degrees/second, and milliseconds.

        The release schedule joins the development calibration centers 390 deg/s
        at 460 ms and 420 deg/s at 450 ms, with a small timing perturbation.
        These are simulation settings under the existing threefold effort cap.
        """
        speed = float(rng.uniform(390.0, 420.0))
        release = 450.0 + (420.0 - speed) / 3.0 + float(rng.uniform(-2.0, 2.0))
        return np.array([2.5, 0.0, speed, release])
