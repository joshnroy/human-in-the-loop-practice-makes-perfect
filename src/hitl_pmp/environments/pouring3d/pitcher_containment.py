"""KinDER's `PouringPitcher.contains_points`, evaluated for all points at once.

Upstream rebuilds the cavity ring and runs a dozen small numpy calls per point, in a
Python loop; with every payload piece's support vertices classified every control tick,
that loop was most of a tick. This evaluates the same expressions, in the same order, as
arrays, so every float and so every answer is identical.
"""

import numpy as np
from kinder.envs.dynamic3d.pitcher_pour3d import PouringPitcher


class PitcherContainment:
    @staticmethod
    def contains_points(
        *, pitcher: PouringPitcher, points: np.ndarray, tolerance: float = 0.002
    ) -> bool:
        if points[:, 2].min() < pitcher.wall_thickness - tolerance:
            return False
        if points[:, 2].max() > pitcher.height + tolerance:
            return False
        rings = PitcherContainment.cavity_rings(pitcher=pitcher, heights=points[:, 2])
        following = np.roll(rings, -1, axis=1)
        delta = following - rings
        planar = points[:, None, :2]
        u = np.clip(np.sum((planar - rings) * delta, axis=2) / np.sum(delta * delta, axis=2), 0, 1)
        distances = np.linalg.norm(planar - rings - u[:, :, None] * delta, axis=2)
        near = distances.min(axis=1) <= tolerance
        y = points[:, 1:2]
        crossing = (rings[:, :, 1] > y) != (following[:, :, 1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            intersections = rings[:, :, 0] + (y - rings[:, :, 1]) * delta[:, :, 0] / delta[:, :, 1]
        crossings = np.count_nonzero(crossing & (intersections > points[:, 0:1]), axis=1)
        return bool(np.all(near | (crossings % 2 == 1)))

    @staticmethod
    def cavity_rings(*, pitcher: PouringPitcher, heights: np.ndarray) -> np.ndarray:
        """`PouringPitcher.cavity_ring(height)` (inner) for every height, stacked."""
        fraction = np.clip(heights / pitcher.height, 0, 1)
        angles = np.linspace(0, 2 * np.pi, pitcher.wall_segments, endpoint=False)
        taper = 11 / 14 + 3 / 14 * fraction
        radii = (
            np.column_stack((
                pitcher.length / 2 * np.cos(angles),
                pitcher.width / 2 * np.sin(angles),
            ))[None]
            * taper[:, None, None]
        )
        flare = np.maximum(0.0, (fraction - 0.14 / 0.178) / (1 - 0.14 / 0.178))
        radii[:, :, 0] += (pitcher.spout_flare * flare)[:, None] * (
            np.maximum(0, np.cos(angles)) ** 12
        )[None]
        radii -= pitcher.wall_thickness * np.column_stack((np.cos(angles), np.sin(angles)))[None]
        return radii
