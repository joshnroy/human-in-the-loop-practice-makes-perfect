"""Conservative geometric support test, separate from scored bin containment."""

import numpy as np
from scipy.spatial.transform import Rotation


class RimGeometry:
    @staticmethod
    def supported(*, cube: dict[str, float], bin_: dict[str, float], wall_thickness: float) -> bool:
        """A settled cube overlapping the upper wall of an upright bin.

        Compare oriented bounding extents in the bin frame. The 2 mm support
        tolerance allows MuJoCo contact penetration, not airborne near misses.
        Tilted/overturned bins are deliberately outside this controller's contract.
        """

        def rotation(*, features: dict[str, float]) -> np.ndarray:
            return Rotation.from_quat([features[k] for k in ("qx", "qy", "qz", "qw")]).as_matrix()

        if any(np.linalg.norm([obj[k] for k in ("vx", "vy", "vz")]) > 0.02 for obj in (cube, bin_)):
            return False
        rb = rotation(features=bin_)
        if rb[2, 2] < 0.995:
            return False
        rc = rotation(features=cube)
        center = rb.T @ np.array([cube[k] - bin_[k] for k in ("x", "y", "z")])
        half = np.abs(rb.T @ rc) @ (np.array([cube[k] for k in ("bb_x", "bb_y", "bb_z")]) / 2)
        if abs(center[2] - half[2] - bin_["bb_z"]) > 0.002:
            return False
        widths = np.array([bin_["bb_x"], bin_["bb_y"]]) / 2
        low, high = center[:2] - half[:2], center[:2] + half[:2]
        if np.any(low > widths) or np.any(high < -widths):
            return False
        return bool(
            np.any(high >= widths - wall_thickness) or np.any(low <= -widths + wall_thickness)
        )
