"""The one kitchen drive rule every skill plans against.

The robot's footprint (radius `robot_radius`) stays on the kitchen floor: the mats are
the no-drive line. Planning adds the base controller's waypoint tolerance, so a planned
pose cannot be tracked into a violation. Tossing3D's toss planning reads this zone off
the state (`Tossing3DState.drive_zone`). A skill parameter or reach claim that needs a
pose outside this zone is not a hard practice case; it is impossible, and was measured
as such on Della v3.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from hitl_pmp.environments.tossing3d.types import DRIVE_ZONE_TOLERANCE_M

from .domain import KitchenConfig


class KitchenDriveZone:
    # kinder_models.dynamic3d.utils.WAYPOINT_TOLERANCE at the pinned commit.
    TRACKING_MARGIN = 0.04
    # PickCubeController.TARGET_DISTANCE: pick parks its base this far from the cube.
    PICK_REACH = 0.55
    STANDOFF_RESOLUTION = 0.001

    @staticmethod
    def bounds(*, config: KitchenConfig) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper corners of the region the robot's centre may be planned into."""
        inset = config.robot_radius + KitchenDriveZone.TRACKING_MARGIN
        return (
            np.array(config.kitchen_bounds[:2]) + inset,
            np.array(config.kitchen_bounds[2:]) - inset,
        )

    @staticmethod
    def allows(*, config: KitchenConfig, position: npt.ArrayLike) -> bool:
        lower, upper = KitchenDriveZone.bounds(config=config)
        centre = np.asarray(position, dtype=float)[:2]
        tolerance = DRIVE_ZONE_TOLERANCE_M
        return bool(np.all(centre >= lower - tolerance) and np.all(centre <= upper + tolerance))

    @staticmethod
    def pick_reachable(*, config: KitchenConfig, position: npt.ArrayLike) -> bool:
        lower, upper = KitchenDriveZone.bounds(config=config)
        target = np.asarray(position, dtype=float)[:2]
        nearest = np.clip(target, lower, upper)
        return bool(np.linalg.norm(target - nearest) <= KitchenDriveZone.PICK_REACH)

    @staticmethod
    def standoff_range(
        *,
        config: KitchenConfig,
        bin_xy: Sequence[float],
        azimuth: float,
        lower: float,
        upper: float,
    ) -> tuple[float, float] | None:
        """The widest contiguous standoff band in [lower, upper] whose toss pose is allowed.

        The toss controller parks the base `standoff` from the bin along the absolute
        angle `azimuth` (bin yaw plus the toss rotation), facing the bin. None when no
        standoff in the band is reachable from the kitchen floor."""
        count = int(round((upper - lower) / KitchenDriveZone.STANDOFF_RESOLUTION)) + 1
        standoffs = np.linspace(lower, upper, count)
        direction = np.array((np.cos(azimuth), np.sin(azimuth)))
        allowed = [
            KitchenDriveZone.allows(
                config=config, position=np.asarray(bin_xy[:2], dtype=float) - s * direction
            )
            for s in standoffs
        ]
        best: tuple[int, int] | None = None
        start = None
        for index, ok in enumerate([*allowed, False]):
            if ok and start is None:
                start = index
            elif not ok and start is not None:
                if best is None or index - start > best[1] - best[0]:
                    best = (start, index)
                start = None
        if best is None:
            return None
        return float(standoffs[best[0]]), float(standoffs[best[1] - 1])
