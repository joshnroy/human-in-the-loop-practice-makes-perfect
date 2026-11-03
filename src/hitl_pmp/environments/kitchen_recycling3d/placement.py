"""Where a human may put the bin or a recovered item without colliding with anything.

A teleport into another body is resolved by MuJoCo as a violent ejection or a slow
mutual push, neither of which a human placement can produce.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from .domain import KitchenConfig

# Where a human puts the bin when it is brought beside the robot, before any clearance
# shift: on the kitchen floor, north of the robot's start.
NEAR_BIN_POSITION = (0.65, 1.4, 0.0)


def bin_position_retrievable(
    *, config: KitchenConfig, position: tuple[float, ...], tolerance: float = 0.0
) -> bool:
    """A conservative reach certificate for every possible contained item centre.

    The bin-origin sphere encloses its whole volume at any orientation; its distance to
    the allowed robot-centre rectangle plus that radius bounds the reach test.
    """
    lower = np.array(config.kitchen_bounds[:2]) + config.robot_radius
    upper = np.array(config.kitchen_bounds[2:]) - config.robot_radius
    if np.any(lower > upper):
        return False
    center = np.array(position[:2])
    nearest = np.clip(center, lower, upper)
    radius = np.linalg.norm(np.array(config.bin_outside) * [0.5, 0.5, 1.0])
    return bool(
        np.linalg.norm(center - nearest) + radius + tolerance <= config.direct_placement_reach
    )


class KitchenHumanPlacement:
    MARGIN = 0.05
    SEARCH_STEP = 0.05
    SEARCH_RADIUS = 1.5

    @staticmethod
    def near_bin_destination(
        *,
        config: KitchenConfig,
        robot: npt.ArrayLike,
        items: Sequence[npt.ArrayLike],
        furniture: Sequence[Sequence[float]],
    ) -> tuple[float, float, float]:
        """The fixed near spot when it is clear, else the closest clear spot to it.

        A shifted spot must still satisfy the same reach certificate the symbolic
        ResetCubeBinSameSide placement assumes, and stay entirely on the kitchen floor so the
        context stays near. `furniture` rows are (x, y, size_x, size_y)."""
        half = np.array(config.bin_outside[:2]) / 2
        margin = KitchenHumanPlacement.MARGIN
        step = KitchenHumanPlacement.SEARCH_STEP
        count = int(round(KitchenHumanPlacement.SEARCH_RADIUS / step))
        offsets = [
            (i * step, j * step) for i in range(-count, count + 1) for j in range(-count, count + 1)
        ]
        # Closest first; ties broken by offset so the choice is deterministic.
        offsets.sort(key=lambda offset: (round(np.hypot(*offset), 9), offset))
        lower = np.array(config.kitchen_bounds[:2]) + half
        upper = np.array(config.kitchen_bounds[2:]) - half
        for dx, dy in offsets:
            center = np.array(NEAR_BIN_POSITION[:2]) + (dx, dy)
            if np.any(center < lower) or np.any(center > upper):
                continue
            if KitchenHumanPlacement._distance_to_box(point=robot, center=center, half=half) < (
                config.robot_radius + margin
            ):
                continue
            if any(
                KitchenHumanPlacement._distance_to_box(point=item, center=center, half=half)
                < margin
                for item in items
            ):
                continue
            if any(
                np.all(np.abs(center - np.array(row[:2])) < half + np.array(row[2:4]) / 2 + margin)
                for row in furniture
            ):
                continue
            position = (float(center[0]), float(center[1]), NEAR_BIN_POSITION[2])
            if not bin_position_retrievable(config=config, position=position, tolerance=0.025):
                continue
            return NEAR_BIN_POSITION if (dx, dy) == (0, 0) else position
        raise ValueError("No clear reachable near-bin position")

    @staticmethod
    def recovery_clear_of_bin(
        *,
        config: KitchenConfig,
        position: npt.ArrayLike,
        item_radius: float,
        bin_pose: tuple[Sequence[float], Sequence[float]],
    ) -> bool:
        center, quaternion = bin_pose
        yaw = Rotation.from_quat(quaternion).as_euler("xyz")[2]
        cosine, sine = np.cos(-yaw), np.sin(-yaw)
        offset = np.asarray(position)[:2] - np.array(center[:2])
        local = np.array((
            cosine * offset[0] - sine * offset[1],
            sine * offset[0] + cosine * offset[1],
        ))
        half = np.array(config.bin_outside[:2]) / 2
        return KitchenHumanPlacement._distance_to_box(
            point=local, center=(0.0, 0.0), half=half
        ) >= (item_radius + KitchenHumanPlacement.MARGIN)

    @staticmethod
    def _distance_to_box(*, point: npt.ArrayLike, center: npt.ArrayLike, half: np.ndarray) -> float:
        outside = np.maximum(np.abs(np.asarray(point)[:2] - np.asarray(center)) - half, 0.0)
        return float(np.linalg.norm(outside))
