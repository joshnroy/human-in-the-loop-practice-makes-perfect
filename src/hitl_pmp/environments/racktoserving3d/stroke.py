"""Measured, straight sweep paths; the robot, not object teleportation, executes them."""

import numpy as np
from pydantic import BaseModel, ConfigDict


class Stroke(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


def plan_stroke(
    *,
    food_corners: np.ndarray,
    rack_back: float,
    rack_front: float,
    rack_center_y: float,
    rack_top: float,
    angle_deg: float,
    blade_width: float,
    tray_back: float,
    clearance: float = 0.035,
    release_clearance: float = 0.025,
) -> Stroke:
    """Retreat along the selected line until the tilted blade clears all food.

    The angle still selects the trajectory; preparation never silently re-aims it.
    Reach, rack displacement and physical clearance are checked by the controller.
    """
    corners = np.asarray(food_corners, dtype=float)
    scalars = [
        rack_back,
        rack_front,
        rack_center_y,
        rack_top,
        angle_deg,
        blade_width,
        tray_back,
        clearance,
        release_clearance,
    ]
    if (
        corners.ndim != 2
        or corners.shape[1] != 3
        or len(corners) == 0
        or not np.isfinite(corners).all()
        or not np.isfinite(scalars).all()
        or not -80 <= angle_deg <= 80
        or rack_back <= rack_front
        or tray_back > rack_front
        or blade_width <= 0
        or clearance < 0
        or release_clearance < 0
    ):
        raise ValueError("invalid finite sweep geometry or angle")
    radians = np.deg2rad(angle_deg)
    direction = np.array([-np.cos(radians), np.sin(radians), 0.0])
    across = np.array([direction[1], -direction[0], 0.0])
    start = np.array([rack_back - 0.02, rack_center_y, rack_top + 0.001])
    if np.max(np.abs((corners - start) @ across)) > blade_width / 2 - 0.002:
        raise ValueError("food exceeds the selected blade width; choose another angle or pick")
    retreat = max(0.0, clearance - float(np.min((corners - start) @ direction)))
    start -= retreat * direction
    end_x = tray_back - abs(across[0]) * blade_width / 2 - release_clearance
    end = start + (end_x - start[0]) / direction[0] * direction
    return Stroke(start=start, end=end, direction=direction)
