"""Measured, straight sweep paths; the robot, not object teleportation, executes them."""

import numpy as np
from pydantic import BaseModel, ConfigDict

TRAVEL = np.array([-1.0, 0.0, 0.0])


class Stroke(BaseModel):
    """Blade-edge centre path; ``axis`` is the unit blade line, yawed by the plow angle."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    axis: np.ndarray

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
    dy: float,
    phi_deg: float,
    blade_width: float,
    tray_back: float,
    clearance: float = 0.035,
    release_clearance: float = 0.025,
    band_margin: float = 0.01,
) -> Stroke:
    """Push straight toward the plate (world -x) with the blade yawed by ``phi_deg``.

    ``food_corners`` is grouped per food, shape (foods, points, 3). The blade's
    centre starts ``dy`` off the rack centre line and retreats along -travel until
    it is ``clearance`` behind every food whose lateral extent meets the blade's
    swept band (plus ``band_margin``). Food outside the band is not refused: the
    blade never touches it and it stays where physics leaves it. The stroke ends
    once the whole blade edge has passed the plate's near edge by
    ``release_clearance``. Reach, rack displacement and physical clearance are
    checked by the controller. Positive ``phi_deg`` makes the blade's +y end trail.
    """
    corners = np.asarray(food_corners, dtype=float)
    scalars = [
        rack_back,
        rack_front,
        rack_center_y,
        rack_top,
        dy,
        phi_deg,
        blade_width,
        tray_back,
        clearance,
        release_clearance,
        band_margin,
    ]
    if (
        corners.ndim != 3
        or corners.shape[2] != 3
        or corners.shape[0] == 0
        or corners.shape[1] == 0
        or not np.isfinite(corners).all()
        or not np.isfinite(scalars).all()
        or not -80 <= phi_deg <= 80
        or rack_back <= rack_front
        or tray_back > rack_front
        or blade_width <= 0
        or clearance < 0
        or release_clearance < 0
        or band_margin < 0
    ):
        raise ValueError("invalid finite sweep geometry, plow angle or food grouping")
    radians = np.deg2rad(phi_deg)
    axis = np.array([np.sin(radians), np.cos(radians), 0.0])
    start = np.array([rack_back - 0.02, rack_center_y + dy, rack_top + 0.001])
    half_band = blade_width / 2 * axis[1] + band_margin
    lows, highs = corners[:, :, 1].min(axis=1), corners[:, :, 1].max(axis=1)
    touched = corners[(highs >= start[1] - half_band) & (lows <= start[1] + half_band)].reshape(
        -1, 3
    )
    if len(touched):
        blade_x = start[0] + (touched[:, 1] - start[1]) * np.tan(radians)
        start[0] += max(0.0, float(np.max(touched[:, 0] + clearance - blade_x)))
    end_x = tray_back - abs(axis[0]) * blade_width / 2 - release_clearance
    if end_x >= start[0]:
        raise ValueError("serving edge lies behind the sweep start")
    end = np.array([end_x, start[1], start[2]])
    return Stroke(start=start, end=end, direction=TRAVEL.copy(), axis=axis)
