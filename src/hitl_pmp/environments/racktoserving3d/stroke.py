"""Measured, straight GatherAndSweep paths; the robot, not object teleportation, executes them.

GatherAndSweep(delta, d, phi) is three straight blade pushes over measured geometry:
two gather strokes across the rack with the blade spanning world x, from the +y side
and then the -y side, each stopping ``delta`` past the line y = c (the plate centre's
y); then one sweep toward the plate (world -x) with the blade yawed by ``phi`` that
ends when the blade edge's centre is ``d`` past the plate's near edge. The sweep is
planned from food measured after the gather, so the two plans are separate calls.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

SWEEP_TRAVEL = np.array([-1.0, 0.0, 0.0])
APPROACH_CLEARANCE = 0.035
SIDE_MARGIN = 0.03
BAND_MARGIN = 0.01
EDGE_LIFT = 0.001


class Stroke(BaseModel):
    """Blade-edge centre path; ``axis`` is the unit blade line."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    axis: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


class GatherAndSweepPlanner:
    @staticmethod
    def gather(
        *,
        food_corners: np.ndarray,
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_x: float,
        plate_center_y: float,
        plate_radius: float,
        delta: float,
        blade_width: float,
    ) -> tuple[Stroke, Stroke]:
        """The +y-side stroke (travel -y), then the -y-side stroke (travel +y).

        The blade spans world x, centred on the rack's depth. Each stroke starts
        ``SIDE_MARGIN`` outside the rack edge, or further if food overhangs it, and
        ends with the blade edge ``delta`` past y = c along its own travel.
        """
        corners = GatherAndSweepPlanner._corners(food_corners=food_corners)
        if (
            not np.isfinite([
                rack_back,
                rack_front,
                rack_y_min,
                rack_y_max,
                rack_top,
                plate_center_x,
                plate_center_y,
                plate_radius,
                delta,
                blade_width,
            ]).all()
            or rack_back <= rack_front
            or rack_y_max <= rack_y_min
            or blade_width <= 0
            or not rack_y_min < plate_center_y < rack_y_max
        ):
            raise ValueError("invalid finite gather geometry or gather line off the rack")
        x = (rack_front + rack_back) / 2
        z = rack_top + EDGE_LIFT
        strokes = []
        for side, rack_edge in ((1.0, rack_y_max), (-1.0, rack_y_min)):
            # Distances measured outward from the centre line along this side.
            outermost_food = float(np.max(side * corners[:, :, 1]))
            start_y = side * max(
                side * rack_edge + SIDE_MARGIN, outermost_food + APPROACH_CLEARANCE
            )
            end_y = plate_center_y - side * delta
            if side * (start_y - end_y) <= 0:
                raise ValueError("gather line lies behind the gather start")
            strokes.append(
                Stroke(
                    start=np.array([x, start_y, z]),
                    end=np.array([x, end_y, z]),
                    direction=np.array([0.0, -side, 0.0]),
                    axis=np.array([1.0, 0.0, 0.0]),
                )
            )
        return strokes[0], strokes[1]

    @staticmethod
    def sweep(
        *,
        food_corners: np.ndarray,
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_x: float,
        plate_center_y: float,
        plate_radius: float,
        d: float,
        phi_deg: float,
        blade_width: float,
    ) -> Stroke:
        """Push straight toward the plate (world -x) on y = c with the blade yawed by phi.

        The start retreats until the blade is ``APPROACH_CLEARANCE`` behind every food
        whose lateral extent meets the blade's swept band; food outside the band is
        left where it is. Positive ``phi_deg`` makes the blade's +y end trail.
        """
        corners = GatherAndSweepPlanner._corners(food_corners=food_corners)
        near_edge = plate_center_x + plate_radius
        if (
            not np.isfinite([
                rack_back,
                rack_front,
                rack_y_min,
                rack_y_max,
                rack_top,
                plate_center_x,
                plate_center_y,
                plate_radius,
                d,
                phi_deg,
                blade_width,
            ]).all()
            or not -80 <= phi_deg <= 80
            or rack_back <= rack_front
            or blade_width <= 0
            or plate_radius <= 0
            or near_edge > rack_front
            or not 0 <= d <= 2 * plate_radius
        ):
            raise ValueError("invalid finite sweep geometry, plow angle or end depth")
        radians = np.deg2rad(phi_deg)
        axis = np.array([np.sin(radians), np.cos(radians), 0.0])
        start = np.array([rack_back - 0.02, plate_center_y, rack_top + EDGE_LIFT])
        half_band = blade_width / 2 * axis[1] + BAND_MARGIN
        lows, highs = corners[:, :, 1].min(axis=1), corners[:, :, 1].max(axis=1)
        touched = corners[(highs >= start[1] - half_band) & (lows <= start[1] + half_band)]
        touched = touched.reshape(-1, 3)
        if len(touched):
            blade_x = start[0] + (touched[:, 1] - start[1]) * np.tan(radians)
            start[0] += max(0.0, float(np.max(touched[:, 0] + APPROACH_CLEARANCE - blade_x)))
        end = np.array([near_edge - d, plate_center_y, start[2]])
        return Stroke(start=start, end=end, direction=SWEEP_TRAVEL.copy(), axis=axis)

    @staticmethod
    def _corners(*, food_corners: np.ndarray) -> np.ndarray:
        corners = np.asarray(food_corners, dtype=float)
        if (
            corners.ndim != 3
            or corners.shape[2] != 3
            or corners.shape[0] == 0
            or corners.shape[1] == 0
            or not np.isfinite(corners).all()
        ):
            raise ValueError("food corners must be finite and grouped per food")
        return corners
