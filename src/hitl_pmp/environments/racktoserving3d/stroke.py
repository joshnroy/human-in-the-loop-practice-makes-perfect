"""Measured GatherAndSweep paths; the robot, not object teleportation, executes them.

GatherAndSweep(s, delta, d) is five blade pushes over measured geometry:

1-2. x squeeze: the blade spans world y and pushes toward the plate (world -x) from
     just behind the rack's back edge, stopping ``s`` behind the rack's front edge.
     The blade is narrower than the rack, so there are two passes, centred on
     y = c +/- (rack width - blade width) / 2, where c is the plate centre's y.
3-4. y gathers: the blade spans world x at the rack's depth centre and pushes from
     just outside each long side toward y = c, stopping ``delta`` (an absolute
     distance, whatever the food) from it, +y side first.
5.   sweep: along y = c toward the plate, its height following rack, gap and plate,
     ending ``d`` before the plate's outer near edge.

Each phase is planned from food measured after the previous one.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict

TOWARD_PLATE = np.array([-1.0, 0.0, 0.0])
BEHIND_RACK = 0.03
APPROACH_CLEARANCE = 0.035
SIDE_MARGIN = 0.03
# Blade-edge height above the rack top. 3 mm, not 1: the arm's IK accepts up to 2 deg
# of rotation error, which drops one end of the 25 cm blade 4.4 mm, and the held-tool
# fixture check allows 3 mm of support overlap, so a 1 mm glide could be planned and
# then rejected mid-push (real-arm gathers, layouts 1-2, 2026-11-03).
EDGE_LIFT = 0.003
DESCENT_START_BEFORE_FRONT = 0.005
DESCENT_SLOPE = 0.6


class Stroke(BaseModel):
    """Blade-edge centre path; ``axis`` is the unit blade line.

    The path is straight in plan. Its height is constant unless ``descend_below_x``
    is set: past that world x (travelling -x) it drops at ``descent_slope`` until it
    reaches ``floor_z``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    start: np.ndarray
    end: np.ndarray
    direction: np.ndarray
    axis: np.ndarray
    descend_below_x: float | None = None
    descent_slope: float = DESCENT_SLOPE
    floor_z: float | None = None

    @property
    def length(self) -> float:
        return float(np.linalg.norm((self.end - self.start)[:2]))

    def point(self, *, distance: float) -> np.ndarray:
        distance = float(np.clip(distance, 0.0, self.length))
        point = self.start + distance * self.direction
        if self.descend_below_x is not None and self.floor_z is not None:
            drop = max(0.0, self.descend_below_x - point[0]) * self.descent_slope
            point[2] = max(self.floor_z, self.start[2] - drop)
        return point


class BladeHeightFeedback:
    """Hold the sweep's blade at its planned height when food under it lifts it.

    Past the rack's front edge a dumpling that has dropped onto the plate's rim sits
    below the blade edge's line; the edge meets its rounded top, the arm yields upward
    (10-13 mm measured, env seeds 10 and 18, 2026-11-12) and the blade rides over it.
    The correction lowers the commanded height by the measured lift, never by more
    than ``limit`` and never when the blade is at or below its plan, so the command
    stays above the rim by the plate clearance minus ``limit``.
    """

    @staticmethod
    def lowering(*, planned_z: float, actual_z: float, limit: float) -> float:
        return float(min(max(actual_z - planned_z, 0.0), max(limit, 0.0)))


class GatherAndSweepPlanner:
    @staticmethod
    def squeeze(
        *,
        s: float,
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_x: float,
        plate_center_y: float,
        plate_radius: float,
        blade_width: float,
    ) -> tuple[Stroke, Stroke]:
        """The +y-side pass, then the -y-side pass, both toward the plate."""
        GatherAndSweepPlanner._check_rack(
            values=(s, plate_center_x, plate_radius),
            rack_back=rack_back,
            rack_front=rack_front,
            rack_y_min=rack_y_min,
            rack_y_max=rack_y_max,
            rack_top=rack_top,
            plate_center_y=plate_center_y,
            blade_width=blade_width,
        )
        if not 0 <= s < rack_back - rack_front or blade_width > rack_y_max - rack_y_min:
            raise ValueError("squeeze stop must lie on the rack and the blade fit within it")
        half = (rack_y_max - rack_y_min) / 2 - blade_width / 2
        z = rack_top + EDGE_LIFT
        passes = tuple(
            Stroke(
                start=np.array([rack_back + BEHIND_RACK, y, z]),
                end=np.array([rack_front + s, y, z]),
                direction=TOWARD_PLATE.copy(),
                axis=np.array([0.0, 1.0, 0.0]),
            )
            for y in (plate_center_y + half, plate_center_y - half)
        )
        return passes[0], passes[1]

    @staticmethod
    def gather(
        *,
        food_corners: np.ndarray,
        delta: float,
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_x: float,
        plate_center_y: float,
        plate_radius: float,
        blade_width: float,
    ) -> tuple[Stroke, Stroke]:
        """The +y-side stroke (travel -y), then the -y-side stroke (travel +y).

        The blade spans world x, centred on the rack's depth. Each stroke starts
        ``SIDE_MARGIN`` outside the rack edge, or further if food overhangs it, and
        stops with the blade edge ``delta`` from y = c.
        """
        corners = GatherAndSweepPlanner._corners(food_corners=food_corners)
        GatherAndSweepPlanner._check_rack(
            values=(delta, plate_center_x, plate_radius),
            rack_back=rack_back,
            rack_front=rack_front,
            rack_y_min=rack_y_min,
            rack_y_max=rack_y_max,
            rack_top=rack_top,
            plate_center_y=plate_center_y,
            blade_width=blade_width,
        )
        if not delta > 0:
            raise ValueError("gather stop distance must be positive")
        x = (rack_front + rack_back) / 2
        z = rack_top + EDGE_LIFT
        stop = delta
        strokes = []
        for side, rack_edge in ((1.0, rack_y_max), (-1.0, rack_y_min)):
            outermost_food = float(np.max(side * corners[:, :, 1]))
            start_y = side * max(
                side * rack_edge + SIDE_MARGIN, outermost_food + APPROACH_CLEARANCE
            )
            end_y = plate_center_y + side * stop
            if side * (start_y - end_y) <= 0:
                raise ValueError("gather stop lies behind the gather start")
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
        d: float,
        plate_clear_z: float,
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_x: float,
        plate_center_y: float,
        plate_radius: float,
        blade_width: float,
    ) -> Stroke:
        """Push along y = c from behind the rack to ``d`` before the plate's near edge.

        The blade glides at rack height, then descends past the rack's front edge to
        ``plate_clear_z`` (just above the plate's rim) over the gap and the plate.
        """
        GatherAndSweepPlanner._check_rack(
            values=(d, plate_center_x, plate_radius, plate_clear_z),
            rack_back=rack_back,
            rack_front=rack_front,
            rack_y_min=rack_y_min,
            rack_y_max=rack_y_max,
            rack_top=rack_top,
            plate_center_y=plate_center_y,
            blade_width=blade_width,
        )
        near_edge = plate_center_x + plate_radius
        if (
            plate_radius <= 0
            or near_edge > rack_front
            or not 0 <= d <= 2 * plate_radius
            or plate_clear_z > rack_top + EDGE_LIFT
        ):
            raise ValueError("invalid sweep end depth, plate edge or plate clearance")
        z = rack_top + EDGE_LIFT
        descend_below_x = rack_front - DESCENT_START_BEFORE_FRONT
        end_x = near_edge - d
        end_z = max(plate_clear_z, z - max(0.0, descend_below_x - end_x) * DESCENT_SLOPE)
        return Stroke(
            start=np.array([rack_back + BEHIND_RACK, plate_center_y, z]),
            end=np.array([end_x, plate_center_y, end_z]),
            direction=TOWARD_PLATE.copy(),
            axis=np.array([0.0, 1.0, 0.0]),
            descend_below_x=descend_below_x,
            floor_z=plate_clear_z,
        )

    @staticmethod
    def _check_rack(
        *,
        values: tuple[float, ...],
        rack_back: float,
        rack_front: float,
        rack_y_min: float,
        rack_y_max: float,
        rack_top: float,
        plate_center_y: float,
        blade_width: float,
    ) -> None:
        if (
            not np.isfinite([
                *values,
                rack_back,
                rack_front,
                rack_y_min,
                rack_y_max,
                rack_top,
                plate_center_y,
                blade_width,
            ]).all()
            or rack_back <= rack_front
            or rack_y_max <= rack_y_min
            or blade_width <= 0
            or not rack_y_min < plate_center_y < rack_y_max
        ):
            raise ValueError("invalid finite stroke geometry or plate centre line off the rack")

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
