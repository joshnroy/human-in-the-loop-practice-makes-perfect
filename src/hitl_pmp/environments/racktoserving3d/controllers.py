"""Physical rack transfer and individual reset controllers, without object pose writes."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation, Slerp

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .backend import RackServingBackend
from .config import RackServingConfig
from .food import RackFoodController
from .servo import SweepServo
from .skills import GATHER_AND_SWEEP_BOUNDS, SWEEP_BOUNDS, SWEEP_SPEED_M_S
from .stroke import GatherAndSweepPlanner, Stroke
from .tool import RackScraper, RackShelfSkills

RELEASE_RETREAT = 0.03
PLATE_CLEARANCE = 0.008
LIFT_OFF_HEIGHT = 0.01
LIFT_OFF_STEP = 0.002
EXIT_BAND = 0.03
EXIT_RETREAT = 0.10
EXIT_STEP = 0.005


class StrokeStances(BaseModel):
    """Where the robot stands for each GatherAndSweep stroke, and the base routes between.

    The side stances (rack centre x, +/-0.85, facing the counter) and the rear stance
    (counter rear + 0.35, stroke y - 0.15, pi - 0.30) are the poses the 2026-11-03
    real-arm gather probe drove the arm from; ``rear_base`` is also the old SweepRack
    stance. Lanes run 0.5 m outside the counter's long sides and 0.65 m behind it.
    """

    model_config = ConfigDict(frozen=True)
    side_offset: float = 0.35
    lane_clearance: float = 0.50
    rear_clearance: float = 0.65
    rear_offset: float = 0.35
    rear_y_offset: float = -0.15
    rear_heading: float = np.pi - 0.30
    side_exit: Literal["front", "rear_lane"] = "rear_lane"

    def side_base(self, *, side: int, config: RackServingConfig) -> np.ndarray:
        return np.array([
            config.rack_center_x,
            side * (config.counter_width / 2 + self.side_offset),
            -side * np.pi / 2,
        ])

    def rear_base(self, *, y: float, config: RackServingConfig) -> np.ndarray:
        return np.array([
            config.counter_x_max + self.rear_offset,
            y + self.rear_y_offset,
            self.rear_heading,
        ])

    def lane(self, *, side: int, config: RackServingConfig) -> float:
        return side * (config.counter_width / 2 + self.lane_clearance)

    def rear_x(self, *, config: RackServingConfig) -> float:
        return config.counter_x_max + self.rear_clearance

    def rear_to_side(
        self, *, side: int, start: np.ndarray, config: RackServingConfig
    ) -> list[np.ndarray]:
        """From a rear stance: back to the rear lane, along it, turn, down the side lane."""
        lane, rear_x = self.lane(side=side, config=config), self.rear_x(config=config)
        base = self.side_base(side=side, config=config)
        return [
            np.array([rear_x, start[1], start[2]]),
            np.array([rear_x, lane, start[2]]),
            np.array([rear_x, lane, base[2]]),
            np.array([base[0], lane, base[2]]),
            base,
        ]

    def side_to_side(self, *, side: int, config: RackServingConfig) -> list[np.ndarray]:
        """From the other side stance round the rear to ``side``'s stance (probe route)."""
        here, there = self.lane(side=-side, config=config), self.lane(side=side, config=config)
        rear_x, heading = self.rear_x(config=config), side * np.pi / 2
        base = self.side_base(side=side, config=config)
        return [
            np.array([config.rack_center_x, here, -heading]),
            np.array([rear_x, here, -heading]),
            np.array([rear_x, there, -heading]),
            np.array([rear_x, there, heading]),
            np.array([base[0], there, heading]),
            base,
        ]

    def side_to_rear(self, *, side: int, config: RackServingConfig) -> list[np.ndarray]:
        """From a side stance out to its lane, to the rear lane, and turn to face the rack."""
        lane, rear_x = self.lane(side=side, config=config), self.rear_x(config=config)
        heading = self.side_base(side=side, config=config)[2]
        return [
            np.array([config.rack_center_x, lane, heading]),
            np.array([rear_x, lane, heading]),
            np.array([rear_x, lane, np.pi]),
        ]


class RackControllers:
    stances: ClassVar[StrokeStances] = StrokeStances()
    SETTLE_STEPS: ClassVar[int] = 30
    SETTLE_TOLERANCE: ClassVar[float] = 0.005
    SETTLE_LIMIT: ClassVar[float] = 0.06

    def __init__(self, *, backend: RackServingBackend, step_limit: int = 12000):
        self.backend = backend
        self.shelf = RackShelfSkills(env=backend.env(), tick_limit=step_limit)
        self.shelf.cupboard_geoms |= backend.body_geoms(name="rack_0")
        self.tool = RackScraper(shelf=self.shelf, config=backend.config)
        self.held_food: str | None = None
        self.metrics: dict[str, Any] = {}

    @property
    def steps(self) -> int:
        return self.shelf.ticks

    def initialize(self) -> None:
        self.tool.initialize_rest()
        self.backend._tool_home = self.tool.at_home

    @contextmanager
    def feedback(self) -> Generator[None, None, None]:
        self.shelf.ticks = 0
        try:
            with (
                self.tool.compensate_payload(),
                SweepServo(
                    shelf=self.shelf,
                    wiper=self.tool,
                    max_normal_force_n=0,
                ),
            ):
                yield
        finally:
            self.backend.grip = self.shelf.grip
            self.backend.sync_clock()

    def pick_scraper(self) -> None:
        if any(self.backend.held(name=name) for name in self.backend.cube_names):
            raise SkillFailed("food must be placed before picking the scraper")
        with self.feedback():
            self.tool.pick()

    def return_scraper(self) -> None:
        """Planned from wherever the scraper is held, behind the counter included: the
        base path goes round it. A scripted clear-and-circulate prefix used to run first
        from behind the counter and, from an arm left at a joint limit, failed after a
        few steps on every retry."""
        with self.feedback():
            self.tool.place()

    @staticmethod
    def push_rotation(*, phi: float, tilt: float = 45) -> np.ndarray:
        """Blade line along local x yawed to (sin phi, cos phi); lean stays normal to it."""
        return (
            Rotation.from_euler("z", -phi, degrees=True).as_matrix()
            @ Rotation.from_euler("y", tilt, degrees=True).as_matrix()
            @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
        )

    def around_counter(self, *, to_rear: bool) -> None:
        config, shelf, tool = self.backend.config, self.shelf, self.tool
        rear_x = config.counter_x_max + 0.60
        outside_y = config.counter_y_min - 0.80
        front_x = tool.base[0]
        route = (
            [
                [front_x, outside_y, 0.0],
                [rear_x, outside_y, 0.0],
                [rear_x, outside_y, np.pi],
                [rear_x, config.rack_center_y, np.pi],
            ]
            if to_rear
            else [
                [rear_x, shelf.data.qpos[shelf.ba[1]], np.pi],
                [rear_x, outside_y, np.pi],
                [rear_x, outside_y, 0.0],
                [front_x, outside_y, 0.0],
                tool.base.tolist(),
            ]
        )
        for index, waypoint in enumerate(route):
            if not to_rear and index == len(route) - 1:
                shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ tool.wiper_pose()
                tool.transit(
                    position=shelf.site_pose()[:3, 3],
                    rotation=tool.sweep_body_rotation(yaw_deg=0) @ shelf.in_hand[:3, :3].T,
                    label="return_outside_counter_reorient",
                    holding=True,
                )
            tool.move_base(target=np.asarray(waypoint), label=f"counter_circulation_{index}")

    def approach_surface(
        self,
        *,
        position: np.ndarray,
        rotation: np.ndarray,
        base: np.ndarray,
        holding: bool,
        label: str,
    ) -> None:
        shelf, tool = self.shelf, self.tool
        arm, _ = shelf.ik(position=position, rotation=rotation, base=base, holding=holding)
        tool.check_configuration(base=base, arm=arm, holding=holding)
        staging = shelf.site_at(base=shelf.data.qpos[shelf.ba].copy(), arm=arm).copy()
        tool.transit(
            position=staging[:3, 3],
            rotation=staging[:3, :3],
            label=f"{label}_arm_clearance",
            holding=holding,
        )
        tool.move_base(target=base, label=f"{label}_base")
        tool.line(position=position, rotation=rotation, label=f"{label}_hover", holding=holding)

    def hand_for_tool(self, *, body: np.ndarray) -> np.ndarray:
        self.tool.require_held(label="measured_tool_feedback")
        self.shelf.in_hand = np.linalg.inv(self.shelf.site_pose()) @ self.tool.wiper_pose()
        return body @ np.linalg.inv(self.shelf.in_hand)

    def move_tool(self, *, body: np.ndarray, label: str) -> None:
        for _ in range(5):
            target = self.hand_for_tool(body=body)
            self.tool.line(
                position=target[:3, 3], rotation=target[:3, :3], label=label, holding=True
            )
            actual = self.tool.wiper_pose()
            if (
                np.linalg.norm(actual[:3, 3] - body[:3, 3]) < 0.003
                and self.tool.rotation_error(actual=actual[:3, :3], target=body[:3, :3]) < 1.0
            ):
                return
        position_error = float(np.linalg.norm(actual[:3, 3] - body[:3, 3]))
        angle_error = self.tool.rotation_error(actual=actual[:3, :3], target=body[:3, :3])
        raise SkillFailed(
            f"{label}: measured scraper pose failed to converge "
            f"({position_error * 1000:.2f} mm / {angle_error:.2f} deg)"
        )

    def gather_and_sweep(self, *, s: float, delta: float, d: float) -> None:
        """Squeeze toward the plate in two passes, gather from +y then -y, then sweep."""
        bounds = GATHER_AND_SWEEP_BOUNDS
        values = (s, delta, d)
        if not np.isfinite(values).all() or any(
            not low <= value <= high for value, (low, high) in zip(values, bounds, strict=True)
        ):
            (s_low, s_high), (delta_low, delta_high), (d_low, d_high) = bounds
            raise SkillFailed(
                f"GatherAndSweep requires s [{s_low},{s_high}] m,"
                f" delta [{delta_low},{delta_high}] m and d [{d_low},{d_high}] m"
            )
        backend, shelf, tool = self.backend, self.shelf, self.tool
        config = backend.config
        if not self.on_rack():
            raise SkillFailed("no food is physically on the rack")
        rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
        if rack.top_z < tray.top_z or tray.x_max > rack.x_min:
            raise SkillFailed("live rack/tray geometry no longer provides a downward open exit")
        geometry = self.stroke_geometry()
        squeeze = GatherAndSweepPlanner.squeeze(s=s, **geometry)
        self.metrics = {
            "measured_gap_m": rack.x_min - tray.x_max,
            "rack_top_m": rack.top_z,
            "tray_top_m": tray.top_z,
            "s_m": s,
            "delta_m": delta,
            "d_m": d,
            "speed_m_s": SWEEP_SPEED_M_S,
            "max_rack_displacement_m": 0.0,
            "strokes": [],
        }
        rack_pose = np.asarray(backend.core().get_object("rack_0").get_pose()[0]).copy()
        stances = self.stances
        with self.feedback():
            for index, stroke in enumerate(squeeze):
                self.push_stroke(
                    stroke=stroke,
                    base=stances.rear_base(y=float(stroke.start[1]), config=config),
                    travel=self.to_rear if index == 0 else lambda: None,
                    rack_pose=rack_pose,
                    label=f"squeeze_{'plus' if index == 0 else 'minus'}_y",
                )
            names = self.on_rack() or list(backend.cube_names)
            gather = GatherAndSweepPlanner.gather(
                food_corners=np.stack([backend.corners(name=name) for name in names]),
                delta=delta,
                **geometry,
            )
            for side, stroke in zip((1, -1), gather, strict=True):

                def to_side(*, side: int = side) -> None:
                    here = shelf.data.qpos[shelf.ba].copy()
                    route = (
                        stances.rear_to_side(side=side, start=here, config=config)
                        if side > 0
                        else stances.side_to_side(side=side, config=config)
                    )
                    for index, waypoint in enumerate(route[:-1]):
                        tool.move_base(target=waypoint, label=f"gather_route_{index}")

                self.push_stroke(
                    stroke=stroke,
                    base=stances.side_base(side=side, config=config),
                    travel=to_side,
                    rack_pose=rack_pose,
                    label=f"gather_{'plus' if side > 0 else 'minus'}_y",
                )
            sweep = GatherAndSweepPlanner.sweep(d=d, plate_clear_z=self.plate_clear_z(), **geometry)

            self.push_stroke(
                stroke=sweep,
                base=stances.rear_base(y=float(sweep.start[1]), config=config),
                travel=self.to_rear,
                rack_pose=rack_pose,
                label="sweep",
            )

    def sweep(self, *, d: float) -> None:
        """GatherAndSweep's final stroke alone: no squeeze and no gathers."""
        ((d_low, d_high),) = SWEEP_BOUNDS
        if not np.isfinite(d) or not d_low <= d <= d_high:
            raise SkillFailed(f"Sweep requires d [{d_low},{d_high}] m")
        backend = self.backend
        if not self.on_rack():
            raise SkillFailed("no food is physically on the rack")
        rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
        if rack.top_z < tray.top_z or tray.x_max > rack.x_min:
            raise SkillFailed("live rack/tray geometry no longer provides a downward open exit")
        geometry = self.stroke_geometry()
        self.metrics = {
            "measured_gap_m": rack.x_min - tray.x_max,
            "rack_top_m": rack.top_z,
            "tray_top_m": tray.top_z,
            "d_m": d,
            "speed_m_s": SWEEP_SPEED_M_S,
            "max_rack_displacement_m": 0.0,
            "strokes": [],
        }
        rack_pose = np.asarray(backend.core().get_object("rack_0").get_pose()[0]).copy()
        stroke = GatherAndSweepPlanner.sweep(d=d, plate_clear_z=self.plate_clear_z(), **geometry)
        with self.feedback():
            self.push_stroke(
                stroke=stroke,
                base=self.stances.rear_base(y=float(stroke.start[1]), config=backend.config),
                travel=self.to_rear,
                rack_pose=rack_pose,
                label="sweep",
            )

    def on_rack(self) -> list[str]:
        backend = self.backend
        return [name for name in backend.cube_names if backend.location(name=name) == "OnRack"]

    def to_rear(self) -> None:
        """From wherever the base is to behind the counter, facing the rack.

        From a side stance, ``stances.side_exit`` picks the side lane to the rear, or
        the old SweepRack's circulation round the counter's front.
        """
        config, shelf = self.backend.config, self.shelf
        y = float(shelf.data.qpos[shelf.ba[1]])
        if abs(y) > config.counter_width / 2 and self.stances.side_exit == "rear_lane":
            side = 1 if y > 0 else -1
            for index, waypoint in enumerate(self.stances.side_to_rear(side=side, config=config)):
                self.tool.move_base(target=waypoint, label=f"to_rear_{index}")
        elif shelf.data.qpos[shelf.ba[0]] < config.counter_x_max:
            self.around_counter(to_rear=True)

    def plate_clear_z(self) -> float:
        """Sweep floor over the plate: 8 mm above its rim (or its top for flat targets).

        IK accepts 2 deg of blade roll, which drops one end of the blade 4.4 mm. Floors
        of 2 mm and 5 mm both let the blade clip box_0_rim_5..7 mid-sweep in 4/10
        real-arm layouts at d = 0.10, with the blade's lowest point 5 mm under its edge
        centre (2026-11-03). At 8 mm the descent from the 3 mm rack glide is 1 mm.
        """
        config, tray = self.backend.config, self.backend.support(name="box_0")
        top = config.corelle_rim_top_z if config.serving_target == "corelle_plate" else tray.top_z
        return float(max(top, tray.top_z) + PLATE_CLEARANCE)

    def stroke_geometry(self) -> dict[str, float]:
        """Live rack and plate geometry; the plate radius is its outer, not scoring, edge."""
        backend = self.backend
        rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
        center_x = (tray.x_min + tray.x_max) / 2
        return {
            "rack_back": rack.x_max,
            "rack_front": rack.x_min,
            "rack_y_min": rack.y_min,
            "rack_y_max": rack.y_max,
            "rack_top": rack.top_z,
            "plate_center_x": center_x,
            "plate_center_y": (tray.y_min + tray.y_max) / 2,
            "plate_radius": backend.serving_near_edge_x() - center_x,
            "blade_width": float(2 * self.shelf.model.geom_size[self.tool.head_geom, 0]),
        }

    @staticmethod
    def stroke_rotation(*, stroke: Stroke, tilt: float = 45) -> np.ndarray:
        """The sweep's held pose turned to this stroke's travel, so the lean matches it."""
        heading = float(np.arctan2(stroke.direction[1], stroke.direction[0]) - np.pi)
        local = Rotation.from_euler("z", -heading).as_matrix() @ stroke.axis
        if local[1] < 0:
            local = -local
        phi = float(np.degrees(np.arctan2(local[0], local[1])))
        return Rotation.from_euler("z", heading).as_matrix() @ RackControllers.push_rotation(
            phi=phi, tilt=tilt
        )

    def push_stroke(
        self,
        *,
        stroke: Stroke,
        base: np.ndarray,
        travel: Callable[[], None],
        rack_pose: np.ndarray,
        label: str,
    ) -> None:
        """Raise, travel, approach, lower onto the stroke start, push at fixed speed, lift."""
        backend, shelf, tool = self.backend, self.shelf, self.tool
        speed = SWEEP_SPEED_M_S
        rack = backend.support(name="rack_0")
        tool.require_held(label=f"{label}_start")
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ tool.wiper_pose()
        tool.top, tool.front = rack.top_z, rack.x_min
        body = np.eye(4)
        body[:3, :3] = self.stroke_rotation(stroke=stroke, tilt=backend.config.scraper_tilt_degrees)
        body[:3, 3] = stroke.start - tool.blade_edge(pose=body).mean(axis=0)
        target = body @ np.linalg.inv(shelf.in_hand)
        rotation, position = target[:3, :3], target[:3, 3]
        self.exit_surface(label=label)
        here = shelf.site_pose()
        carry_z = max(here[2, 3], position[2] + 0.06)
        tool.line(
            position=np.r_[here[:2, 3], carry_z],
            rotation=here[:3, :3],
            label=f"{label}_raise",
            holding=True,
        )
        travel()
        self.approach_surface(
            position=np.r_[position[:2], carry_z],
            rotation=rotation,
            base=base,
            label=label,
            holding=True,
        )
        hover_body = body.copy()
        hover_body[2, 3] += carry_z - position[2]
        self.move_tool(body=hover_body, label=f"{label}_level_measured_tool")
        for height in np.linspace(carry_z - position[2], 0, 7)[1:]:
            lower_body = body.copy()
            lower_body[2, 3] += height
            self.move_tool(body=lower_body, label=f"{label}_lower_measured_tool")
        arm = shelf.data.qpos[shelf.qa].copy()
        distance = 0.0
        for _ in range(int(stroke.length / (speed * 0.1)) * 4 + 100):
            live_rack = np.asarray(backend.core().get_object("rack_0").get_pose()[0])
            displacement = float(np.linalg.norm(live_rack - rack_pose))
            self.metrics["max_rack_displacement_m"] = max(
                self.metrics.get("max_rack_displacement_m", 0.0), displacement
            )
            if displacement > 0.015:
                raise SkillFailed("rack moved more than 15 mm; stop and reobserve")
            if not backend.held(name="wiper_0"):
                raise SkillFailed(f"scraper lost during {label}")
            actual = float((tool.blade_edge().mean(axis=0) - stroke.start) @ stroke.direction)
            if distance - actual < 0.015:
                distance = min(stroke.length, distance + speed * 0.1)
            desired_body = body.copy()
            desired_body[:3, 3] += stroke.point(distance=distance) - stroke.start
            waypoint = self.hand_for_tool(body=desired_body)
            shelf.base_target = base.copy()
            arm = tool.ik(
                position=waypoint[:3, 3], rotation=waypoint[:3, :3], holding=True, seed=arm
            )
            tool.step(arm=arm, label=f"{label}_push")
            if distance >= stroke.length and distance - actual < 0.01:
                break
        else:
            raise SkillFailed(f"{label} stalled; preserving the observed physical state")
        before_settle, _ = self.settle_stroke_end(
            stroke=stroke, body=body, base=base, arm=arm, label=label
        )
        edge = tool.blade_edge().mean(axis=0)
        self.metrics.setdefault("strokes", []).append({
            "label": label,
            "start": stroke.start.round(4).tolist(),
            "planned_end": stroke.end.round(4).tolist(),
            "blade_end": edge.round(4).tolist(),
            "blade_end_error_m": float(np.linalg.norm(edge - stroke.end)),
            "blade_end_error_before_settle_m": before_settle,
            "base": base.round(4).tolist(),
        })
        self.release_swept_food(rotation=body[:3, :3], retreat=-RELEASE_RETREAT * stroke.direction)

    def exit_surface(self, *, label: str) -> None:
        """Leave a blade still at a stroke's working height the way a stroke's own
        release would: off the surface, then back toward the base, before the raise.

        A GatherAndSweep that fails mid-push skips ``release_swept_food`` and leaves
        the blade at the stroke's end, flat on the rack with the arm at full reach
        from the side stance. Raising straight up from there is what every later
        call tried, and could not do (Della 16788008: 101/119 calls at 0 steps):
        with the blade on the wires, 1 cm waypoints ask ~0.065 rad of several joints
        at once, and the axis-clipped joint path the arm follows rolls one blade end
        4 mm into the rack, beyond the support allowance; with the arm stretched, a
        6 cm vertical raise is near-singular (0.7 rad of joint 4, 3.6 mm IK error).

        "Over the rack or the plate" is the whole area the strokes work in: the
        rectangle spanning the rack, the plate out to its outer rim, and the gap
        between them. Only the rack and the plate's well once counted, and a Sweep that
        failed at the plate left the blade resting on the rim 6 mm outside the well:
        every later stroke then raised straight up from the rim and refused at 0 steps
        (Della c6ed1a6 Curtis seed 0: 76 calls in a row).

        So when the blade's edge centre is over the rack or the plate and its lowest
        point is within ``EXIT_BAND`` of the rack top or the plate's sweep floor, it
        is first lifted to ``LIFT_OFF_HEIGHT`` above that floor in ``LIFT_OFF_STEP``
        waypoints (each joint then moves less than the 0.025 rad the path is checked
        at, so the executed path is the straight one), then drawn ``EXIT_RETREAT``
        horizontally toward the base, over the region the stroke has just swept,
        where the arm is no longer stretched. The hand keeps its rotation throughout.
        After a completed stroke ``release_swept_food`` has already lifted the blade
        6 cm, so this does nothing.
        """
        shelf, tool, backend = self.shelf, self.tool, self.backend
        floor = max(backend.support(name="rack_0").top_z, self.plate_clear_z())
        edge = tool.blade_edge()
        low = float(edge[:, 2].min())
        x, y = edge.mean(axis=0)[:2]
        work = self.stroke_geometry()
        reach = work["plate_radius"]
        y_min = min(work["rack_y_min"], work["plate_center_y"] - reach)
        y_max = max(work["rack_y_max"], work["plate_center_y"] + reach)
        over = work["plate_center_x"] - reach <= x <= work["rack_back"] and y_min <= y <= y_max
        if not over or low >= floor + EXIT_BAND:
            return
        here = shelf.site_pose()
        rise = max(0.0, floor + LIFT_OFF_HEIGHT - low)
        toward = shelf.data.qpos[shelf.ba][:2] - edge.mean(axis=0)[:2]
        toward /= np.linalg.norm(toward)
        lift = np.array([0.0, 0.0, rise])
        retreat = EXIT_RETREAT * np.r_[toward, 0.0]
        moves = [(lift, "lift_off", LIFT_OFF_STEP), (retreat, "exit_retreat", EXIT_STEP)]
        # From the rear stance the arm reaching the plate is nearly straight: lifting
        # there asks 0.3 rad of joints 3 and 5 per 2 mm, and IK fails within 2 cm,
        # while drawing the blade toward the base is well conditioned. So when the
        # lift cannot be planned where the blade is, the retreat goes first.
        if rise > 0 and not tool.line_plannable(
            position=here[:3, 3] + lift, rotation=here[:3, :3], holding=True, step=LIFT_OFF_STEP
        ):
            moves.reverse()
        for offset, name, step in moves:
            if not offset.any():
                continue
            here = shelf.site_pose()
            tool.line(
                position=here[:3, 3] + offset,
                rotation=here[:3, :3],
                label=f"{label}_{name}",
                holding=True,
                step=step,
            )

    def settle_stroke_end(
        self,
        *,
        stroke: Stroke,
        body: np.ndarray,
        base: np.ndarray,
        arm: np.ndarray,
        label: str,
    ) -> tuple[float, float]:
        """Hold the stroke's end until the measured blade edge is on it along travel and z.

        The push loop ends on progress along travel only, so a blade the food has held
        back or lifted stops short. This commands the end pose plus an accumulated
        correction of the measured error, for at most ``SETTLE_STEPS`` steps, and never
        commands the edge centre more than 2 mm below the planned end. The entire
        blade and joint path still need the collision guard: a tilted corner can
        dip lower. Unsafe corrections are backtracked toward the measured pose;
        when no checked correction is available, this optional settling ends.
        Error along the blade's own length is neither corrected nor waited
        for: dragging the blade sideways rearranged the food and made the next gather
        climb a dumpling, failing (0.10, 0.0, 0.08) on 7/7 real-arm layouts (2026-11-03).
        Returns the full edge error before and after, in metres; reaching the tolerance
        is not required.
        """
        tool, shelf = self.tool, self.shelf
        along = np.asarray(stroke.axis, dtype=float) / np.linalg.norm(stroke.axis)

        def correctable() -> np.ndarray:
            error = stroke.end - tool.blade_edge().mean(axis=0)
            return error - (error @ along) * along

        before = float(np.linalg.norm(stroke.end - tool.blade_edge().mean(axis=0)))
        offset = np.zeros(3)
        end_offset = stroke.end - stroke.start
        for _ in range(self.SETTLE_STEPS):
            error = correctable()
            if np.linalg.norm(error) < self.SETTLE_TOLERANCE:
                break
            proposed = np.clip(offset + error, -self.SETTLE_LIMIT, self.SETTLE_LIMIT)
            proposed[2] = max(proposed[2], -0.002)
            desired = body.copy()
            desired[:3, 3] += end_offset + proposed
            shelf.base_target = base.copy()
            correction, fraction = self.settling_correction(body=desired, label=label)
            if correction is None:
                break
            offset += fraction * (proposed - offset)
            arm = correction
            # Execution failures, lost grasps and exhausted budgets must propagate.
            tool.step(arm=arm, label=f"{label}_settle")
        return before, float(np.linalg.norm(stroke.end - tool.blade_edge().mean(axis=0)))

    def settling_correction(
        self, *, body: np.ndarray, label: str
    ) -> tuple[np.ndarray | None, float]:
        """Find a smaller checked correction; never execute a rejected joint path."""
        tool, shelf = self.tool, self.shelf
        current = tool.wiper_pose().copy()
        rotation = Slerp([0, 1], Rotation.from_matrix([current[:3, :3], body[:3, :3]]))
        reason = ""
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            candidate = np.eye(4)
            candidate[:3, :3] = rotation(fraction).as_matrix()
            candidate[:3, 3] = current[:3, 3] + fraction * (body[:3, 3] - current[:3, 3])
            waypoint = self.hand_for_tool(body=candidate)
            try:
                arm = tool.ik(
                    position=waypoint[:3, 3],
                    rotation=waypoint[:3, :3],
                    holding=True,
                    seed=shelf.data.qpos[shelf.qa].copy(),
                )
            except SkillFailed as exc:
                reason = str(exc)
                continue
            if fraction < 1:
                self.metrics.setdefault("settling_corrections", []).append({
                    "label": label,
                    "fraction": fraction,
                    "reason": reason,
                })
            return arm, fraction
        self.metrics.setdefault("settling_corrections", []).append({
            "label": label,
            "fraction": 0,
            "reason": reason,
        })
        return None, 0.0

    def release_swept_food(
        self, *, rotation: np.ndarray | None = None, retreat: np.ndarray | None = None
    ) -> None:
        """Clear vertically before settling; do not drag supported food toward the rack.

        With ``rotation`` (the stroke's planned, level blade), the blade is first turned
        back to it about its edge centre, 2 mm up and moved by ``retreat`` (back
        against the push, away from the food), then lifted. At a deep side-gather reach
        the arm holds the blade rolled about its length, and lifting it there drags its
        low end through the rack wires (2026-11-03, gathers stopping 3 cm from c).
        """
        tool = self.tool
        lifted_body = tool.wiper_pose().copy()
        if rotation is not None:
            edge = tool.blade_edge().mean(axis=0) + [0.0, 0.0, 0.002]
            if retreat is not None:
                edge = edge + retreat
            level = np.eye(4)
            level[:3, :3] = rotation
            level[:3, 3] = edge - tool.blade_edge(pose=level).mean(axis=0)
            self.move_tool(body=level, label="sweep_level")
            lifted_body = level.copy()
        lifted_body[2, 3] += 0.06
        self.move_tool(body=lifted_body, label="sweep_lift")
        self.shelf.hold(count=40, label="food_settle")

    def pick_food(self, *, name: str) -> None:
        RackFoodController(controllers=self).pick(name=name)

    def place_food(self, *, name: str, destination: str) -> None:
        RackFoodController(controllers=self).place(name=name, destination=destination)
