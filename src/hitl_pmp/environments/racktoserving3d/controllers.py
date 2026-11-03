"""Physical rack transfer and individual reset controllers, without object pose writes."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .backend import RackServingBackend
from .config import RackServingConfig
from .food import RackFoodController
from .servo import SweepServo
from .skills import GATHER_AND_SWEEP_BOUNDS, SWEEP_SPEED_M_S
from .stroke import GatherAndSweepPlanner, Stroke
from .tool import RackScraper, RackShelfSkills


class GatherBasePlacement(BaseModel):
    """Where the robot stands for each gather stroke, and how it gets there and back.

    PROVISIONAL, and the only place gather base poses live. Chosen by a static reach
    probe (IK plus scratch collision checks along the stroke, 2026-11-03): from the
    counter's rear none of 147 base poses tried reached the first gather waypoint,
    while a stance beside the counter on the stroke's own side did. That probe's
    executed strokes still failed a tool-clearance check near y = c, so these numbers
    await the dedicated reach probe and are not a validated placement.
    """

    model_config = ConfigDict(frozen=True)
    side_offset: float = 0.35
    lane_clearance: float = 0.80
    front_clearance: float = 0.40
    heading_offset: float = -0.30

    def base(self, *, side: int, config: RackServingConfig) -> np.ndarray:
        return np.array([
            config.rack_center_x,
            side * (config.counter_width / 2 + self.side_offset),
            -side * np.pi / 2 + self.heading_offset,
        ])

    def lane(self, *, side: int, config: RackServingConfig) -> float:
        return side * (config.counter_width / 2 + self.lane_clearance)

    def route(self, *, side: int, config: RackServingConfig, start: np.ndarray) -> list[np.ndarray]:
        """Out along the start's own lane, round the front, down this side's lane, in."""
        base = self.base(side=side, config=config)
        lane = self.lane(side=side, config=config)
        front_x = config.counter_x_min - self.front_clearance
        route: list[np.ndarray] = []
        if abs(start[1]) > config.counter_width / 2:
            own = self.lane(side=int(np.sign(start[1])), config=config)
            route += [np.array([start[0], own, start[2]]), np.array([front_x, own, start[2]])]
        route += [
            np.array([front_x, lane, start[2]]),
            np.array([front_x, lane, base[2]]),
            np.array([base[0], lane, base[2]]),
            base,
        ]
        return route

    def to_rear(self, *, side: int, config: RackServingConfig) -> list[np.ndarray]:
        """From a side stance, down that lane to the rear, facing the counter."""
        lane = self.lane(side=side, config=config)
        rear_x = config.counter_x_max + 0.60
        heading = self.base(side=side, config=config)[2]
        return [
            np.array([config.rack_center_x, lane, heading]),
            np.array([rear_x, lane, heading]),
            np.array([rear_x, lane, np.pi]),
            np.array([rear_x, config.rack_center_y, np.pi]),
        ]


class RackControllers:
    gather_placement: ClassVar[GatherBasePlacement] = GatherBasePlacement()

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
        with self.feedback():
            if self.shelf.data.qpos[self.shelf.ba[0]] > self.backend.config.counter_center_x:
                here = self.shelf.site_pose()
                self.tool.line(
                    position=here[:3, 3] + [0.10, 0, 0.06],
                    rotation=here[:3, :3],
                    label="return_clear_rack",
                    holding=True,
                )
                self.around_counter(to_rear=False)
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
        raise SkillFailed(f"{label}: measured scraper pose failed to converge")

    def wait_until_ready(self) -> None:
        with self.feedback():
            self.backend.sync_clock()
            remaining = max(
                0, self.backend.ledger.cooling_seconds - self.backend.ledger.elapsed_seconds
            )
            for _ in range(int(np.ceil(remaining / 0.1)) + (remaining > 0)):
                self.shelf.hold(count=1, label="wait_elapsed_time_not_temperature")
                self.backend.sync_clock()
                if self.backend.ledger.ready:
                    break

    def gather_and_sweep(self, *, delta: float, d: float, phi: float) -> None:
        """Gather from +y then -y toward y = c, then push the gathered line onto the plate."""
        bounds = GATHER_AND_SWEEP_BOUNDS
        values = (delta, d, phi)
        if not np.isfinite(values).all() or any(
            not low <= value <= high for value, (low, high) in zip(values, bounds, strict=True)
        ):
            (delta_low, delta_high), (d_low, d_high), (phi_low, phi_high) = bounds
            raise SkillFailed(
                f"GatherAndSweep requires delta [{delta_low},{delta_high}] m,"
                f" d [{d_low},{d_high}] m and phi [{phi_low},{phi_high}] deg"
            )
        self.backend.sync_clock()
        if not self.backend.ledger.ready:
            raise SkillFailed("food is not elapsed-time ready; wait first")
        backend, shelf, tool = self.backend, self.shelf, self.tool
        names = [name for name in backend.cube_names if backend.location(name=name) == "OnRack"]
        if not names:
            raise SkillFailed("no food is physically on the rack")
        rack, tray = backend.support(name="rack_0"), backend.support(name="box_0")
        if rack.top_z < tray.top_z or tray.x_max > rack.x_min:
            raise SkillFailed("live rack/tray geometry no longer provides a downward open exit")
        geometry = self.stroke_geometry()
        gather = GatherAndSweepPlanner.gather(
            food_corners=np.stack([backend.corners(name=name) for name in names]),
            delta=delta,
            **geometry,
        )
        self.metrics = {
            "measured_gap_m": rack.x_min - tray.x_max,
            "rack_top_m": rack.top_z,
            "tray_top_m": tray.top_z,
            "gather_edges": [[stroke.start.tolist(), stroke.end.tolist()] for stroke in gather],
            "max_rack_displacement_m": 0.0,
            "delta_m": delta,
            "d_m": d,
            "phi_deg": phi,
            "speed_m_s": SWEEP_SPEED_M_S,
        }
        rack_pose = np.asarray(backend.core().get_object("rack_0").get_pose()[0]).copy()
        placement = self.gather_placement
        with self.feedback():
            for side, stroke in zip((1, -1), gather, strict=True):

                def to_side(*, side: int = side) -> None:
                    route = placement.route(
                        side=side, config=backend.config, start=shelf.data.qpos[shelf.ba].copy()
                    )
                    for index, waypoint in enumerate(route[:-1]):
                        tool.move_base(target=waypoint, label=f"gather_route_{index}")

                self.push_stroke(
                    stroke=stroke,
                    base=placement.base(side=side, config=backend.config),
                    travel=to_side,
                    rack_pose=rack_pose,
                    label=f"gather_{'plus' if side > 0 else 'minus'}_y",
                )
            sweep = GatherAndSweepPlanner.sweep(
                food_corners=np.stack([backend.corners(name=name) for name in backend.cube_names]),
                d=d,
                phi_deg=phi,
                **geometry,
            )
            self.metrics["sweep_edges"] = [sweep.start.tolist(), sweep.end.tolist()]

            def to_rear() -> None:
                y = float(shelf.data.qpos[shelf.ba[1]])
                if abs(y) > backend.config.counter_width / 2:
                    side = 1 if y > 0 else -1
                    for index, waypoint in enumerate(
                        placement.to_rear(side=side, config=backend.config)
                    ):
                        tool.move_base(target=waypoint, label=f"sweep_route_{index}")
                elif shelf.data.qpos[shelf.ba[0]] < backend.config.counter_x_max:
                    self.around_counter(to_rear=True)

            self.push_stroke(
                stroke=sweep,
                base=np.array([
                    backend.config.counter_x_max + 0.35,
                    sweep.start[1] - 0.15,
                    np.pi - 0.30,
                ]),
                travel=to_rear,
                rack_pose=rack_pose,
                label="sweep",
            )

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
            desired_body[:3, 3] += distance * stroke.direction
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
        self.release_swept_food()

    def release_swept_food(self) -> None:
        """Clear vertically before settling; do not drag supported food toward the rack."""
        lifted_body = self.tool.wiper_pose().copy()
        lifted_body[2, 3] += 0.06
        self.move_tool(body=lifted_body, label="sweep_lift")
        self.shelf.hold(count=40, label="food_settle")

    def pick_food(self, *, name: str) -> None:
        RackFoodController(controllers=self).pick(name=name)

    def place_food(self, *, name: str, destination: str) -> None:
        RackFoodController(controllers=self).place(name=name, destination=destination)
