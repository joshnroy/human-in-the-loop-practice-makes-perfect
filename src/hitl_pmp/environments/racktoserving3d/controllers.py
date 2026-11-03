"""Physical rack transfer and individual reset controllers, without object pose writes."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed

from .backend import RackServingBackend
from .food import RackFoodController
from .servo import SweepServo
from .skills import SWEEP_BOUNDS, SWEEP_SPEED_M_S
from .stroke import plan_stroke
from .tool import RackScraper, RackShelfSkills


class RackControllers:
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

    def sweep(self, *, dy: float, phi: float) -> None:
        """Straight push toward the plate at a fixed speed; (dy, phi) choose the blade."""
        (dy_low, dy_high), (phi_low, phi_high) = SWEEP_BOUNDS
        if not np.isfinite([dy, phi]).all() or not (
            dy_low <= dy <= dy_high and phi_low <= phi <= phi_high
        ):
            raise SkillFailed(
                f"SweepRack requires dy [{dy_low},{dy_high}] m and phi [{phi_low},{phi_high}] deg"
            )
        speed = SWEEP_SPEED_M_S
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
        rack_pose = np.asarray(backend.core().get_object("rack_0").get_pose()[0]).copy()
        stroke = plan_stroke(
            food_corners=np.stack([backend.corners(name=name) for name in names]),
            rack_back=rack.x_max,
            rack_front=rack.x_min,
            rack_center_y=(rack.y_min + rack.y_max) / 2,
            rack_top=rack.top_z,
            dy=dy,
            phi_deg=phi,
            blade_width=2 * shelf.model.geom_size[tool.head_geom, 0],
            tray_back=tray.x_max,
            release_clearance=0.01
            + max(
                backend.config.dumpling_depth,
                backend.config.dumpling_width,
                backend.config.dumpling_height,
            ),
        )
        self.metrics = {
            "measured_gap_m": rack.x_min - tray.x_max,
            "rack_top_m": rack.top_z,
            "tray_top_m": tray.top_z,
            "start_edge": stroke.start.tolist(),
            "end_edge": stroke.end.tolist(),
            "max_rack_displacement_m": 0.0,
            "dy_m": dy,
            "phi_deg": phi,
            "speed_m_s": speed,
        }
        with self.feedback():
            tool.require_held(label="sweep_start")
            shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ tool.wiper_pose()
            tool.top, tool.front = rack.top_z, rack.x_min
            body = np.eye(4)
            body[:3, :3] = self.push_rotation(phi=phi, tilt=backend.config.scraper_tilt_degrees)
            body[:3, 3] = stroke.start - tool.blade_edge(pose=body).mean(axis=0)
            target = body @ np.linalg.inv(shelf.in_hand)
            rotation, position = target[:3, :3], target[:3, 3]
            here = shelf.site_pose()
            carry_z = max(here[2, 3], position[2] + 0.06)
            tool.line(
                position=np.r_[here[:2, 3], carry_z],
                rotation=here[:3, :3],
                label="sweep_raise",
                holding=True,
            )
            self.around_counter(to_rear=True)
            base = np.array([
                backend.config.counter_x_max + 0.35,
                stroke.start[1] - 0.15,
                np.pi - 0.30,
            ])
            self.approach_surface(
                position=np.r_[position[:2], carry_z],
                rotation=rotation,
                base=base,
                label="sweep",
                holding=True,
            )
            hover_body = body.copy()
            hover_body[2, 3] += carry_z - position[2]
            self.move_tool(body=hover_body, label="sweep_level_measured_tool")
            for height in np.linspace(carry_z - position[2], 0, 7)[1:]:
                lower_body = body.copy()
                lower_body[2, 3] += height
                self.move_tool(body=lower_body, label="sweep_lower_measured_tool")
            arm = shelf.data.qpos[shelf.qa].copy()
            distance = 0.0
            for _ in range(int(stroke.length / (speed * 0.1)) * 4 + 100):
                live_rack = np.asarray(backend.core().get_object("rack_0").get_pose()[0])
                displacement = float(np.linalg.norm(live_rack - rack_pose))
                self.metrics["max_rack_displacement_m"] = max(
                    self.metrics["max_rack_displacement_m"], displacement
                )
                if displacement > 0.015:
                    raise SkillFailed("rack moved more than 15 mm; stop and reobserve")
                if not backend.held(name="wiper_0"):
                    raise SkillFailed("scraper lost during sweep")
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
                tool.step(arm=arm, label="sweep_pull")
                if distance >= stroke.length and distance - actual < 0.01:
                    break
            else:
                raise SkillFailed("sweep stalled; preserving the observed physical state")
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
