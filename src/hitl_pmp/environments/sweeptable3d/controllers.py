"""One physical angle/speed stroke and contact-tolerant ball restoration."""

from collections.abc import Callable

import numpy as np
from pydantic import Field, PrivateAttr

from hitl_pmp.environments.sweepoffshelf3d.controllers import (
    ControllerFailure,
    SweepOffShelf3DControllers,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills, SkillFailed

from .backend import SweepTable3DBackend
from .recovery import TableKinderSkills
from .skills import SWEEP_ANGLE_BOUNDS, SWEEP_SPEED_BOUNDS
from .wiper import TableWiper


class SweepTable3DControllers(SweepOffShelf3DControllers):
    backend: SweepTable3DBackend
    _wiper: TableWiper | None = PrivateAttr(default=None)
    contact_metrics: dict[str, float | int] = Field(default_factory=dict)

    def on_scene_reset(self) -> None:
        self._skills = TableShelfSkills(env=self.backend.env(), tick_limit=self.step_limit)
        self._kinder = TableKinderSkills(shelf=self._skills, backend=self.backend)
        self._wiper = TableWiper(shelf=self._skills)
        self.backend.set_home_check(check=self._wiper.at_home)
        self.backend.grip = 0.0
        self.contact_metrics = {}

    def _run_table(self, *, skill: Callable[[], None]) -> None:
        shelf = self._skills
        if shelf is None:
            raise ControllerFailure("reset the scene before acting")
        shelf.ticks = 0
        try:
            skill()
        except SkillFailed as failure:
            raise ControllerFailure(str(failure)) from failure
        finally:
            self.steps = shelf.ticks
            self.backend.grip = shelf.grip

    def pick_wiper(self) -> None:
        assert self._wiper is not None
        self._run_table(skill=self._wiper.pick)

    def return_wiper(self) -> None:
        assert self._wiper is not None
        failed = self.backend.task_failed()
        if failed:
            self.backend.begin_robot_reset()
        try:
            self._run_table(skill=self._wiper.place)
        finally:
            if failed:
                self.backend.finish_robot_reset()

    def sweep_table(self, *, angle: float, speed: float) -> None:
        from .contact import ContactTableSweep

        if not np.isfinite([angle, speed]).all() or not (
            SWEEP_ANGLE_BOUNDS[0] <= angle <= SWEEP_ANGLE_BOUNDS[1]
            and SWEEP_SPEED_BOUNDS[0] <= speed <= SWEEP_SPEED_BOUNDS[1]
        ):
            raise SkillFailed("SweepTable parameters outside [angle_deg, speed_m_s] bounds")
        assert self._wiper is not None
        stroke = ContactTableSweep(wiper=self._wiper, center_y=self.backend.table_center_y())
        stroke.SPEED = speed
        completed = False
        self.backend.begin_sweep()
        try:
            self._run_table(skill=lambda: stroke.run(offset=0, yaw=angle))
            completed = True
        finally:
            self.contact_metrics = stroke.metrics
            self.backend.finish_sweep(completed=completed)

    def return_cube_to_table(self, *, cube: str) -> None:
        assert self._skills is not None and isinstance(self._kinder, TableKinderSkills)
        shelf, kinder = self._skills, self._kinder

        def reset_cube() -> None:
            self.backend.begin_robot_reset(restores_balls=True)
            try:
                with kinder.allow_ball_contact(cube=cube):
                    kinder.run(skill=kinder.pick_cube_from_floor, label="reset_pick", cube=cube)
                    xy = self.backend.robot_reset_xy(cube=cube)
                    shelf.place_cube_on_shelf(cube=cube, xy=xy)
            finally:
                self.backend.finish_robot_reset()

        self._run_table(skill=reset_cube)


class TableShelfSkills(ShelfSkills):
    def step(self, *, arm: np.ndarray, base: np.ndarray | None = None, label: str = "move") -> None:
        super().step(arm=arm, base=base, label=label)
        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if (
                pair & (self.robot_geoms | set(self.wiper_geoms))
                and pair & (self.cupboard_geoms | self.box_geoms)
                and -float(contact.dist) > 0.003
            ):
                raise SkillFailed(f"{label}: round-tool motion penetrated a fixture")


class TableSweep:
    """One straight stroke; no aiming, gathering, or object pose writes."""

    START_X = 1.60
    SPEED = 0.05
    TRACK_TOLERANCE = 0.025
    HEIGHT_BAND = 0.003

    def __init__(self, *, wiper: TableWiper, center_y: float) -> None:
        self.wiper = wiper
        self.shelf = wiper.shelf
        self.center_y = center_y

    @staticmethod
    def direction(*, yaw: float) -> np.ndarray:
        radians = np.deg2rad(yaw)
        return np.array([-np.cos(radians), np.sin(radians), 0.0])

    def stroke_length(self, *, yaw: float) -> float:
        direction = self.direction(yaw=yaw)
        half_length = self.wiper.model.geom_size[self.wiper.head_geom, 1]
        overhang = half_length * abs(direction[1]) + self.wiper.radius
        return float((self.START_X - self.shelf.front_x + overhang + 0.015) / -direction[0])

    def run(self, *, offset: float, yaw: float) -> None:
        if not np.isfinite([offset, yaw]).all() or not (
            offset == 0 and SWEEP_ANGLE_BOUNDS[0] <= yaw <= SWEEP_ANGLE_BOUNDS[1]
        ):
            raise SkillFailed("SweepTable parameters outside [offset_m, yaw_deg] bounds")
        wiper, shelf = self.wiper, self.shelf
        wiper.require_held(label="sweep_start")
        shelf.in_hand = np.linalg.inv(shelf.site_pose()) @ wiper.wiper_pose()
        start_y = self.center_y + offset
        body = np.eye(4)
        body[:3, :3] = wiper.sweep_body_rotation(yaw_deg=-yaw)
        center = np.array([self.START_X, start_y, wiper.top + wiper.radius + 0.001])
        body[:3, 3] = center - body[:3, :3] @ wiper.model.geom_pos[wiper.head_geom]
        target = body @ np.linalg.inv(shelf.in_hand)
        rotation = target[:3, :3]
        here = shelf.site_pose()
        carry_z = max(float(here[2, 3]), float(target[2, 3] + 0.12))
        wiper.line(
            position=np.array([here[0, 3], here[1, 3], carry_z]),
            rotation=here[:3, :3],
            label="sweep_raise",
            holding=True,
        )
        shelf.move_base(target=np.array([0.80, start_y, 0.0]), label="sweep_base")
        hover = target[:3, 3].copy()
        hover[2] = carry_z
        wiper.transit(position=hover, rotation=rotation, label="sweep_hover", holding=True)
        position = target[:3, 3].copy()
        wiper.line(position=position, rotation=rotation, label="sweep_lower", holding=True)
        edge = wiper.top + 0.001
        for _ in range(5):
            error = wiper.capsule_bottom() - edge
            if abs(error) <= 0.0005:
                break
            position[2] -= float(np.clip(error, -0.004, 0.004))
            wiper.line(position=position, rotation=rotation, label="sweep_seat", holding=True)
        if abs(wiper.capsule_bottom() - edge) > 0.003:
            raise SkillFailed("SweepTable could not establish blade height")
        self.pull(position=position, rotation=rotation, yaw=yaw)

    def pull(self, *, position: np.ndarray, rotation: np.ndarray, yaw: float) -> None:
        shelf, wiper = self.shelf, self.wiper
        direction = self.direction(yaw=yaw)
        length = self.stroke_length(yaw=yaw)
        base = shelf.data.qpos[shelf.ba].copy()
        distance = 0.0
        advance = self.SPEED * 0.1
        height_origin = float(position[2])
        height = height_origin
        arm = shelf.data.qpos[shelf.qa].copy()
        base_target = base.copy()
        completed = False
        for _ in range(int(3 * length / advance) + 50):
            waypoint = position + distance * direction
            waypoint[2] = height
            lag = float((waypoint - shelf.data.site_xpos[shelf.site]) @ direction)
            if lag < self.TRACK_TOLERANCE:
                if distance >= length:
                    completed = True
                    break
                distance = min(length, distance + advance)
            height = float(
                np.clip(
                    height - 0.3 * (wiper.capsule_bottom() - wiper.top - 0.001),
                    height_origin - self.HEIGHT_BAND,
                    height_origin + self.HEIGHT_BAND,
                )
            )
            waypoint = position + distance * direction
            waypoint[2] = height
            base_target = base + distance * direction
            shelf.base_target = base_target.copy()
            previous = arm
            arm, _ = shelf.ik(
                position=waypoint,
                rotation=rotation,
                base=base_target,
                holding=True,
                attempts=4,
                seed=previous,
            )
            jump = np.max(np.abs((arm - previous + np.pi) % (2 * np.pi) - np.pi))
            if not np.isfinite(arm).all() or jump > 0.6:
                raise SkillFailed("SweepTable IK lost its continuous branch")
            shelf.step(arm=arm, base=base_target, label="sweep_pull")
            wiper.require_held(label="sweep_pull")
        if not completed:
            raise SkillFailed(f"SweepTable stalled at {distance:.3f}/{length:.3f} m")
        shelf.hold(count=10, label="sweep_settle")
        wiper.line(
            position=shelf.data.site_xpos[shelf.site] + [0, 0, 0.10],
            rotation=rotation,
            label="sweep_lift",
            holding=True,
        )
        shelf.hold(count=50, label="sweep_settle")
