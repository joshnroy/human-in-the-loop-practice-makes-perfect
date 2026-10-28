"""Physical shrimp recovery and observable ticks around the unchanged pour law."""

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.pouring3d.feedback_controller import FeedbackPourController
from hitl_pmp.environments.pouring3d.shrimp_native import CookedShrimp


class PracticePourController(FeedbackPourController):
    before_step: Callable[[], None] | None = None
    after_step: Callable[[], None] | None = None

    @classmethod
    def create(cls, *, env: Any) -> "PracticePourController":
        return cast(PracticePourController, super().create(env=env))

    def motor_step(self, *, target_q: np.ndarray, grip: float) -> None:
        if self.before_step is not None:
            self.before_step()
        before = self.steps
        try:
            super().motor_step(target_q=target_q, grip=grip)
            if self.held is not None and self.held in self.scene.cube_names:
                self.lost_contacts = (
                    0 if len(self.finger_contacts(name=self.held)) == 2 else self.lost_contacts + 1
                )
                if self.lost_contacts > 3:
                    self.held = None
                    raise ValueError("Physical shrimp grasp lost")
        finally:
            if self.steps > before and self.after_step is not None:
                self.after_step()

    def ready(self) -> bool:
        return (
            self.held is None
            and self.scene.cube_locations() == ["pitcher"] * len(self.scene.cube_names)
            and self.scene.cubes_settled()
            and self.vessel_home(name="pitcher")
            and self.vessel_home(name="tray")
        )

    def pick_shrimp(self, *, index: int, location: str) -> None:
        name = self.scene.cube_names[index]
        if self.held is not None or self.scene.cube_locations()[index] != location:
            raise ValueError("Shrimp pick requires an empty hand and the named source")
        position, rotation = self.body_pose(name=name)
        # Enclose the entire curl before closing. Preclosing around one segment
        # brushes its neighbour and pushes the low-friction piece out of the grasp.
        endpoints = np.array([p for a, b, _ in CookedShrimp.segments() for p in (a, b)])
        midpoint = (endpoints.min(axis=0) + endpoints.max(axis=0)) / 2
        point = position + rotation.apply(midpoint)
        across = rotation.apply([0, 1, 0])
        yaw = np.arctan2(across[1], across[0])
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if location == "tray":
            tray, orientation = self.body_pose(name="tray")
            geometry = self.scene.get_object("tray")
            clearance = np.array([geometry.length, geometry.width]) / 2 - geometry.wall_thickness
            clearance -= np.abs(orientation.inv().apply(point - tray)[:2])

            def margin(angle: float) -> float:  # noqa: PLR0917
                footprint = np.array([
                    0.06 * abs(np.cos(angle)) + 0.02 * abs(np.sin(angle)),
                    0.06 * abs(np.sin(angle)) + 0.02 * abs(np.cos(angle)),
                ])
                return float(np.min(clearance - footprint))

            if margin(yaw) < 0.005:
                yaw = max((yaw, yaw + np.pi / 2, 0.0, np.pi / 2), key=margin)
        down = Rotation.from_euler("z", yaw) * Rotation.from_euler("y", np.pi)
        self.move_base(
            target=np.array([point[0] - 0.65, point[1], 0]),
            phase="position for shrimp pick",
            seconds=5,
        )
        self.move(
            position=point + [0, 0, 0.25],
            rotation=down,
            seconds=5,
            grip=0.4,
            phase="approach shrimp",
        )
        self.move(
            position=point + [0, 0, 0.039],
            rotation=down,
            seconds=4,
            grip=0.4,
            phase="descend to shrimp body",
        )
        self.pause(phase="settle shrimp approach", seconds=1, grip=0.4)
        self.phase = "pinch shrimp"
        for grip in np.linspace(0.4, 1, round(3 * self.frequency) + 1)[1:]:
            self.motor_step(target_q=self.last_q, grip=float(grip))
        self.pause(phase="retain shrimp", seconds=1, grip=1)
        if len(self.finger_contacts(name=name)) != 2:
            raise ValueError("Shrimp grasp lacks both finger contacts")
        self.held, self.lost_contacts = name, 0
        self.move(
            position=point + [0, 0, 0.28], rotation=down, seconds=4, grip=1, phase="lift shrimp"
        )
        if self.body_pose(name=name)[0][2] < position[2] + 0.20:
            self.held = None
            raise ValueError("Shrimp not retained through lift")

    def drop_shrimp(self, *, index: int) -> None:
        name = self.scene.cube_names[index]
        if self.held != name or not self.vessel_home(name="pitcher"):
            raise ValueError("Drop requires held shrimp and home pitcher")
        pitcher, _ = self.body_pose(name="pitcher")
        _, down = self.hand_pose()
        self.move_base(
            target=np.array([pitcher[0] - 0.65, pitcher[1], 0]),
            phase="position for shrimp drop",
            seconds=5,
        )
        self.move(
            position=pitcher + [0, 0, 0.35],
            rotation=down,
            seconds=5,
            grip=1,
            phase="transport shrimp over pitcher",
        )
        if len(self.finger_contacts(name=name)) != 2:
            self.held = None
            raise ValueError("Shrimp lost before release")
        self.held = None
        self.pause(phase="release shrimp into pitcher", seconds=3, grip=0)
        if not self.scene.cube_in_vessel(name, "pitcher"):
            raise ValueError("Dropped shrimp did not land inside pitcher")
