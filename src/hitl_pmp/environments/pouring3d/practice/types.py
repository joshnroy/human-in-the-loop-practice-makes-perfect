"""Observed state and typed cast for count-conditioned pouring."""

from typing import ClassVar

import numpy as np  # noqa: F401
from pydantic import BaseModel

from hitl_pmp.core.problem.environment.types import Object, State, Type


class FrozenPourState(State):
    abstract_atoms: frozenset[tuple[str, tuple[str, ...]]] = frozenset()


class PourObjects(BaseModel):
    count: int = 3
    pose_features: ClassVar[tuple[str, ...]] = (
        "x",
        "y",
        "z",
        "qw",
        "qx",
        "qy",
        "qz",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
    )
    robot_type: ClassVar[Type] = Type(name="pour_robot", feature_names=("x", "y", "yaw", "grip"))
    shrimp_type: ClassVar[Type] = Type(name="shrimp", feature_names=pose_features)
    pitcher_type: ClassVar[Type] = Type(name="pitcher", feature_names=pose_features)
    tray_type: ClassVar[Type] = Type(name="pour_tray", feature_names=pose_features)
    request_type: ClassVar[Type] = Type(name="pour_request", feature_names=("count",))
    scene_type: ClassVar[Type] = Type(
        name="pour_scene", feature_names=("seed", "steps_taken", "requested_count", "surface_mu")
    )
    robot: ClassVar[Object] = Object(name="robot", type=robot_type)
    pitcher: ClassVar[Object] = Object(name="pitcher", type=pitcher_type)
    tray: ClassVar[Object] = Object(name="tray", type=tray_type)
    scene: ClassVar[Object] = Object(name="scene", type=scene_type)

    def shrimp(self) -> tuple[Object, ...]:
        return tuple(Object(name=f"shrimp_{i}", type=self.shrimp_type) for i in range(self.count))

    def requests(self) -> tuple[Object, ...]:
        return tuple(
            Object(name=f"request_{i}", type=self.request_type) for i in range(1, self.count + 1)
        )

    def all(self) -> tuple[Object, ...]:
        return (self.robot, self.pitcher, self.tray, *self.shrimp(), *self.requests())

    @classmethod
    def types(cls) -> tuple[Type, ...]:
        return (cls.robot_type, cls.pitcher_type, cls.tray_type, cls.shrimp_type, cls.request_type)
