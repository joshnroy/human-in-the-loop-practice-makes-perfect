"""Shared observed-state schema; feature flags are captured by the environment.

Positions are metres, quaternions xyzw, yaw radians and speeds SI units. Boolean
features use 0/1. ``reachable`` includes robot-accessible in-bin retrieval, not
just floor objects. ``hand_empty`` means the gripper is open and unoccupied.
``snapshot`` is an opaque simulator checkpoint, never read by predicates.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import model_validator

from hitl_pmp.core.problem.environment.types import Object, State, Type

from .domain import KitchenConfig

NEAR_BIN_POSITION = (0.65, 1.4, 0.0)


class KitchenState(State):
    snapshot: Any = None

    @model_validator(mode="after")
    def copy_observation(self) -> "KitchenState":
        self.data = {
            obj: np.array(features, dtype=float, copy=True) for obj, features in self.data.items()
        }
        return self


class KitchenObjects:
    """Stable symbolic identities, with item indices matching ``config.objects``."""

    robot_type: ClassVar[Type] = Type(
        name="robot",
        feature_names=(
            "x",
            "y",
            "yaw",
            "hand_empty",
            "training",
            "seed",
            "skills_taken",
            "gripper_open",
        ),
    )
    item_type: ClassVar[Type] = Type(
        name="item",
        feature_names=(
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "linear_speed",
            "angular_speed",
            "finger_contacts",
            "held",
            "settled",
            "reachable",
            "on_kitchen_floor",
            "blocked",
        ),
    )
    bin_type: ClassVar[Type] = Type(
        name="bin", feature_names=("x", "y", "z", "qx", "qy", "qz", "qw", "near", "at_target")
    )

    def __init__(self, *, config: KitchenConfig):
        self.robot = Object(name="robot", type=self.robot_type)
        self.bin = Object(name="bin", type=self.bin_type)
        self.items = tuple(Object(name=spec.name, type=self.item_type) for spec in config.objects)
        self.by_name = {obj.name: obj for obj in self.all_objects()}

    @classmethod
    def all_types(cls) -> tuple[Type, ...]:
        return cls.robot_type, cls.item_type, cls.bin_type

    def all_objects(self) -> tuple[Object, ...]:
        return self.robot, *self.items, self.bin
