"""Kitchen configuration and object geometry: layout, objects and grasp geometry.

All dimensions are metres, masses kg, quaternions xyzw and times seconds.
Simulation assumptions are deliberately separate from measured hardware facts.
"""

from __future__ import annotations

from math import atan2, isfinite, pi
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.spatial.transform import Rotation

BOTTLE_CAP_HEIGHT = 0.018
BOTTLE_CAP_RADIUS_FACTOR = 0.55


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ObjectSpec(FrozenModel):
    name: str
    kind: Literal["cube", "bottle", "can", "packaging"]
    dimensions: tuple[float, float, float]
    simulation_mass: float = Field(gt=0)
    measured_mass: float | None = Field(default=None, gt=0)
    dimensions_verified: bool = False
    rigidity_verified: bool = False

    @model_validator(mode="after")
    def valid_geometry(self) -> ObjectSpec:
        if not self.name or min(self.dimensions) <= 0:
            raise ValueError("Object names and positive dimensions are required")
        if self.kind in ("bottle", "can") and self.dimensions[0] != self.dimensions[1]:
            raise ValueError("Cylindrical objects require equal transverse diameters")
        if self.kind == "bottle" and self.dimensions[2] <= BOTTLE_CAP_HEIGHT:
            raise ValueError("Bottle must be taller than its cap")
        return self


class Calibration(FrozenModel):
    """Per-object, per-context calibration; defaults are SIMULATION proposals only."""

    grasp_offset: float = Field(default=0.03, ge=0, le=0.1)
    minimum_lift: float = Field(default=0.07, gt=0)
    holding_duration: float = Field(default=0.15, gt=0)
    maximum_linear_speed: float = Field(default=0.025, gt=0)
    maximum_angular_speed: float = Field(default=0.15, gt=0)
    settle_duration: float = Field(default=0.6, gt=0)
    settle_timeout: float = Field(default=15.0, gt=0)
    toss_standoff: float = Field(default=1.5, gt=0)
    toss_azimuth: float = Field(default=0, ge=-pi, le=pi)
    toss_speed_deg_s: float = Field(default=130.0, gt=0)
    release_ms: float = Field(default=720.0, ge=0, le=1000)
    hardware_validated: bool = False

    @model_validator(mode="after")
    def valid_settling_window(self) -> Calibration:
        if self.settle_timeout <= self.settle_duration:
            raise ValueError("Settling timeout must exceed the confirmation window")
        return self


class KitchenConfig(FrozenModel):
    variant: Literal["kitchen-recycling-v1"] = "kitchen-recycling-v1"
    scatter_objects: bool = True
    mat_product: Literal["existing Sprout mats; SKU unverified"] = (
        "existing Sprout mats; SKU unverified"
    )
    mat_bounds: tuple[float, float, float, float] = (1.15, -2.5, 3.6, 2.5)
    kitchen_bounds: tuple[float, float, float, float] = (-1.5, -2.5, 1.15, 2.5)
    layout_measured: bool = False
    crossing_difficulty_demonstrated: bool = False
    robot_radius: float = Field(default=0.40, gt=0)
    direct_placement_reach: float = Field(default=0.85, gt=0)
    bin_outside: tuple[float, float, float] = (0.381, 0.32385, 0.1651)
    bin_position: tuple[float, float, float] = (2.05, 0.0, 0.0)
    simulation_wall: float = Field(default=0.006, gt=0)
    simulation_bin_mass: float = Field(default=0.4, gt=0)
    measured_bin_opening: tuple[float, float] | None = None
    measured_bin_rim: float | None = Field(default=None, gt=0)
    measured_bin_mass: float | None = Field(default=None, gt=0)
    interruption_cost: float = Field(default=10.0, gt=0)
    relocation_cost: float = Field(default=2.0, gt=0)
    restoration_cost: float = Field(default=2.0, gt=0)
    near_distance: float = Field(default=0.5, gt=0)
    objects: tuple[ObjectSpec, ...] = (
        ObjectSpec(
            name="cube_control", kind="cube", dimensions=(0.05, 0.05, 0.05), simulation_mass=0.1
        ),
        ObjectSpec(
            name="gatorade_20oz",
            kind="bottle",
            dimensions=(0.073, 0.073, 0.195),
            simulation_mass=0.035,
        ),
        ObjectSpec(
            name="food_can", kind="can", dimensions=(0.07, 0.07, 0.10), simulation_mass=0.04
        ),
    )
    required_objects: tuple[str, ...] = ("gatorade_20oz", "food_can")
    calibrations: dict[str, Calibration] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_layout(self) -> KitchenConfig:
        for bounds in (self.kitchen_bounds, self.mat_bounds):
            if not bounds[0] < bounds[2] or not bounds[1] < bounds[3]:
                raise ValueError("Zone bounds must have positive area")
        if self.kitchen_bounds[2] > self.mat_bounds[0]:
            raise ValueError("Kitchen must be separate from the restricted mat zone")
        if min(self.bin_outside) <= 2 * self.simulation_wall:
            raise ValueError("Bin wall must leave a positive interior")
        nearest_robot = self.kitchen_bounds[2] - self.robot_radius
        nearest_bin = self.bin_position[0] - self.bin_outside[0] / 2
        if nearest_bin - nearest_robot <= self.direct_placement_reach:
            raise ValueError("Deployment bin must be beyond direct placement reach")
        if not inside_rect(
            position=self.bin_position[:2], bounds=self.mat_bounds, dimensions=self.bin_outside[:2]
        ):
            raise ValueError("Deployment bin must fit in the living-room zone")
        names = [spec.name for spec in self.objects]
        if len(set(names)) != len(names) or not self.required_objects:
            raise ValueError("Unique object identities and a nonempty goal are required")
        if not set(self.required_objects) <= set(names):
            raise ValueError("Every goal object must exist")
        allowed = {f"{name}/{context}" for name in names for context in ("near", "far")}
        if not set(self.calibrations) <= allowed:
            raise ValueError("Calibration keys must be explicit object/near or object/far")
        return self

    def object(self, *, name: str) -> ObjectSpec:
        return next(spec for spec in self.objects if spec.name == name)

    def calibration(self, *, name: str, context: str) -> Calibration:
        self.object(name=name)
        if context not in ("near", "far"):
            raise ValueError("Unknown practice context")
        return self.calibrations.get(f"{name}/{context}", Calibration())

    def hardware_blockers(self) -> list[str]:
        missing = []
        if not self.layout_measured:
            missing.append("Sprout SKU/layout and robot swept-footprint/reach survey")
        if None in (self.measured_bin_opening, self.measured_bin_rim, self.measured_bin_mass):
            missing.append("Bin interior opening, rim geometry and empty mass")
        for spec in self.objects:
            if not spec.dimensions_verified or spec.measured_mass is None:
                missing.append(f"{spec.name}: actual dimensions and empty mass")
            if not spec.rigidity_verified:
                missing.append(f"{spec.name}: rigidity/friction and safe grasp pressure")
            for context in ("near", "far"):
                if not self.calibration(name=spec.name, context=context).hardware_validated:
                    missing.append(f"{spec.name}/{context}: grasp/release/scoring validation")
        return missing


def inside_rect(*, position: Any, bounds: Any, dimensions: Any = (0.0, 0.0)) -> bool:
    return bool(
        all(isfinite(float(value)) for value in (*position, *bounds, *dimensions))
        and bounds[0] <= position[0] - dimensions[0] / 2
        and position[0] + dimensions[0] / 2 <= bounds[2]
        and bounds[1] <= position[1] - dimensions[1] / 2
        and position[1] + dimensions[1] / 2 <= bounds[3]
    )


class ObjectObservation(FrozenModel):
    name: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (0, 0, 0, 1)
    linear_speed: float = Field(default=0, ge=0)
    angular_speed: float = Field(default=0, ge=0)
    finger_contacts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def unit_rotation(self) -> ObjectObservation:
        if not np.isclose(np.linalg.norm(self.quaternion), 1.0, atol=1e-5):
            raise ValueError("Observation quaternion must be normalized")
        return self


def world_extent(*, spec: ObjectSpec, observation: ObjectObservation) -> np.ndarray:
    lower, upper = projected_bounds(spec=spec, rotation=Rotation.from_quat(observation.quaternion))
    return 2 * np.maximum(np.abs(lower), np.abs(upper))


def projected_bounds(*, spec: ObjectSpec, rotation: Rotation) -> tuple[Any, Any]:
    """Support against coordinate planes, including cylindrical rather than box corners."""
    if spec.kind in ("cube", "packaging"):
        half = np.abs(rotation.as_matrix()) @ np.array(spec.dimensions) / 2
        return -half, half
    radius = spec.dimensions[0] / 2
    half_height = spec.dimensions[2] / 2
    axis = rotation.apply([0, 0, 1])
    radial = np.sqrt(np.maximum(0, 1 - axis**2))
    parts: list[tuple[float, float, float]] = [(0, half_height, radius)]
    if spec.kind == "bottle":
        parts = [
            (-BOTTLE_CAP_HEIGHT / 2, half_height - BOTTLE_CAP_HEIGHT / 2, radius),
            (
                half_height - BOTTLE_CAP_HEIGHT / 2,
                BOTTLE_CAP_HEIGHT / 2,
                radius * BOTTLE_CAP_RADIUS_FACTOR,
            ),
        ]
    lower, upper = [], []
    for offset, length, width in parts:
        extent = length * np.abs(axis) + width * radial
        lower.append(offset * axis - extent)
        upper.append(offset * axis + extent)
    return np.min(lower, axis=0), np.max(upper, axis=0)


def classify_zone(*, config: KitchenConfig, observation: ObjectObservation) -> str:
    extent = world_extent(spec=config.object(name=observation.name), observation=observation)
    for name, bounds in (("kitchen", config.kitchen_bounds), ("living_room", config.mat_bounds)):
        if inside_rect(position=observation.position[:2], bounds=bounds, dimensions=extent[:2]):
            return name
    return "boundary_or_outside"


def grasp_yaws(*, spec: ObjectSpec, observation: ObjectObservation) -> tuple[float, ...]:
    """Cylinder side grasps follow its observed axis, not an upright-cube assumption."""
    rotation = Rotation.from_quat(observation.quaternion)
    axis = rotation.apply([0, 0, 1])
    if spec.kind in ("bottle", "can") and abs(axis[2]) < 0.7:
        yaw = atan2(axis[1], axis[0])
        return yaw, yaw + pi
    heading = rotation.apply([1, 0, 0])
    yaw = atan2(heading[1], heading[0])
    return tuple(yaw + offset * pi / 2 for offset in range(4))
