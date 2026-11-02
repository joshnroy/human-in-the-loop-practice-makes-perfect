"""Pure measured geometry and episode accounting; no config or simulator imports.

The backend must obtain a new ledger with ``charge_human`` BEFORE any human
scene writes. Commit that ledger even if the attempted intervention fails.
Returning cooked food to the rack never starts a new cooling interval.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FoodLocation = Literal["HoldingFood", "OnTray", "OnRack", "SpilledCounter", "SpilledFloor"]
HumanIntervention = Literal["human_batch_return", "human_tool_recovery"]
HUMAN_BATCH_RETURN = "human_batch_return"
HUMAN_TOOL_RECOVERY = "human_tool_recovery"


class FoodMeasurement(BaseModel):
    """World-space centre and measured axis-aligned half extents, in metres.

    ``footprint`` optionally carries points whose convex hull encloses the
    food's horizontal shadow; a round support tests those instead of the box.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    position: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    held: bool = False
    footprint: tuple[tuple[float, float], ...] | None = None

    def horizontal_points(self) -> tuple[tuple[float, float], ...]:
        if self.footprint:
            return self.footprint
        centre_x, centre_y, _ = self.position
        half_x, half_y, _ = self.half_extents
        return tuple(
            (centre_x + sign_x * half_x, centre_y + sign_y * half_y)
            for sign_x in (-1, 1)
            for sign_y in (-1, 1)
        )

    def rests_on(self, *, top_z: float, tolerance: float) -> bool:
        return abs(self.position[2] - self.half_extents[2] - top_z) <= tolerance

    @model_validator(mode="after")
    def _positive_extents(self) -> FoodMeasurement:
        if any(extent <= 0 for extent in self.half_extents):
            raise ValueError("food half extents must be positive")
        return self


class SupportRegion(BaseModel):
    """Measured horizontal support footprint and its top elevation, in metres."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    top_z: float

    @model_validator(mode="after")
    def _ordered_bounds(self) -> SupportRegion:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("support bounds must have positive area")
        return self

    def supports(self, *, food: FoodMeasurement, tolerance: float) -> bool:
        centre_x, centre_y, centre_z = food.position
        half_x, half_y, half_z = food.half_extents
        return (
            self.x_min <= centre_x - half_x
            and centre_x + half_x <= self.x_max
            and self.y_min <= centre_y - half_y
            and centre_y + half_y <= self.y_max
            and abs(centre_z - half_z - self.top_z) <= tolerance
        )

    def contains_xy(self, *, points: Iterable[Sequence[float]], margin: float = 0.0) -> bool:
        return all(
            self.x_min + margin <= x <= self.x_max - margin
            and self.y_min + margin <= y <= self.y_max - margin
            for x, y, *_ in points
        )


class CircularSupport(BaseModel):
    """Measured round support (a plate): live centre, radius and top elevation, in metres."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    center_x: float
    center_y: float
    radius: float = Field(gt=0)
    top_z: float

    @property
    def x_min(self) -> float:
        return self.center_x - self.radius

    @property
    def x_max(self) -> float:
        return self.center_x + self.radius

    @property
    def y_min(self) -> float:
        return self.center_y - self.radius

    @property
    def y_max(self) -> float:
        return self.center_y + self.radius

    def contains_xy(self, *, points: Iterable[Sequence[float]], margin: float = 0.0) -> bool:
        return all(
            math.hypot(x - self.center_x, y - self.center_y) <= self.radius - margin
            for x, y, *_ in points
        )

    def supports(self, *, food: FoodMeasurement, tolerance: float) -> bool:
        """The whole horizontal footprint lies inside the circle, resting on its top."""
        return self.contains_xy(points=food.horizontal_points()) and food.rests_on(
            top_z=self.top_z, tolerance=tolerance
        )


class RectangularSupport(BaseModel):
    """Measured oriented rectangle (a lipped platter's interior) and its floor elevation.

    ``half_x``/``half_y`` are the lip's inner half extents along the platter's own
    axes, yawed by ``yaw`` about world z. Containment is tested in that frame, so a
    yawed platter never scores points that only its axis-aligned box contains.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    center_x: float
    center_y: float
    yaw: float
    half_x: float = Field(gt=0)
    half_y: float = Field(gt=0)
    top_z: float

    def _aabb_half(self) -> tuple[float, float]:
        cos, sin = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        return self.half_x * cos + self.half_y * sin, self.half_x * sin + self.half_y * cos

    @property
    def x_min(self) -> float:
        return self.center_x - self._aabb_half()[0]

    @property
    def x_max(self) -> float:
        return self.center_x + self._aabb_half()[0]

    @property
    def y_min(self) -> float:
        return self.center_y - self._aabb_half()[1]

    @property
    def y_max(self) -> float:
        return self.center_y + self._aabb_half()[1]

    def contains_xy(self, *, points: Iterable[Sequence[float]], margin: float = 0.0) -> bool:
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        for x, y, *_ in points:
            dx, dy = x - self.center_x, y - self.center_y
            local_x, local_y = cos * dx + sin * dy, -sin * dx + cos * dy
            if abs(local_x) > self.half_x - margin or abs(local_y) > self.half_y - margin:
                return False
        return True

    def supports(self, *, food: FoodMeasurement, tolerance: float) -> bool:
        """The whole horizontal footprint lies inside the lip, resting on the floor."""
        return self.contains_xy(points=food.horizontal_points()) and food.rests_on(
            top_z=self.top_z, tolerance=tolerance
        )


ServingRegion = SupportRegion | CircularSupport | RectangularSupport


class _GeometryTolerance(BaseModel):
    value: float = Field(ge=0, allow_inf_nan=False)
    floor_z: float = Field(allow_inf_nan=False)


def classify_food(
    *,
    food: FoodMeasurement,
    rack: ServingRegion,
    tray: ServingRegion,
    counter: ServingRegion,
    support_contacts: frozenset[Literal["rack", "tray", "counter", "floor"]],
    floor_z: float = 0.0,
    support_tolerance: float = 0.005,
) -> FoodLocation | None:
    """Exclusive location: grasp > tray > rack > counter > floor.

    Geometry is supplied from the live scene, not nominal config coordinates.
    An unsupported proxy (including one in flight) has no location predicate;
    failure to reach a destination alone is not evidence of a floor spill.
    """
    tolerance = _GeometryTolerance(value=support_tolerance, floor_z=floor_z)
    if food.held:
        return "HoldingFood"
    candidates: tuple[tuple[ServingRegion, str, FoodLocation], ...] = (
        (tray, "tray", "OnTray"),
        (rack, "rack", "OnRack"),
        (counter, "counter", "SpilledCounter"),
    )
    for region, contact, location in candidates:
        if contact in support_contacts and region.supports(food=food, tolerance=tolerance.value):
            return location
    if (
        "floor" in support_contacts
        and abs(food.position[2] - food.half_extents[2] - tolerance.floor_z) <= tolerance.value
    ):
        return "SpilledFloor"
    return None


class RackServingLedger(BaseModel):
    """Immutable clock/cost state, constructed from scalar config values.

    ``elapsed_seconds`` is initialized from ``initial_elapsed_seconds`` once, at
    episode creation. Every physical action may advance it by measured duration.
    Neither human operation resets it; readiness depends on this clock alone.
    """

    model_config = ConfigDict(frozen=True)

    mode: Literal["training", "deployment"] = "training"
    cooling_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)
    elapsed_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    wait_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    human_interruption_cost: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    human_cost: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    human_interventions: int = Field(default=0, ge=0, strict=True)

    @property
    def ready(self) -> bool:
        return self.elapsed_seconds >= self.cooling_seconds

    def advance(self, *, seconds: float) -> RackServingLedger:
        duration = _Duration(seconds=seconds)
        return RackServingLedger(**{
            **self.model_dump(),
            "elapsed_seconds": self.elapsed_seconds + duration.seconds,
        })

    def wait(self) -> RackServingLedger:
        return self.advance(seconds=self.wait_seconds)

    def charge_human(self, *, intervention: HumanIntervention) -> RackServingLedger:
        """Authorize and charge an attempt before the caller touches the scene."""
        if self.mode != "training":
            raise PermissionError("human interventions are training-only")
        if intervention not in (HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY):
            raise ValueError(f"unknown human intervention: {intervention}")
        return RackServingLedger(**{
            **self.model_dump(),
            "human_cost": self.human_cost + self.human_interruption_cost,
            "human_interventions": self.human_interventions + 1,
        })


class _Duration(BaseModel):
    seconds: float = Field(ge=0, allow_inf_nan=False)
