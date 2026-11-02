"""Pure measured geometry and episode accounting; no config or simulator imports.

The backend must obtain a new ledger with ``charge_human`` BEFORE any human
scene writes. Commit that ledger even if the attempted intervention fails.
Returning cooked food to the rack never starts a new cooling interval.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FoodLocation = Literal["HoldingFood", "OnTray", "OnRack", "SpilledCounter", "SpilledFloor"]
HumanIntervention = Literal["human_batch_return", "human_tool_recovery"]
HUMAN_BATCH_RETURN = "human_batch_return"
HUMAN_TOOL_RECOVERY = "human_tool_recovery"


class FoodMeasurement(BaseModel):
    """World-space centre and measured axis-aligned half extents, in metres."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    position: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    held: bool = False

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


class _GeometryTolerance(BaseModel):
    value: float = Field(ge=0, allow_inf_nan=False)
    floor_z: float = Field(allow_inf_nan=False)


def classify_food(
    *,
    food: FoodMeasurement,
    rack: SupportRegion,
    tray: SupportRegion,
    counter: SupportRegion,
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
    candidates: tuple[tuple[SupportRegion, str, FoodLocation], ...] = (
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
