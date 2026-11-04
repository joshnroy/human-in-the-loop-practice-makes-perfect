"""Metric layout and explicitly assumed hardware for rack-to-serving.

The specified rack footprint is 0.375 m wide (world y) by 0.255 m deep
(world x). Rack height and mass are UNVERIFIED configurable assumptions, not
manufacturer specifications. All other hardware dimensions, masses, friction,
and cooling times below are modeling assumptions requiring physical calibration.
The dumpling is a rigid cooked-food proxy, not a deformable or thermal model.
``concept`` denotes an invented perforated platform, not the specified wire rack.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0)]
Nonnegative = Annotated[float, Field(ge=0)]
Friction = tuple[Nonnegative, Nonnegative, Nonnegative]
INCH = 0.0254


class RackServingConfig(BaseModel):
    """Frozen SI-unit configuration; front/exit is the negative world-x side.

    Rack and serving-object poses use bottom-center body origins. Dumpling
    poses use their geometric centers. The serving object (``box_0``) is, by
    default, a model of a Corelle Winter Frost White 10-1/4 in dinner plate (a
    flat well, a low rim rising to 3/4 in at the edge, and a foot ring); or a
    flat round plate, the original rectangular slate, or a wide platter with a
    low continuous lip, selected by ``serving_target``; its position is derived
    to preserve ``gap`` in front of the rack's exit edge, centred on the rack's
    centre line. Neither the rack nor the serving object is fixed or assumed
    ungraspable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    surface_kind: Literal["wire", "concept"] = "wire"
    mode: Literal["training", "deployment"] = "training"
    counter_height: Positive = 0.90
    rack_height: Positive = Field(default=0.025, description="UNVERIFIED height assumption")
    rack_mass: Positive = Field(default=0.5, description="UNVERIFIED mass assumption")
    tray_thickness: Positive = 0.012
    gap: Nonnegative = 0.010
    cooling_seconds: Nonnegative = 300
    initial_elapsed_seconds: Nonnegative = 0
    wait_seconds: Positive = 30
    human_interruption_cost: Positive = 5

    rack_width: Positive = 0.375
    rack_depth: Positive = 0.255
    rack_center_x: float = 1.5
    rack_center_y: float = 0.0
    rack_wire_radius: Positive = 0.0015
    rack_wire_spacing: Positive = 0.015
    rack_leg_width: Positive = 0.012
    rack_friction: Friction = (0.6, 0.005, 0.003)
    support_contact_dimension: Literal[3, 4, 6] = 6
    concept_web_width: Positive = 0.008
    concept_pitch: Positive = 0.025
    concept_thickness: Positive = 0.003

    serving_target: Literal["corelle_plate", "round_plate", "slate", "lipped_platter"] = (
        "corelle_plate"
    )
    plate_radius: Positive = 0.13
    tray_width: Positive = Field(default=0.375, description="slate only")
    tray_depth: Positive = Field(default=0.30, description="slate only")
    tray_mass: Positive = 1.0
    platter_width: Positive = Field(default=0.45, description="lipped_platter only")
    platter_depth: Positive = Field(default=0.30, description="lipped_platter only")
    platter_base_thickness: Positive = Field(default=0.006, description="lipped_platter only")
    platter_lip_height: Positive = Field(
        default=0.012, description="lipped_platter only; above the platter floor"
    )
    platter_lip_wall: Positive = Field(default=0.008, description="lipped_platter only")
    platter_mass: Positive = Field(default=1.5, description="lipped_platter only")
    corelle_diameter: Positive = Field(
        default=10.25 * INCH, description="corelle_plate only; published 10-1/4 in (KC35285)"
    )
    corelle_height: Positive = Field(
        default=0.75 * INCH,
        description="corelle_plate only; published 3/4 in overall, foot to rim top",
    )
    corelle_well_radius: Positive = Field(
        default=0.095,
        description="corelle_plate only; UNVERIFIED estimate of the rim's inner edge radius",
    )
    corelle_glass_thickness: Positive = Field(
        default=0.003, description="corelle_plate only; UNVERIFIED Vitrelle glass thickness"
    )
    corelle_foot_height: Positive = Field(
        default=0.004, description="corelle_plate only; UNVERIFIED foot-ring height"
    )
    corelle_foot_radius: Positive = Field(
        default=0.080, description="corelle_plate only; UNVERIFIED foot-ring outer radius"
    )
    corelle_foot_width: Positive = Field(
        default=0.006, description="corelle_plate only; UNVERIFIED foot-ring radial width"
    )
    corelle_mass: Positive = Field(
        default=0.40, description="corelle_plate only; UNVERIFIED mass estimate"
    )
    corelle_rim_segments: int = Field(
        default=48, ge=12, description="corelle_plate only; convex boxes approximating the rim"
    )
    tray_friction: Friction = (0.6, 0.005, 0.003)
    counter_x_min: float = 0.98
    counter_x_max: float = 1.70
    counter_width: Positive = 1.0
    counter_thickness: Positive = 0.04
    counter_leg_width: Positive = 0.05
    counter_friction: Friction = (0.7, 0.005, 0.0001)

    dumpling_depth: Positive = 0.030
    dumpling_width: Positive = 0.045
    dumpling_height: Positive = 0.025
    dumpling_mass: Positive = 0.020
    dumpling_friction: Friction = (0.45, 0.005, 0.003)
    dumpling_contact_dimension: Literal[3, 4, 6] = 6
    # Closing position commanded on food; fully closed, so the force limit below,
    # not the position, decides where the fingers stop.
    food_grip_command: float = Field(default=1.0, gt=0, le=1)
    food_grip_force: Positive = Field(
        default=20.0,
        description=(
            "Normal force per pad (N) the gripper closes to on food, as the real 2F-85 "
            "is commanded a force: its adjustable range is about 20-235 N, and the "
            "prpl-tidybot driver sends 100% (motor_cmd.force = 100)."
        ),
    )
    gripper_contact_dimension: Literal[3, 4, 6] = 4
    scraper_width: Positive = 0.25
    scraper_blade_height: Positive = 0.12
    scraper_blade_thickness: Positive = 0.001
    scraper_handle_width: Positive = 0.018
    scraper_handle_height: Positive = 0.020
    scraper_mass: Positive = 0.275
    scraper_tilt_degrees: float = Field(default=45, ge=-45, le=45)
    scraper_friction: Friction = (0.4, 0.005, 0.0001)
    scraper_blade_friction: Friction = (0.25, 0.005, 0.0001)

    @property
    def rack_front_x(self) -> float:
        return self.rack_center_x - self.rack_depth / 2

    @property
    def rack_back_x(self) -> float:
        return self.rack_center_x + self.rack_depth / 2

    @property
    def rack_top_z(self) -> float:
        return self.counter_height + self.rack_height

    @property
    def serving_depth(self) -> float:
        """World-x extent of the serving object's footprint."""
        if self.serving_target == "lipped_platter":
            return self.platter_depth
        if self.serving_target == "corelle_plate":
            return self.corelle_diameter
        return 2 * self.plate_radius if self.serving_target == "round_plate" else self.tray_depth

    @property
    def serving_width(self) -> float:
        """World-y extent of the serving object's footprint."""
        if self.serving_target == "lipped_platter":
            return self.platter_width
        if self.serving_target == "corelle_plate":
            return self.corelle_diameter
        return 2 * self.plate_radius if self.serving_target == "round_plate" else self.tray_width

    @property
    def serving_thickness(self) -> float:
        """Height of the surface food rests on: the platter floor or plate well, not a rim."""
        if self.serving_target == "lipped_platter":
            return self.platter_base_thickness
        if self.serving_target == "corelle_plate":
            return self.corelle_foot_height + self.corelle_glass_thickness
        return self.tray_thickness

    @property
    def serving_mass(self) -> float:
        return {"lipped_platter": self.platter_mass, "corelle_plate": self.corelle_mass}.get(
            self.serving_target, self.tray_mass
        )

    @property
    def corelle_rim_top_z(self) -> float:
        return self.counter_height + self.corelle_height

    @property
    def platter_lip_top_z(self) -> float:
        return self.counter_height + self.platter_base_thickness + self.platter_lip_height

    @property
    def tray_center_x(self) -> float:
        return self.rack_front_x - self.gap - self.serving_depth / 2

    @property
    def tray_center_y(self) -> float:
        return self.rack_center_y

    @property
    def tray_front_x(self) -> float:
        return self.tray_center_x - self.serving_depth / 2

    @property
    def tray_back_x(self) -> float:
        return self.rack_front_x - self.gap

    @property
    def tray_top_z(self) -> float:
        return self.counter_height + self.serving_thickness

    @property
    def counter_front_x(self) -> float:
        return self.counter_x_min

    @property
    def counter_center_x(self) -> float:
        return (self.counter_x_min + self.counter_x_max) / 2

    @property
    def counter_depth(self) -> float:
        return self.counter_x_max - self.counter_x_min

    @property
    def counter_y_min(self) -> float:
        return -self.counter_width / 2

    @property
    def counter_y_max(self) -> float:
        return self.counter_width / 2

    @model_validator(mode="after")
    def validate_layout(self) -> "RackServingConfig":
        """Reject unsupported layouts rather than silently clipping hardware."""
        if self.rack_height < self.serving_thickness:
            raise ValueError("Rack top must be at or above tray top")
        if self.serving_target == "lipped_platter":
            if self.platter_lip_top_z >= self.rack_top_z:
                raise ValueError("Platter lip must stay below the rack top")
            if 2 * self.platter_lip_wall >= min(self.platter_depth, self.platter_width):
                raise ValueError("Platter lip must leave an interior")
        if self.serving_target == "corelle_plate":
            if self.corelle_rim_top_z >= self.rack_top_z:
                raise ValueError("Plate rim must stay below the rack top")
            if not (
                self.corelle_foot_width
                < self.corelle_foot_radius
                < self.corelle_well_radius
                < self.corelle_diameter / 2
            ):
                raise ValueError("Plate must nest foot ring inside well inside rim")
            if self.corelle_foot_height + self.corelle_glass_thickness >= self.corelle_height:
                raise ValueError("Plate well must sit below the rim top")
        if not self.counter_x_min <= self.tray_front_x < self.rack_back_x <= self.counter_x_max:
            raise ValueError("Rack and tray must fit on the counter in x")
        if (
            abs(self.rack_center_y) + max(self.rack_width, self.serving_width) / 2
            > self.counter_width / 2
        ):
            raise ValueError("Rack and tray must fit on the counter in y")
        if self.counter_thickness >= self.counter_height:
            raise ValueError("Counter thickness must leave room for legs")
        if self.counter_leg_width >= min(self.counter_depth, self.counter_width) / 2:
            raise ValueError("Counter legs must fit under the counter")
        if max(2 * self.rack_wire_radius, self.concept_thickness) >= self.rack_height:
            raise ValueError("Rack surface must leave room for support legs")
        if (
            not 2 * self.rack_wire_radius
            < self.rack_wire_spacing
            < min(self.rack_depth, self.rack_width)
        ):
            raise ValueError("Wire spacing must leave open gaps within the rack footprint")
        if not self.concept_web_width < self.concept_pitch < min(self.rack_depth, self.rack_width):
            raise ValueError("Concept platform must have actual perforations")
        if self.rack_leg_width >= min(self.rack_depth, self.rack_width) / 2:
            raise ValueError("Rack legs must fit within its footprint")
        if self.dumpling_depth * 3 >= self.rack_depth or self.dumpling_width * 3 >= self.rack_width:
            raise ValueError("Rack must accommodate five separated dumpling proxies")
        if self.scraper_blade_thickness >= self.scraper_handle_width:
            raise ValueError("Scraper blade must be thinner than its grip")
        if self.scraper_width >= self.rack_width:
            raise ValueError("Scraper must fit across the rack")
        return self
