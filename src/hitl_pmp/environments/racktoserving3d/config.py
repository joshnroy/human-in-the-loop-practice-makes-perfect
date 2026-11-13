"""Metric layout and explicitly assumed hardware for rack-to-serving.

The specified rack footprint is 0.375 m wide (world y) by 0.255 m deep
(world x). Rack height and mass are UNVERIFIED configurable assumptions, not
manufacturer specifications. All other hardware dimensions, masses and friction
below are modeling assumptions requiring physical calibration.
The food is Pouring3D's rigid frozen potsticker (`potsticker.py`), whose extents
are read from its shape, not configured; it is not a deformable or thermal model.
``wire_solid`` renders the wire rack with one invisible solid support slab;
``wire`` retains the original wire contact geometry for historical replays.
``concept`` denotes an invented perforated platform, not the specified wire rack.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .potsticker import CollisionModel, RackPotsticker

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
    centre line. Both the rack and the serving object are welded to the counter
    (Josh, 2026-11-06): no push, sweep or dropped food can move either.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    surface_kind: Literal["wire_solid", "wire", "concept"] = "wire_solid"
    # "welded" (default since 2026-11-06, Josh): PickFood, PlaceFoodOnTray and
    # ReturnFoodToRack drive and reach for real, but ignore food in planning, never
    # touch food physically, and take or set down the dumpling by a grasp weld and a
    # snapped, upright pose (`WeldedFood`). "physical" is the force-closed grasp.
    food_handling: Literal["physical", "welded"] = "welded"
    # grasp_weld includes home_weld and rigid transport after a verified grasp.
    # Keep the historical physical mode as the default for matched comparisons.
    scraper_attachment: Literal["physical", "home_weld", "grasp_weld"] = "physical"
    mode: Literal["training", "deployment"] = "training"
    # Sweep(d), GatherAndSweep's final stroke alone, as a seventh robot skill
    # (default since 2026-11-05, Josh). start_layout chooses an opt-in fixed layout
    # (`RackLayout.lined_up`) for episode starts.
    sweep_skill: bool = True
    # The benchmark's bulk-transfer protocol (Josh, 2026-11-11): PlaceFoodOnTray is not
    # offered, in practice or evaluation, and a held-out task ends with its first Sweep
    # or GatherAndSweep, whatever it delivered. With placement on offer every method
    # solved 36/36 tasks before and after practice by carrying the leftovers, so what
    # the sweeps had learned could not show.
    bulk_transfer_only: bool = False
    # The two-route reward structure (Josh, 2026-11-12), opt-in: Sweep succeeds when at
    # least three dumplings are on the plate and that outcome is worth 3; GatherAndSweep
    # succeeds when all five are and is worth 5 (`TWO_ROUTE_REWARDS`). The rewards are
    # given to the methods, never estimated. Needs `bulk_transfer_only`, whose one
    # bulk attempt per task is what makes each bulk skill a route.
    two_route_rewards: bool = False
    # RobotBatchReturn (Josh, 2026-11-13), opt-in: one robot skill that puts every
    # dumpling back on its start slot, the robot counterpart of the person's batch
    # return, offered in training only (`robot_batch_return.py`). It needs a start
    # layout with slots (not "scattered", whose return has no place to restore) and
    # welded food, the only handling its slot placement is written for.
    robot_batch_return: bool = False
    # How a dumpling the robot's batch return sets down gets its slot's yaw. "snap":
    # the set-down writes it, as welded handling writes the spot (an abstraction, made
    # in `WeldedFood.set_down` alone). "hand": the hand turns the dumpling it holds to
    # that yaw and nothing is written; measured on setup A (2026-11-13), see the
    # skill's module docstring. Read only when `robot_batch_return` is on.
    robot_batch_return_yaw: Literal["snap", "hand"] = "snap"
    # "spaced" (Josh, 2026-11-12): the two-route start, `RackLayout.spaced`.
    # "middle_three" (Josh, 2026-11-12): three `spaced_pitch` apart in the middle and
    # one `side_offset` out on each side, `RackLayout.middle_three`.
    # "custom" (2026-11-12 measurement study): `RackLayout.custom`, one explicit
    # offset per dumpling in `custom_layout`.
    start_layout: Literal[
        "scattered", "pile2col", "stack", "row", "spaced", "middle_three", "custom"
    ] = "scattered"
    # start_layout custom only: per dumpling, in slot order, (metres along the row from
    # the plate's centre line, metres up the rack away from the plate from the rack's x
    # centre line, radians of yaw added to `spaced_long_axis`).
    custom_layout: tuple[tuple[float, float, float], ...] = ()
    spaced_pitch: Positive = Field(
        default=0.04, description="start_layout spaced and middle_three only"
    )
    side_offset: Positive = Field(default=0.13, description="start_layout middle_three only")
    spaced_long_axis: Literal["x", "y"] = "x"
    # scattered only: the probability an episode start or a human batch return places
    # the aligned "stack" pile instead of a scattered layout (0.25 since 2026-11-05,
    # Josh). At 0 no coin is drawn, so the scattered draws are exactly the old ones.
    gathered_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    counter_height: Positive = 0.90
    rack_height: Positive = Field(default=0.025, description="UNVERIFIED height assumption")
    rack_mass: Positive = Field(default=0.5, description="UNVERIFIED mass assumption")
    tray_thickness: Positive = 0.012
    gap: Nonnegative = 0.010
    # Opt-in (2026-11-12 measurement study), Corelle plate only: how far the plate's
    # outer near edge sits under the rack's front edge, replacing `gap`. Past the rim's
    # 35 mm width the rack's front edge overhangs the scored well. 0 is off.
    plate_tuck: Nonnegative = 0.0
    # Opt-in (2026-11-12 measurement study): speed (m/s) of Sweep's stroke and of
    # GatherAndSweep's final stroke. None is the constant `SWEEP_SPEED_M_S`.
    final_sweep_speed: Positive | None = None
    # What the learned sampler's classifier reads for Sweep and GatherAndSweep
    # (Josh, 2026-11-13), `RackServingSkillProvider.hand_selected_feature_transform`.
    # "parameters" (the default): a bias and the skill's parameters.
    # "parameters_and_layout": the bias, each dumpling's (x, y) from the rack's centre in
    # slot order, then the parameters; on recorded setup-A tries, where the layout is a
    # +-5 mm jitter, it lowered the classifier's held-out accuracy.
    # "full_state": the method's own row, every object's whole state, which is what
    # every run before 2026-11-13 used. That row carries the robot's base and the
    # scraper's pose: practice consults the sampler with the robot at the rack and an
    # evaluation task consults it right after PickScraper, so a sampler fitted on it
    # was scored on states it had not seen.
    sampler_features: Literal["parameters", "parameters_and_layout", "full_state"] = "parameters"
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
    # The scraper's home on the counter; its blade lies along y, so it must sit inside.
    tool_rest_y: float = -0.34
    counter_thickness: Positive = 0.04
    counter_leg_width: Positive = 0.05
    counter_friction: Friction = (0.7, 0.005, 0.0001)

    dumpling_mass: Positive = 0.020
    dumpling_friction: Friction = (0.45, 0.005, 0.003)
    dumpling_contact_dimension: Literal[3, 4, 6] = 6
    dumpling_collision: CollisionModel = Field(
        default="hull",
        description=(
            "Potsticker contact geometry; every option keeps the full textured mesh as "
            "visual geometry and the full model's mass and inertia (see potsticker.py)"
        ),
    )
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
    # Opt-in (2026-11-12 prototype): how far below its plan the final sweep's blade may
    # be commanded while food under it holds it up (`BladeHeightFeedback`). 0 is off.
    # The command mirrors the lift (plan minus lift), so it is below the plate
    # clearance only while the blade itself is held more than that far above its plan.
    blade_height_feedback_limit: float = Field(default=0.0, ge=0.0, le=0.02)
    scraper_friction: Friction = (0.4, 0.005, 0.0001)
    scraper_blade_friction: Friction = (0.25, 0.005, 0.0001)

    @property
    def dumpling_depth(self) -> float:
        """World-x extent of the potsticker at zero yaw: its short axis."""
        return float(RackPotsticker.extents()[0])

    @property
    def dumpling_width(self) -> float:
        """World-y extent of the potsticker at zero yaw: its 55 mm long axis."""
        return float(RackPotsticker.extents()[1])

    @property
    def dumpling_height(self) -> float:
        return float(RackPotsticker.extents()[2])

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
        return self.tray_back_x - self.serving_depth / 2

    @property
    def tray_center_y(self) -> float:
        return self.rack_center_y

    @property
    def tray_front_x(self) -> float:
        return self.tray_center_x - self.serving_depth / 2

    @property
    def tray_back_x(self) -> float:
        if self.plate_tuck > 0:
            return self.rack_front_x + self.plate_tuck
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
        if (self.start_layout == "custom") != (len(self.custom_layout) == 5):
            raise ValueError("start_layout custom needs exactly five custom_layout offsets")
        if self.two_route_rewards and not (self.bulk_transfer_only and self.sweep_skill):
            raise ValueError("two_route_rewards needs bulk_transfer_only and sweep_skill")
        if self.robot_batch_return and (
            self.start_layout == "scattered" or self.food_handling != "welded"
        ):
            raise ValueError("robot_batch_return needs a slotted start_layout and welded food")
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
        if self.plate_tuck > 0:
            if self.serving_target != "corelle_plate" or self.surface_kind == "concept":
                raise ValueError("plate_tuck needs the Corelle plate under a wire rack")
            if self.corelle_rim_top_z >= self.rack_top_z - 2 * self.rack_wire_radius:
                raise ValueError("Plate rim must fit under the rack's deck to be tucked")
            foot_dx = self.rack_front_x - self.tray_center_x
            foot_dy = self.rack_width / 2 - self.rack_leg_width
            if foot_dx**2 + foot_dy**2 <= (self.corelle_diameter / 2) ** 2:
                raise ValueError("A tucked plate must stay clear of the rack's front feet")
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
