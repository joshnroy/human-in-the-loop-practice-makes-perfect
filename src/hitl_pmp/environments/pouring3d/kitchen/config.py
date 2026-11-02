"""Product candidates and explicit, replaceable simulation assumptions."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KitchenGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # Candidate identity and user-supplied empty mass; no robot payload claim.
    pan_product: str = "Lodge L8SK3 10.25-inch skillet"
    pan_source: str = "https://www.lodgecastiron.com/collections/skillets/products/round-cast-iron-classic-skillet"
    pan_diameter_m: float = Field(default=0.26035, gt=0)
    pan_empty_mass_kg: float = Field(default=2.57, gt=0)
    jug_product: str = "Norpro 3037, approximately 1 L"
    jug_source: str = (
        "https://www.norpro.com/wp-content/uploads/2023/08/2023-Norpro50thYrCatalogDigitalw1.pdf"
    )
    # None is deliberately retained in the exported manifest: not measured.
    measured_pan_inner_diameter_m: float | None = None
    measured_pan_depth_m: float | None = None
    measured_jug_dimensions_m: tuple[float, float, float] | None = None
    measured_jug_handle_clearance_m: float | None = None
    measured_jug_mass_kg: float | None = None
    counter_height_m: float = Field(default=0.84, ge=0.70, le=1.0)
    # The pan sits directly on a fixed-height desk (~29 in, unmeasured) standing in
    # for the stove. At the previous 0.84 m the dumpling pickup was often out of reach.
    hob_height_m: float = Field(default=0.74, ge=0.70, le=1.0)
    pan_height_proxy_m: float = Field(default=0.0508, gt=0)
    pan_bottom_radius_proxy_m: float = Field(default=0.105, gt=0)
    pan_wall_proxy_m: float = Field(default=0.0045, gt=0)
    pan_handle_proxy_m: float = Field(default=0.10, gt=0)
    jug_dimensions_proxy_m: tuple[float, float, float] = (0.11, 0.114, 0.14)
    jug_wall_proxy_m: float = Field(default=0.003, gt=0)
    jug_mass_proxy_kg: float = Field(default=0.17, gt=0)
    jug_handle_outset_proxy_m: float = Field(default=0.016, gt=0)
    water_depth_proxy_m: float = Field(default=0.015, gt=0)
    retreat_distance_m: float = Field(default=0.45, ge=0.35, le=0.8)

    @model_validator(mode="after")
    def validate_cavities(self) -> "KitchenGeometry":
        if not self.pan_wall_proxy_m < self.water_depth_proxy_m < self.pan_height_proxy_m:
            raise ValueError("Water proxy must lie inside the open pan")
        if not self.pan_wall_proxy_m < self.pan_bottom_radius_proxy_m < self.pan_diameter_m / 2:
            raise ValueError("Invalid pan cavity proxy")
        if any(x <= 2 * self.jug_wall_proxy_m for x in self.jug_dimensions_proxy_m):
            raise ValueError("Jug dimensions must enclose a cavity")
        return self

    def scene_config(self, *, deployment: bool = False) -> dict[str, Any]:
        source = Path(__file__).parents[1] / "assets/potsticker-pour3-stations.json"
        scene = json.loads(source.read_text())
        scene["description"] = (
            "Kitchen pouring: cold practice; simulated cooking activation at deployment."
        )
        scene["fixtures"]["table"]["prep_table"]["height"] = self.counter_height_m
        scene["fixtures"]["pourstovetop"]["stovetop"]["height"] = self.hob_height_m
        for region, height in (
            ("pitcher_home", self.counter_height_m),
            ("tray_home", self.hob_height_m),
        ):
            bounds = scene["regions"][region]["ranges"][0]
            bounds[2] = bounds[5] = height + 0.0005
        pan = scene["objects"]["castironskillet"]["tray"]
        pan.update(
            length=self.pan_diameter_m,
            width=self.pan_diameter_m,
            height=self.pan_height_proxy_m,
            mass=self.pan_empty_mass_kg,
            bottom_radius=self.pan_bottom_radius_proxy_m,
            wall_thickness=self.pan_wall_proxy_m,
            handle_extension=self.pan_handle_proxy_m,
            water_depth_proxy_m=self.water_depth_proxy_m,
            water_present=deployment,
        )
        jug = scene["objects"]["pouringpitcher"]["pitcher"]
        jug.update(zip(("length", "width", "height"), self.jug_dimensions_proxy_m, strict=True))
        jug.update(
            mass=self.jug_mass_proxy_kg,
            wall_thickness=self.jug_wall_proxy_m,
            handle_outset=self.jug_handle_outset_proxy_m,
        )
        scene["regions"]["tray_interior"]["ranges"] = [
            [-0.08, -0.08, self.pan_wall_proxy_m, 0.08, 0.08, self.pan_height_proxy_m]
        ]
        return scene
