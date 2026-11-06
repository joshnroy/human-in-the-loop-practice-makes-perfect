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
    # The real setup stands the pan and the jug on the same 29 in desk (0.7366 m,
    # kept at 0.74 m), the pan's part standing in for the stove. At the previous
    # 0.84 m the dumpling pickup was often out of reach.
    counter_height_m: float = Field(default=0.74, ge=0.70, le=1.0)
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
    # Station layout, world (x, y) in metres. The defaults are the stations asset's: the
    # jug's table 2.1 m from the stove across an aisle. The robot's home x stays at the
    # aisle's 0.0 (`PracticePourController.create` pins it), so only its y is free.
    prep_table_xy: tuple[float, float] = (1.0, -1.05)
    pitcher_home_xy: tuple[float, float] = (0.7, -1.05)
    robot_home_y: float = -1.05
    # How far the base backs off in -x after PlacePitcherAtHome releases the jug.
    clear_workspace_backoff_m: float = Field(default=0.35, ge=0.0, le=0.8)

    @model_validator(mode="after")
    def validate_cavities(self) -> "KitchenGeometry":
        if not self.pan_wall_proxy_m < self.water_depth_proxy_m < self.pan_height_proxy_m:
            raise ValueError("Water proxy must lie inside the open pan")
        if not self.pan_wall_proxy_m < self.pan_bottom_radius_proxy_m < self.pan_diameter_m / 2:
            raise ValueError("Invalid pan cavity proxy")
        if any(x <= 2 * self.jug_wall_proxy_m for x in self.jug_dimensions_proxy_m):
            raise ValueError("Jug dimensions must enclose a cavity")
        # The prep table is 0.8 m square; the jug's footprint must stand on it.
        margin = 0.4 - max(self.jug_dimensions_proxy_m[:2]) / 2
        offset = [j - t for j, t in zip(self.pitcher_home_xy, self.prep_table_xy, strict=True)]
        if max(abs(d) for d in offset) > margin:
            raise ValueError("The jug's home must stand on the prep table")
        return self

    def scene_config(self, *, deployment: bool = False) -> dict[str, Any]:
        source = Path(__file__).parents[1] / "assets/potsticker-pour3-stations.json"
        scene = json.loads(source.read_text())
        scene["description"] = (
            "Kitchen pouring: cold practice; simulated cooking activation at deployment."
        )
        scene["fixtures"]["table"]["prep_table"]["height"] = self.counter_height_m
        for region, xy in (
            ("prep_table_home", self.prep_table_xy),
            ("pitcher_home", self.pitcher_home_xy),
            ("robot_home", (0.0, self.robot_home_y)),
        ):
            for bounds in scene["regions"][region]["ranges"]:
                bounds[0] = bounds[3] = xy[0]
                bounds[1] = bounds[4] = xy[1]
        # The jug's close-up camera follows the jug; its default pose is unchanged.
        detail = scene["cameras"]["pitcher_detail"]
        shift = [h - d for h, d in zip(self.pitcher_home_xy, (0.7, -1.05), strict=True)]
        for key in ("position", "lookat"):
            detail[key] = [detail[key][0] + shift[0], detail[key][1] + shift[1], detail[key][2]]
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
