"""Task-local native MuJoCo geometry; no KINDER defaults or classes are patched.

Wire mode has round crossing wires and open gaps. Concept mode has a flat
rectangular-web perforated platform. Neither has a hidden supporting plate or
raised front lip. Both have four feet, one free joint, and frictional support.
Hardware properties are assumptions documented in :mod:`.config`.
"""

import math
import xml.etree.ElementTree as ET
from typing import Any

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.scene import (
    CORELLE_TYPE,
    COUNTER_RGBA,
    COUNTER_TYPE,
    DUMPLING_TYPE,
    PLATE_TYPE,
    PLATTER_TYPE,
    RACK_TYPE,
    SCRAPER_TYPE,
    TRAY_TYPE,
)

from .potsticker import RackPotsticker


def _values(*, values: Any) -> str:
    return " ".join(str(value) for value in values)


def _bottom_bounds(*, position: Any, options: dict[str, Any]) -> list[float]:
    half_x, half_y, half_z = options["size"]
    return [
        float(position[0]) - half_x,
        float(position[1]) - half_y,
        float(position[2]),
        float(position[0]) + half_x,
        float(position[1]) + half_y,
        float(position[2]) + 2 * half_z,
    ]


def _grid_centers(*, span: float, edge_width: float, spacing: float) -> list[float]:
    intervals = math.ceil((span - edge_width) / spacing)
    return [
        -span / 2 + edge_width / 2 + index * (span - edge_width) / intervals
        for index in range(intervals + 1)
    ]


def install_geometry() -> None:
    """Idempotently register only rack-serving names; imports KINDER lazily."""
    from kinder.envs.dynamic3d.objects.base import (
        REGISTERED_FIXTURES,
        REGISTERED_OBJECTS,
        register_fixture,
        register_object,
    )
    from kinder.envs.dynamic3d.objects.fixtures import Cupboard
    from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid, Wiper

    if COUNTER_TYPE not in REGISTERED_FIXTURES:

        class RackServingCounter(Cupboard):
            """Only the physical counter and legs collide; shelves are sites."""

            def _compute_shelf_z_positions(self) -> list[float]:
                config = RackServingConfig.model_validate(
                    self.fixture_config["rack_serving_config"]
                )
                return [config.counter_height, config.rack_top_z, config.counter_height]

            def _create_xml_element(self) -> ET.Element:
                config = RackServingConfig.model_validate(
                    self.fixture_config["rack_serving_config"]
                )
                body = ET.Element(
                    "body",
                    name=self.name,
                    pos=_values(values=self.position),
                    quat=_values(values=(math.cos(self.yaw / 2), 0, 0, math.sin(self.yaw / 2))),
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{self.name}_counter",
                    type="box",
                    size=_values(
                        values=(
                            config.counter_depth / 2,
                            config.counter_width / 2,
                            config.counter_thickness / 2,
                        )
                    ),
                    pos=f"0 0 {config.counter_height - config.counter_thickness / 2}",
                    rgba=_values(values=COUNTER_RGBA),
                    friction=_values(values=config.counter_friction),
                )
                leg_height = config.counter_height - config.counter_thickness
                for index, (sign_x, sign_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{self.name}_leg_{index}",
                        type="box",
                        size=_values(
                            values=(
                                config.counter_leg_width / 2,
                                config.counter_leg_width / 2,
                                leg_height / 2,
                            )
                        ),
                        pos=_values(
                            values=(
                                sign_x * (config.counter_depth - config.counter_leg_width) / 2,
                                sign_y * (config.counter_width - config.counter_leg_width) / 2,
                                leg_height / 2,
                            )
                        ),
                        rgba="0.30 0.32 0.34 1",
                        friction=_values(values=config.counter_friction),
                    )
                for shelf, height in ((1, config.rack_top_z), (2, config.counter_height)):
                    ET.SubElement(
                        body,
                        "site",
                        name=f"{self.name}_shelf{shelf}",
                        type="box",
                        size=_values(values=(config.rack_depth / 2, config.rack_width / 2, 0.0001)),
                        pos=_values(
                            values=(
                                config.rack_center_x - config.counter_center_x,
                                config.rack_center_y,
                                height - 0.0001,
                            )
                        ),
                        rgba="0 0 0 0",
                        group="3",
                    )
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, fixture_config: dict[str, Any]
            ) -> list[float]:
                config = RackServingConfig.model_validate(fixture_config["rack_serving_config"])
                return [
                    float(pos[0]) - config.counter_depth / 2,
                    float(pos[1]) - config.counter_width / 2,
                    float(pos[2]),
                    float(pos[0]) + config.counter_depth / 2,
                    float(pos[1]) + config.counter_width / 2,
                    float(pos[2]) + config.counter_height,
                ]

        register_fixture(RackServingCounter)

    if RACK_TYPE not in REGISTERED_OBJECTS:

        class CoolingRack(Cuboid):
            """Bottom-origin movable rack with native wire or perforation colliders."""

            def _create_xml_element(self) -> ET.Element:
                config = RackServingConfig.model_validate(self.options["rack_serving_config"])
                body = ET.Element("body", name=self.name)
                ET.SubElement(body, "freejoint", name=self.joint_name)
                radius = config.rack_wire_radius
                wire = config.surface_kind == "wire"
                thickness = 2 * radius if wire else config.concept_thickness
                web_width = 2 * radius if wire else config.concept_web_width
                pitch = config.rack_wire_spacing if wire else config.concept_pitch
                surface_z = config.rack_height - thickness / 2
                for axis, span, cross_span in (
                    ("x", config.rack_depth, config.rack_width),
                    ("y", config.rack_width, config.rack_depth),
                ):
                    for index, offset in enumerate(
                        _grid_centers(span=cross_span, edge_width=web_width, spacing=pitch)
                    ):
                        attributes = {
                            "name": f"{self.name}_{'wire' if wire else 'web'}_{axis}_{index}"
                        }
                        if wire:
                            half_length = span / 2 - radius
                            ends = (
                                (-half_length, offset, surface_z, half_length, offset, surface_z)
                                if axis == "x"
                                else (
                                    offset,
                                    -half_length,
                                    surface_z,
                                    offset,
                                    half_length,
                                    surface_z,
                                )
                            )
                            attributes.update(
                                type="capsule", size=str(radius), fromto=_values(values=ends)
                            )
                        else:
                            size = (
                                (span / 2, web_width / 2, thickness / 2)
                                if axis == "x"
                                else (web_width / 2, span / 2, thickness / 2)
                            )
                            position = (
                                (0, offset, surface_z) if axis == "x" else (offset, 0, surface_z)
                            )
                            attributes.update(
                                type="box", size=_values(values=size), pos=_values(values=position)
                            )
                        ET.SubElement(body, "geom", attrib=attributes)
                leg_height = config.rack_height - thickness
                for index, (sign_x, sign_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{self.name}_foot_{index}",
                        type="box",
                        size=_values(
                            values=(
                                config.rack_leg_width / 2,
                                config.rack_leg_width / 2,
                                leg_height / 2,
                            )
                        ),
                        pos=_values(
                            values=(
                                sign_x * (config.rack_depth - config.rack_leg_width) / 2,
                                sign_y * (config.rack_width - config.rack_leg_width) / 2,
                                leg_height / 2,
                            )
                        ),
                    )
                geoms = body.findall("geom")
                for geom in geoms:
                    geom.set("mass", str(config.rack_mass / len(geoms)))
                    geom.set("friction", _values(values=config.rack_friction))
                    geom.set("condim", str(config.support_contact_dimension))
                    geom.set("rgba", "0.65 0.68 0.71 1")
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                return _bottom_bounds(position=pos, options=object_config)

        register_object(name=RACK_TYPE)(CoolingRack)

    if TRAY_TYPE not in REGISTERED_OBJECTS:

        class SlateTray(Cuboid):
            """Flat bottom-origin slate slab, with a free joint and no walls."""

            def _create_xml_element(self) -> ET.Element:
                self.length, self.width, self.height = (2 * extent for extent in self.size)
                self.wall_thickness = 0.0
                self.fixed = False
                body = super()._create_xml_element()
                geom = body.find("geom")
                assert geom is not None
                geom.set("name", f"{self.name}_slate")
                geom.set("pos", f"0 0 {self.height / 2}")
                geom.set("friction", _values(values=self.options["friction"]))
                geom.set("condim", str(self.options["contact_dimension"]))
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                return _bottom_bounds(position=pos, options=object_config)

        register_object(name=TRAY_TYPE)(SlateTray)

    if PLATE_TYPE not in REGISTERED_OBJECTS:

        class RoundPlate(Cuboid):
            """Flat bottom-origin circular plate: one cylinder, free joint, no rim."""

            def _create_xml_element(self) -> ET.Element:
                self.length, self.width, self.height = (2 * extent for extent in self.size)
                self.wall_thickness = 0.0
                self.fixed = False
                body = super()._create_xml_element()
                geom = body.find("geom")
                assert geom is not None
                geom.set("name", f"{self.name}_plate")
                geom.set("type", "cylinder")
                geom.set("size", _values(values=(self.length / 2, self.height / 2)))
                geom.set("pos", f"0 0 {self.height / 2}")
                geom.set("friction", _values(values=self.options["friction"]))
                geom.set("condim", str(self.options["contact_dimension"]))
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                return _bottom_bounds(position=pos, options=object_config)

        register_object(name=PLATE_TYPE)(RoundPlate)

    if PLATTER_TYPE not in REGISTERED_OBJECTS:

        class LippedPlatter(Cuboid):
            """Bottom-origin rectangular platter: one floor slab and a closed four-wall lip.

            The walls sit on the floor's outer perimeter, so the outer footprint is
            the floor's. Mass is split by volume so the body totals the configured
            mass; every geom shares the slate's friction and contact dimension.
            """

            def _create_xml_element(self) -> ET.Element:
                half_x, half_y, _ = self.size
                base = self.options["base_thickness"]
                lip = self.options["lip_height"]
                wall = self.options["lip_wall"]
                self.length, self.width, self.height = 2 * half_x, 2 * half_y, base
                self.wall_thickness = 0.0
                self.fixed = False
                body = ET.Element("body", name=self.name)
                ET.SubElement(body, "freejoint", name=self.joint_name)
                lip_z = base + lip / 2
                parts = (
                    ("platter_floor", (0, 0, base / 2), (half_x, half_y, base / 2)),
                    (
                        "lip_front",
                        (-half_x + wall / 2, 0, lip_z),
                        (wall / 2, half_y - wall, lip / 2),
                    ),
                    ("lip_back", (half_x - wall / 2, 0, lip_z), (wall / 2, half_y - wall, lip / 2)),
                    ("lip_left", (0, -half_y + wall / 2, lip_z), (half_x, wall / 2, lip / 2)),
                    ("lip_right", (0, half_y - wall / 2, lip_z), (half_x, wall / 2, lip / 2)),
                )
                volume = sum(math.prod(size) for _, _, size in parts)
                for name, position, size in parts:
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{self.name}_{name}",
                        type="box",
                        pos=_values(values=position),
                        size=_values(values=size),
                        mass=str(self.options["mass"] * math.prod(size) / volume),
                        friction=_values(values=self.options["friction"]),
                        condim=str(self.options["contact_dimension"]),
                        rgba=_values(values=self.options["rgba"]),
                    )
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                half_x, half_y, _ = object_config["size"]
                height = object_config["base_thickness"] + object_config["lip_height"]
                return _bottom_bounds(position=pos, options={"size": [half_x, half_y, height / 2]})

        register_object(name=PLATTER_TYPE)(LippedPlatter)

    if CORELLE_TYPE not in REGISTERED_OBJECTS:

        class CorellePlate(Cuboid):
            """Bottom-origin model of a Corelle 10-1/4 in dinner plate, from convex pieces.

            MuJoCo collides only convex geoms, so the concave dish is a flat well
            cylinder, a ring of tilted boxes whose top faces rise from the well's
            top at its edge to the published overall height at the outer edge,
            and a ring of foot boxes under the well. Mass is split by volume.
            """

            def _create_xml_element(self) -> ET.Element:
                options = self.options
                outer = options["corelle_diameter"] / 2
                well = options["corelle_well_radius"]
                glass = options["corelle_glass_thickness"]
                foot_height = options["corelle_foot_height"]
                foot_outer = options["corelle_foot_radius"]
                foot_width = options["corelle_foot_width"]
                rim_count = int(options["corelle_rim_segments"])
                well_top = foot_height + glass
                self.length = self.width = 2 * outer
                self.height = options["corelle_height"]
                self.wall_thickness = 0.0
                self.fixed = False
                body = ET.Element("body", name=self.name)
                ET.SubElement(body, "freejoint", name=self.joint_name)
                parts: list[tuple[str, str, tuple[float, ...], tuple[float, ...], float]] = [
                    ("well", "cylinder", (0, 0, foot_height + glass / 2), (well, glass / 2), 0.0)
                ]
                run, rise = outer - well, options["corelle_height"] - well_top
                slope = math.atan2(rise, run)
                half_slant = math.hypot(run, rise) / 2
                half_tangent = outer * math.tan(math.pi / rim_count)
                radial = (well + outer) / 2 + glass / 2 * math.sin(slope)
                height = (well_top + options["corelle_height"]) / 2 - glass / 2 * math.cos(slope)
                for index in range(rim_count):
                    yaw = 2 * math.pi * index / rim_count
                    parts.append((
                        f"rim_{index}",
                        "box",
                        (radial * math.cos(yaw), radial * math.sin(yaw), height),
                        (half_slant, half_tangent, glass / 2),
                        yaw,
                    ))
                foot_count = max(12, rim_count // 2)
                foot_radius = foot_outer - foot_width / 2
                for index in range(foot_count):
                    yaw = 2 * math.pi * index / foot_count
                    parts.append((
                        f"foot_{index}",
                        "box",
                        (foot_radius * math.cos(yaw), foot_radius * math.sin(yaw), foot_height / 2),
                        (
                            foot_width / 2,
                            foot_outer * math.tan(math.pi / foot_count),
                            foot_height / 2,
                        ),
                        yaw,
                    ))

                def volume(*, kind: str, size: tuple[float, ...]) -> float:
                    return (
                        2 * math.pi * size[0] ** 2 * size[1]
                        if kind == "cylinder"
                        else 8 * math.prod(size)
                    )

                total = sum(volume(kind=kind, size=size) for _, kind, _, size, _ in parts)
                for name, kind, position, size, yaw in parts:
                    # Yaw about z, then pitch the rim's radial axis up its slope.
                    pitch = slope if name.startswith("rim_") else 0.0
                    quaternion = (
                        math.cos(yaw / 2) * math.cos(pitch / 2),
                        math.sin(yaw / 2) * math.sin(pitch / 2),
                        -math.cos(yaw / 2) * math.sin(pitch / 2),
                        math.sin(yaw / 2) * math.cos(pitch / 2),
                    )
                    ET.SubElement(
                        body,
                        "geom",
                        name=f"{self.name}_{name}",
                        type=kind,
                        pos=_values(values=position),
                        quat=_values(values=quaternion),
                        size=_values(values=size),
                        mass=str(options["mass"] * volume(kind=kind, size=size) / total),
                        friction=_values(values=options["friction"]),
                        condim=str(options["contact_dimension"]),
                        rgba=_values(values=options["rgba"]),
                    )
                return body

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                outer = object_config["corelle_diameter"] / 2
                height = object_config["corelle_height"]
                return _bottom_bounds(position=pos, options={"size": [outer, outer, height / 2]})

        register_object(name=CORELLE_TYPE)(CorellePlate)

    if DUMPLING_TYPE not in REGISTERED_OBJECTS:

        class RigidPotsticker(Cuboid):
            """Pouring3D's rigid frozen potsticker; no claims about dough deformation."""

            def _create_xml_element(self) -> ET.Element:
                return RackPotsticker.body(
                    name=self.name,
                    joint=self.joint_name,
                    mass=float(self.options["mass"]),
                    friction=_values(values=self.options["friction"]),
                    condim=int(self.options["contact_dimension"]),
                    index=int(self.name.rsplit("_", 1)[1]),
                    collision=self.options.get("collision", "full"),
                )

            def get_assets(self) -> list[ET.Element]:
                return RackPotsticker.assets(collision=self.options.get("collision", "full"))

        register_object(name=DUMPLING_TYPE)(RigidPotsticker)

    if SCRAPER_TYPE not in REGISTERED_OBJECTS:

        class BenchScraper(Wiper):
            """Broad steel blade along local x, with a full-width top grip.

            The backend identifies handle/blade by name, overriding ShelfSkills'
            legacy argmax-based classification for the original narrow wiper.
            Body origin is the blade's bottom edge, not its center of mass.
            """

            def _create_xml_element(self) -> ET.Element:
                body = ET.Element("body", name=self.name)
                ET.SubElement(body, "freejoint", name=self.joint_name)
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{self.name}_handle",
                    type="box",
                    size=_values(
                        values=(
                            self.head_length / 2,
                            self.handle_width / 2,
                            self.handle_height / 2,
                        )
                    ),
                    pos=f"0 0 {self.head_height + self.handle_height / 2}",
                    rgba=self.handle_rgba,
                    mass=str(self.mass * 2 / 11),
                    friction=_values(values=self.options["friction"]),
                )
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{self.name}_blade",
                    type="box",
                    size=_values(
                        values=(
                            self.head_length / 2,
                            self.options["blade_thickness"] / 2,
                            self.head_height / 2,
                        )
                    ),
                    pos=f"0 0 {self.head_height / 2}",
                    rgba=self.head_rgba,
                    mass=str(self.mass * 9 / 11),
                    friction=_values(values=self.options["blade_friction"]),
                    priority="1",
                )
                return body

            def get_bounding_box_dimensions(self) -> tuple[float, float, float]:
                return self.head_length, self.handle_width, self.head_height + self.handle_height

            @staticmethod
            def get_bounding_box_from_config(  # noqa: PLR0917 -- KINDER calls positionally
                pos: Any, object_config: dict[str, Any]
            ) -> list[float]:
                return _bottom_bounds(
                    position=pos,
                    options={
                        "size": [
                            object_config["head_length"] / 2,
                            object_config["handle_width"] / 2,
                            (object_config["head_height"] + object_config["handle_height"]) / 2,
                        ]
                    },
                )

        register_object(name=SCRAPER_TYPE)(BenchScraper)
