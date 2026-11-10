"""Separate scene specification and explicitly approximate recycling rigid bodies."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, cast

import numpy as np

from .bin_geometry import TaperedBin
from .bin_regions import KitchenBinRegions
from .domain import BOTTLE_CAP_HEIGHT, BOTTLE_CAP_RADIUS_FACTOR, KitchenConfig

GOAL_REGION = "blocks_goal_region"


def internal_name(*, index: int) -> str:
    """KINDER's collision adapter discovers movables by its legacy cube prefix."""
    return f"cube_{index}"


def register_simulation_objects() -> None:
    """Register new names only, leaving every original object class untouched."""
    from kinder.envs.dynamic3d.objects.base import register_fixture, register_object
    from kinder.envs.dynamic3d.objects.fixtures import FixedCuboid
    from kinder.envs.dynamic3d.objects.primitive_objects import Bin, Cuboid

    @register_object(name="kitchen_tapered_bin")
    class KitchenTaperedBin(Bin):
        """KINDER's bin with sloped walls (`TaperedBin`), scored by its interior.

        The goal region keeps KINDER's mechanism -- a box on the bin, sized from its
        geometry -- and adds the taper: beside the narrow base, under the rim, is
        inside the box and outside the bin."""

        def geometry(self) -> TaperedBin:
            return TaperedBin(
                length=self.length,
                width=self.width,
                height=self.height,
                wall=self.wall_thickness,
                base_length=float(self.options["base_length"]),
                base_width=float(self.options["base_width"]),
            )

        def _create_xml_element(self) -> ET.Element:
            body = ET.Element("body", name=self.name)
            if self.fixed:
                body.set("mocap", "true")
            else:
                ET.SubElement(body, "freejoint", name=self.joint_name)
            panels = self.geometry().panels()
            for index, panel in enumerate(panels):
                x, y, z, w = panel.quaternion_xyzw
                geom = ET.SubElement(
                    body,
                    "geom",
                    name=f"{self.name}_panel_{index}",
                    type="box",
                    size=" ".join(map(str, panel.half_extents)),
                    pos=" ".join(map(str, panel.position)),
                    quat=f"{w} {x} {y} {z}",
                    rgba=self.rgba,
                    mass=str(float(self.mass) / len(panels)),
                )
                if self.solref is not None:
                    geom.set("solref", " ".join(map(str, self.solref)))
                if self.solimp is not None:
                    geom.set("solimp", " ".join(map(str, self.solimp)))
                if self.solref is not None or self.solimp is not None:
                    geom.set("priority", "1")
            return body

        def check_in_region(self, *args: Any, **kwargs: Any) -> bool:
            position, region_name, *rest = args
            env = rest[0] if rest else kwargs.get("env")
            if not super().check_in_region(position, region_name, env):
                return False
            if region_name != GOAL_REGION:
                return True
            original = self.env
            self.env = env if env is not None else self.env
            try:
                origin, wxyz = self._read_pose()
            finally:
                self.env = original
            return self.geometry().contains(
                point=position,
                bin_position=origin,
                bin_quaternion_xyzw=(wxyz[1], wxyz[2], wxyz[3], wxyz[0]),
            )

    @register_object(name="kitchen_recycling_proxy")
    class RecyclingProxy(Cuboid):
        def _create_xml_element(self) -> ET.Element:
            if self.options["kind"] in ("cube", "packaging"):
                return super()._create_xml_element()
            body = ET.Element("body", name=self.name)
            ET.SubElement(body, "freejoint", name=self.joint_name)
            radius, _, half_height = self.size
            bottle = self.options["kind"] == "bottle"
            cap_height = BOTTLE_CAP_HEIGHT if bottle else 0.002
            body_height = 2 * half_height - cap_height
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_body",
                type="cylinder",
                size=f"{radius} {body_height / 2}",
                pos=f"0 0 {-cap_height / 2}",
                rgba=self.rgba,
                mass=str(self.mass * 0.9),
                friction="0.8 0.01 0.001",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"{self.name}_cap",
                type="cylinder",
                size=f"{radius * (BOTTLE_CAP_RADIUS_FACTOR if bottle else 1)} {cap_height / 2}",
                pos=f"0 0 {half_height - cap_height / 2}",
                rgba="0.95 0.35 0.04 1" if bottle else "0.7 0.7 0.7 1",
                mass=str(self.mass * 0.1),
            )
            return body

    @register_fixture
    class KitchenMatVisual(FixedCuboid):
        @staticmethod
        def get_bounding_box_from_config(*args: Any) -> list[Any]:
            pos, _ = args
            return [*pos, *pos]

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            for geom in self.xml_element.iter("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")


def initial_positions(*, config: KitchenConfig, seed: int) -> list[tuple[float, float]]:
    if not config.scatter_objects:
        return [
            (0.58, (index - (len(config.objects) - 1) / 2) * 0.45)
            for index in range(len(config.objects))
        ]
    rng = np.random.default_rng(seed)
    lower = np.array(config.kitchen_bounds[:2]) + 0.4
    upper = np.array(config.kitchen_bounds[2:]) - 0.35
    lower[0] = max(lower[0], -0.9)
    if np.any(lower >= upper):
        raise ValueError("Kitchen has no clear area for scattered floor objects")
    positions: list[np.ndarray] = []
    for spec in config.objects:
        radius = float(np.linalg.norm(spec.dimensions)) / 2
        for _ in range(1000):
            position = rng.uniform(lower, upper)
            if np.linalg.norm(position) < config.robot_radius + radius + 0.15:
                continue
            if any(np.linalg.norm(position - previous) < 0.45 + radius for previous in positions):
                continue
            positions.append(position)
            break
        else:
            raise ValueError("Unable to scatter separated objects on the clear kitchen floor")
    return [cast(tuple[float, float], tuple(map(float, position))) for position in positions]


def scene_config(*, config: KitchenConfig, seed: int = 0) -> dict:
    left, bottom, right, top = config.mat_bounds
    # The far region, by scene seed (`bin_regions.py`), so evaluation tasks vary the bin.
    bin_start = KitchenBinRegions.scene_start(seed=seed)
    regions = {
        "robot_start": {"target": "ground", "ranges": [[0, 0, 0, 0]], "yaw_ranges": [[0, 0]]},
        "bin_start": {
            "target": "ground",
            "ranges": [[*bin_start, *bin_start]],
            "yaw_ranges": [[0, 0]],
        },
        "boundary_start": {
            "target": "ground",
            "ranges": [[left, 0, 0.001, left, 0, 0.001]],
            "yaw_ranges": [[0, 0]],
        },
        GOAL_REGION: {
            "target": "bin_0",
            "ranges": [list(TaperedBin.of(config=config).interior_box())],
            "yaw_ranges": [[0, 0]],
            "rgba": [0, 0, 0, 0],
        },
    }
    objects = {}
    initial = [
        ["on", "robot", "robot_start"],
        ["on", "bin_0", "bin_start"],
        ["on", "cuboid_barrier", "boundary_start"],
    ]
    colors = {
        "cube": [0.15, 0.7, 0.25, 1],
        "bottle": [0.55, 0.8, 0.86, 1],
        "can": [0.68, 0.69, 0.72, 1],
        "packaging": [0.65, 0.4, 0.15, 1],
    }
    positions = initial_positions(config=config, seed=seed)
    for index, spec in enumerate(config.objects):
        name = internal_name(index=index)
        horizontal, lateral = positions[index]
        regions[f"start_{index}"] = {
            "target": "ground",
            "ranges": [[horizontal, lateral, horizontal, lateral]],
            "yaw_ranges": [[0, 0]],
        }
        objects[name] = {
            "size": [dimension / 2 for dimension in spec.dimensions],
            "mass": spec.simulation_mass,
            "kind": spec.kind,
            "rgba": colors[spec.kind],
        }
        initial.append(["on", name, f"start_{index}"])
    return {
        "description": "Kitchen recycling preview; unmeasured rigid-body proxies, not hardware",
        "robots": {"tidybot": {"robot": {}}},
        "scene": "lab2",
        "room_layout": "tossing_room.xml",
        "convex_placement_room_body": "tossing_room",
        "observe_static_colliders": True,
        "regions": regions,
        "objects": {
            "kitchen_recycling_proxy": objects,
            "kitchen_tapered_bin": {
                "bin_0": {
                    "length": config.bin_outside[0],
                    "width": config.bin_outside[1],
                    "height": config.bin_outside[2],
                    "wall_thickness": config.simulation_wall,
                    "base_length": config.bin_base[0],
                    "base_width": config.bin_base[1],
                    "mass": config.simulation_bin_mass,
                    "rgba": [0.05, 0.3, 0.75, 1],
                    "fixed": False,
                    "solref": [0.001, 1],
                    "solimp": [0.99, 0.99, 0.001],
                }
            },
        },
        "fixtures": {
            "kitchenmatvisual": {
                "cuboid_barrier": {
                    "size": [0.008, (top - bottom) / 2, 0.0005],
                    "rgba": [0.95, 0.55, 0.08, 1],
                },
            }
        },
        "cameras": {
            "task_view": {
                "position": [1.0, 2.6, 4.4],
                "lookat": [0.65, 0.25, 0.25],
                "fovy": 60,
                "resolution": [960, 720],
            }
        },
        "initial_state": initial,
        # Drawn into the scene XML as a visual only (`furniture.py`): an object here
        # would sit in every planner's obstacle set.
        "kitchen_mats": [left, bottom, right, top],
        "goal_state": [
            ["on", internal_name(index=index), GOAL_REGION]
            for index, spec in enumerate(config.objects)
            if spec.name in config.required_objects
        ],
    }
