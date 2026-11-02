"""Separate scene specification and explicitly approximate recycling rigid bodies."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, cast

import numpy as np

from .domain import BOTTLE_CAP_HEIGHT, BOTTLE_CAP_RADIUS_FACTOR, KitchenConfig


def internal_name(*, index: int) -> str:
    """KINDER's collision adapter discovers movables by its legacy cube prefix."""
    return f"cube_{index}"


def register_simulation_objects() -> None:
    """Register new names only, leaving every original object class untouched."""
    from kinder.envs.dynamic3d.objects.base import register_fixture, register_object
    from kinder.envs.dynamic3d.objects.fixtures import FixedCuboid
    from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid

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
    regions = {
        "robot_start": {"target": "ground", "ranges": [[0, 0, 0, 0]], "yaw_ranges": [[0, 0]]},
        "bin_start": {
            "target": "ground",
            "ranges": [[*config.bin_position, *config.bin_position]],
            "yaw_ranges": [[0, 0]],
        },
        "mat_start": {
            "target": "ground",
            "ranges": [
                [
                    (left + right) / 2,
                    (bottom + top) / 2,
                    0.0006,
                    (left + right) / 2,
                    (bottom + top) / 2,
                    0.0006,
                ]
            ],
            "yaw_ranges": [[0, 0]],
        },
        "boundary_start": {
            "target": "ground",
            "ranges": [[left, 0, 0.001, left, 0, 0.001]],
            "yaw_ranges": [[0, 0]],
        },
        "bin_goal": {
            "target": "bin_0",
            "ranges": [[-0.1, -0.1, 0.01, 0.1, 0.1, 0.14]],
            "yaw_ranges": [[0, 0]],
            "rgba": [0, 0, 0, 0],
        },
    }
    objects = {}
    initial = [
        ["on", "robot", "robot_start"],
        ["on", "bin_0", "bin_start"],
        ["on", "cuboid_barrier", "boundary_start"],
        ["on", "sprout_mats", "mat_start"],
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
            "bin": {
                "bin_0": {
                    "length": config.bin_outside[0],
                    "width": config.bin_outside[1],
                    "height": config.bin_outside[2],
                    "wall_thickness": config.simulation_wall,
                    "mass": config.simulation_bin_mass,
                    "rgba": [0.8, 0.025, 0.035, 1],
                    "fixed": False,
                    "solref": [0.001, 1],
                    "solimp": [0.99, 0.99, 0.001],
                }
            },
        },
        "fixtures": {
            "kitchenmatvisual": {
                "sprout_mats": {
                    "size": [(right - left) / 2, (top - bottom) / 2, 0.0005],
                    "rgba": [0.45, 0.55, 0.58, 1],
                },
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
        "goal_state": [
            ["on", internal_name(index=index), "bin_goal"]
            for index, spec in enumerate(config.objects)
            if spec.name in config.required_objects
        ],
    }
