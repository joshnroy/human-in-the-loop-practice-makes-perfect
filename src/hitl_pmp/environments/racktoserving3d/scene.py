"""Build a fresh KINDER SweepOffShelf3D scene without changing its defaults.

All initial object regions use absolute world coordinates (ground regions with
explicit z). Shelf indices are compatibility metadata only; the backend must
measure live rack/tray geometry for shelf and serving predicates. Cooling and
training/deployment semantics belong to the backend, not MuJoCo materials.
"""

from typing import Any

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig

COUNTER_TYPE = "rackservingcounter"
RACK_TYPE = "rack_serving_rack"
TRAY_TYPE = "rack_serving_slate"
SCRAPER_TYPE = "rack_serving_scraper"
DUMPLING_TYPE = "rack_serving_dumpling"


def build_scene(*, config: RackServingConfig) -> dict[str, Any]:
    """Return JSON-serializable scene data; call install_geometry before make()."""
    regions: dict[str, Any] = {}

    def point_region(*, name: str, position: tuple[float, float, float]) -> str:
        regions[name] = {
            "target": "ground",
            "ranges": [[*position, *position]],
            "yaw_ranges": [[0, 0]],
        }
        return name

    initial = [
        [
            "on",
            "cupboard_1",
            point_region(name="counter_init", position=(config.counter_center_x, 0, 0)),
        ],
        ["on", "robot", point_region(name="robot_init", position=(0, 0, 0))],
        [
            "on",
            "rack_0",
            point_region(
                name="rack_init",
                position=(config.rack_center_x, config.rack_center_y, config.counter_height),
            ),
        ],
        [
            "on",
            "box_0",
            point_region(
                name="tray_init",
                position=(config.tray_center_x, config.tray_center_y, config.counter_height),
            ),
        ],
        [
            "on",
            "wiper_0",
            point_region(
                name="scraper_init",
                position=(
                    config.rack_center_x,
                    config.counter_y_min + config.scraper_width / 2 + 0.025,
                    config.counter_height,
                ),
            ),
        ],
    ]
    dumplings = {}
    for index, (depth_fraction, width_fraction) in enumerate((
        (-0.22, -0.25),
        (-0.22, 0.25),
        (0, 0),
        (0.22, -0.25),
        (0.22, 0.25),
    )):
        name = f"cube_{index}"
        dumplings[name] = {
            "size": [
                config.dumpling_depth / 2,
                config.dumpling_width / 2,
                config.dumpling_height / 2,
            ],
            "mass": config.dumpling_mass,
            "friction": list(config.dumpling_friction),
            "contact_dimension": config.dumpling_contact_dimension,
            "rgba": [0.88, 0.78, 0.59, 1],
        }
        region = point_region(
            name=f"{name}_init",
            position=(
                config.rack_center_x + depth_fraction * config.rack_depth,
                config.rack_center_y + width_fraction * config.rack_width,
                config.rack_top_z + config.dumpling_height / 2 + 0.001,
            ),
        )
        initial.append(["on", name, region])
    regions["serving_goal"] = {
        "target": "ground",
        "ranges": [
            [
                config.tray_front_x,
                config.tray_center_y - config.tray_width / 2,
                config.tray_top_z,
                config.tray_back_x,
                config.tray_center_y + config.tray_width / 2,
                config.tray_top_z + config.dumpling_height,
            ]
        ],
        "yaw_ranges": [[0, 0]],
    }
    return {
        "description": (
            "Movable cooling rack to movable flat slate tray; rigid dumpling proxies. "
            "Hardware height, mass, friction and cooling are unverified assumptions."
        ),
        "robots": {"tidybot": {"robot": {}}},
        "scene": "lab2",
        "observe_static_colliders": True,
        "fixtures": {
            COUNTER_TYPE: {
                "cupboard_1": {
                    "length": config.counter_depth,
                    "depth": config.counter_width,
                    "shelf_heights": [config.rack_height, config.rack_height],
                    "shelf_partitions": [[], []],
                    "shelf_thickness": 0.001,
                    "side_and_back_open": True,
                    "rack_serving_config": config.model_dump(mode="json"),
                }
            }
        },
        "objects": {
            RACK_TYPE: {
                "rack_0": {
                    "size": [config.rack_depth / 2, config.rack_width / 2, config.rack_height / 2],
                    "mass": config.rack_mass,
                    "rack_serving_config": config.model_dump(mode="json"),
                }
            },
            TRAY_TYPE: {
                "box_0": {
                    "size": [
                        config.tray_depth / 2,
                        config.tray_width / 2,
                        config.tray_thickness / 2,
                    ],
                    "length": config.tray_depth,
                    "width": config.tray_width,
                    "height": config.tray_thickness,
                    "wall_thickness": 0.0,
                    "mass": config.tray_mass,
                    "friction": list(config.tray_friction),
                    "contact_dimension": config.support_contact_dimension,
                    "rgba": [0.16, 0.18, 0.19, 1],
                }
            },
            SCRAPER_TYPE: {
                "wiper_0": {
                    "head_length": config.scraper_width,
                    "head_height": config.scraper_blade_height,
                    "blade_thickness": config.scraper_blade_thickness,
                    "handle_width": config.scraper_handle_width,
                    "handle_height": config.scraper_handle_height,
                    "upright": True,
                    "mass": config.scraper_mass,
                    "friction": list(config.scraper_friction),
                    "blade_friction": list(config.scraper_blade_friction),
                    "handle_rgba": [0.12, 0.14, 0.15, 1],
                    "head_rgba": [0.7, 0.73, 0.76, 1],
                }
            },
            DUMPLING_TYPE: dumplings,
        },
        "regions": regions,
        "initial_state": initial,
        "goal_state": [["on", f"cube_{index}", "serving_goal"] for index in range(5)],
        "cameras": {
            "task_view": {
                "position": [0.25, -1.8, 1.8],
                "lookat": [config.counter_center_x, 0, config.rack_top_z],
                "fovy": 45,
                "resolution": [640, 480],
            }
        },
    }
