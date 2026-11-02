"""Offline metric-layout and hardware-assumption checks."""

import json

import pytest
from pydantic import ValidationError

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.scene import build_scene


def test_default_layout_and_semantics() -> None:
    config = RackServingConfig()
    assert config.surface_kind == "wire"
    assert config.mode == "training"
    assert (config.cooling_seconds, config.initial_elapsed_seconds, config.wait_seconds) == (
        300,
        0,
        30,
    )
    assert config.human_interruption_cost == 5
    assert config.rack_front_x == pytest.approx(1.3725)
    assert config.rack_back_x == pytest.approx(1.6275)
    assert config.tray_center_x == pytest.approx(1.2125)
    assert config.rack_top_z == pytest.approx(0.925)
    assert config.tray_top_z == pytest.approx(0.912)
    assert config.rack_front_x - config.tray_back_x == pytest.approx(0.010)
    assert config.counter_x_min <= config.tray_front_x < config.rack_back_x <= config.counter_x_max
    assert config.counter_front_x == config.counter_x_min
    assert config.counter_y_min == -0.5
    assert config.counter_y_max == 0.5
    assert (config.rack_width, config.rack_depth, config.rack_mass) == (0.375, 0.255, 0.5)
    assert (config.scraper_width, config.scraper_blade_height, config.scraper_mass) == (
        0.25,
        0.12,
        0.275,
    )
    assert "UNVERIFIED" in RackServingConfig.model_fields["rack_height"].description
    assert "UNVERIFIED" in RackServingConfig.model_fields["rack_mass"].description


def test_config_is_frozen_and_json_roundtrips() -> None:
    config = RackServingConfig(
        surface_kind="concept", mode="deployment", initial_elapsed_seconds=300
    )
    assert RackServingConfig.model_validate_json(config.model_dump_json()) == config
    with pytest.raises(ValidationError):
        config.rack_mass = 2
    assert isinstance(config.rack_friction, tuple)


@pytest.mark.parametrize(
    "overrides",
    [
        {"surface_kind": "plate"},
        {"mode": "test"},
        {"rack_height": 0.01},
        {"rack_mass": 0},
        {"rack_mass": float("inf")},
        {"rack_center_x": float("nan")},
        {"tray_thickness": -0.1},
        {"gap": -0.01},
        {"gap": 1},
        {"cooling_seconds": -1},
        {"initial_elapsed_seconds": -1},
        {"wait_seconds": 0},
        {"human_interruption_cost": 0},
        {"human_interruption_cost": float("inf")},
        {"counter_x_max": 1.4},
        {"counter_width": 0.1},
        {"rack_friction": [-1, 0, 0]},
        {"rack_friction": [0.5, float("nan"), 0]},
        {"rack_wire_radius": 0.02},
        {"concept_web_width": 0.03},
        {"scraper_blade_thickness": 0.02},
        {"unknown_hardware_option": 1},
    ],
)
def test_invalid_assumptions_are_rejected(*, overrides: dict) -> None:
    with pytest.raises(ValidationError):
        RackServingConfig(**overrides)


def test_layout_is_derived_from_config_not_hardcoded() -> None:
    config = RackServingConfig(
        counter_height=0.85,
        rack_height=0.04,
        tray_thickness=0.02,
        gap=0.02,
        rack_center_y=0.02,
        rack_mass=0.7,
        rack_friction=(0.2, 0.001, 0.0001),
    )
    assert config.rack_top_z == pytest.approx(0.89)
    assert config.tray_top_z == pytest.approx(0.87)
    assert config.tray_center_y == 0.02
    assert config.rack_front_x - config.tray_back_x == pytest.approx(0.02)
    scene = build_scene(config=config)
    assert json.loads(json.dumps(scene)) == scene
    assert scene["regions"]["rack_init"]["ranges"][0][:3] == [1.5, 0.02, 0.85]
    second_scene = build_scene(config=config)
    scene["regions"].clear()
    assert second_scene["regions"]
