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
    assert config.serving_target == "corelle_plate"
    assert config.plate_radius == pytest.approx(0.13)
    assert config.tray_center_x == pytest.approx(1.3725 - 0.010 - 0.0254 * 10.25 / 2)
    assert config.rack_top_z == pytest.approx(0.925)
    assert config.tray_top_z == pytest.approx(0.907)
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
        {"serving_target": "bowl"},
        {"corelle_well_radius": 0.14},
        {"corelle_height": 0.03},
        {"corelle_foot_height": 0.010, "corelle_glass_thickness": 0.010},
        {"corelle_foot_radius": 0.096},
        {"plate_radius": 0},
        {"serving_target": "round_plate", "plate_radius": 0.6},
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
        serving_target="round_plate",
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


def test_round_plate_sits_centred_in_front_of_the_rack_exit() -> None:
    config = RackServingConfig(serving_target="round_plate")
    assert (config.serving_depth, config.serving_width) == pytest.approx((0.26, 0.26))
    assert config.tray_back_x == pytest.approx(config.rack_front_x - config.gap)
    assert config.tray_front_x == pytest.approx(config.tray_back_x - 0.26)
    assert config.tray_center_y == config.rack_center_y
    assert config.tray_top_z < config.rack_top_z
    assert config.tray_mass == 1.0


def test_slate_remains_selectable_with_its_original_geometry() -> None:
    slate = RackServingConfig(serving_target="slate")
    assert (slate.serving_depth, slate.serving_width) == (0.30, 0.375)
    assert slate.tray_center_x == pytest.approx(1.2125)


@pytest.mark.parametrize(
    ("target", "kind"),
    [
        ("round_plate", "rack_serving_round_plate"),
        ("slate", "rack_serving_slate"),
        ("lipped_platter", "rack_serving_lipped_platter"),
        ("corelle_plate", "rack_serving_corelle_plate"),
    ],
)
def test_scene_uses_the_selected_serving_object(*, target, kind) -> None:
    config = RackServingConfig(serving_target=target)
    scene = build_scene(config=config)
    assert list(scene["objects"][kind]) == ["box_0"]
    other = {
        "rack_serving_round_plate",
        "rack_serving_slate",
        "rack_serving_lipped_platter",
        "rack_serving_corelle_plate",
    } - {kind}
    assert not other & set(scene["objects"])
    goal = scene["regions"]["serving_goal"]["ranges"][0]
    assert goal[3] - goal[0] == pytest.approx(config.serving_depth)
    assert goal[4] - goal[1] == pytest.approx(config.serving_width)


def test_lipped_platter_is_wide_low_heavy_and_on_the_slate_centre_line() -> None:
    platter = RackServingConfig(serving_target="lipped_platter")
    slate = RackServingConfig(serving_target="slate")
    assert (platter.serving_depth, platter.serving_width) == pytest.approx((0.30, 0.45))
    assert platter.tray_back_x == pytest.approx(platter.rack_front_x - 0.010)
    assert platter.tray_back_x == pytest.approx(slate.tray_back_x)
    assert platter.tray_center_x == pytest.approx(slate.tray_center_x)
    assert platter.tray_center_y == slate.tray_center_y == platter.rack_center_y
    assert platter.platter_base_thickness == pytest.approx(0.006)
    assert platter.platter_lip_height == pytest.approx(0.012)
    assert platter.platter_lip_wall == pytest.approx(0.008)
    assert platter.serving_mass == pytest.approx(1.5)
    assert slate.serving_mass == pytest.approx(slate.tray_mass)
    assert platter.tray_top_z == pytest.approx(platter.counter_height + 0.006)
    assert platter.platter_lip_top_z == pytest.approx(platter.counter_height + 0.018)
    assert platter.platter_lip_top_z < platter.rack_top_z


def test_lipped_platter_lip_must_stay_below_the_rack_top() -> None:
    with pytest.raises(ValidationError):
        RackServingConfig(serving_target="lipped_platter", platter_lip_height=0.025)


def test_lipped_platter_scene_uses_its_mass_and_floor_goal() -> None:
    config = RackServingConfig(serving_target="lipped_platter")
    options = build_scene(config=config)["objects"]["rack_serving_lipped_platter"]["box_0"]
    assert options["mass"] == pytest.approx(1.5)
    assert options["friction"] == list(config.tray_friction)
    assert options["base_thickness"] == pytest.approx(0.006)
    assert options["lip_height"] == pytest.approx(0.012)
    assert options["lip_wall"] == pytest.approx(0.008)
    goal = build_scene(config=config)["regions"]["serving_goal"]["ranges"][0]
    assert goal[2] == pytest.approx(config.counter_height + 0.006)


INCH = 0.0254


def test_corelle_plate_carries_the_published_dimensions() -> None:
    """Corelle Winter Frost White 10-1/4 in dinner plate (KC35285): 10.25 x 0.75 in."""
    plate = RackServingConfig(serving_target="corelle_plate")
    assert plate.corelle_diameter == pytest.approx(10.25 * INCH)
    assert plate.corelle_height == pytest.approx(0.75 * INCH)
    assert (plate.serving_depth, plate.serving_width) == pytest.approx((10.25 * INCH,) * 2)
    assert plate.tray_back_x == pytest.approx(plate.rack_front_x - plate.gap)
    assert plate.tray_center_y == plate.rack_center_y
    assert plate.corelle_rim_top_z == pytest.approx(plate.counter_height + 0.75 * INCH)
    assert plate.corelle_rim_top_z < plate.rack_top_z


@pytest.mark.parametrize(
    "field",
    [
        "corelle_well_radius",
        "corelle_glass_thickness",
        "corelle_foot_height",
        "corelle_foot_radius",
        "corelle_foot_width",
        "corelle_mass",
    ],
)
def test_every_estimated_corelle_number_is_marked_unverified(*, field: str) -> None:
    assert "UNVERIFIED" in RackServingConfig.model_fields[field].description


def test_corelle_food_rests_on_the_well_above_the_foot_ring() -> None:
    plate = RackServingConfig(serving_target="corelle_plate")
    well_top = plate.counter_height + plate.corelle_foot_height + plate.corelle_glass_thickness
    assert plate.tray_top_z == pytest.approx(well_top)
    assert plate.serving_thickness == pytest.approx(well_top - plate.counter_height)
    assert plate.serving_mass == pytest.approx(plate.corelle_mass)
    assert plate.corelle_foot_radius < plate.corelle_well_radius < plate.corelle_diameter / 2
    assert plate.tray_top_z < plate.corelle_rim_top_z


def test_corelle_scene_options_carry_the_profile() -> None:
    config = RackServingConfig(serving_target="corelle_plate")
    scene = build_scene(config=config)
    options = scene["objects"]["rack_serving_corelle_plate"]["box_0"]
    assert options["mass"] == pytest.approx(config.corelle_mass)
    assert options["friction"] == list(config.tray_friction)
    for key in (
        "corelle_diameter",
        "corelle_height",
        "corelle_well_radius",
        "corelle_glass_thickness",
        "corelle_foot_height",
        "corelle_foot_radius",
        "corelle_foot_width",
        "corelle_rim_segments",
    ):
        assert options[key] == pytest.approx(getattr(config, key))
    goal = scene["regions"]["serving_goal"]["ranges"][0]
    assert goal[2] == pytest.approx(config.tray_top_z)
