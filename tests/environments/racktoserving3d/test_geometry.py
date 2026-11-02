"""Native XML/registration tests, without launching a robot or stepping physics."""

import copy
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.geometry import install_geometry
from hitl_pmp.environments.racktoserving3d.scene import (
    COUNTER_TYPE,
    DUMPLING_TYPE,
    RACK_TYPE,
    SCRAPER_TYPE,
    TRAY_TYPE,
    build_scene,
)


@pytest.fixture
def registry() -> Any:
    pytest.importorskip("kinder")
    from kinder.envs.dynamic3d.objects import base

    install_geometry()
    return base


def test_registration_is_isolated_and_idempotent() -> None:
    pytest.importorskip("kinder")
    from kinder.envs.dynamic3d.objects import base as registry
    from kinder.envs.dynamic3d.objects.fixtures import Cupboard
    from kinder.envs.dynamic3d.objects.primitive_objects import Cuboid, Wiper

    originals = (
        Cupboard._create_xml_element,
        Wiper._create_xml_element,
        Cuboid._create_xml_element,
    )
    objects_before = dict(registry.REGISTERED_OBJECTS)
    fixtures_before = dict(registry.REGISTERED_FIXTURES)
    install_geometry()
    assert all(registry.REGISTERED_OBJECTS[name] is kind for name, kind in objects_before.items())
    assert all(registry.REGISTERED_FIXTURES[name] is kind for name, kind in fixtures_before.items())
    registered_objects = dict(registry.REGISTERED_OBJECTS)
    registered_fixtures = dict(registry.REGISTERED_FIXTURES)
    install_geometry()
    assert registered_objects == registry.REGISTERED_OBJECTS
    assert registered_fixtures == registry.REGISTERED_FIXTURES
    assert registry.REGISTERED_OBJECTS["wiper"] is Wiper
    assert registry.REGISTERED_FIXTURES["cupboard"] is Cupboard
    assert originals == (
        Cupboard._create_xml_element,
        Wiper._create_xml_element,
        Cuboid._create_xml_element,
    )


@pytest.mark.parametrize("surface_kind", ["wire", "concept"])
def test_counter_and_free_objects_compile_without_hidden_support(
    *, registry: Any, surface_kind: str
) -> None:
    import mujoco

    config = RackServingConfig(surface_kind=surface_kind)
    scene = build_scene(config=config)
    fixture_options = scene["fixtures"][COUNTER_TYPE]["cupboard_1"]
    counter = registry.REGISTERED_FIXTURES[COUNTER_TYPE](
        name="cupboard_1",
        fixture_config=fixture_options,
        position=[config.counter_center_x, 0, 0],
        yaw=0,
    )
    counter_geoms = counter.xml_element.findall("geom")
    assert len(counter_geoms) == 5
    assert all("shelf" not in geom.get("name") for geom in counter_geoms)
    assert counter.xml_element.find("freejoint") is None
    assert counter._shelf_z_positions == [
        config.counter_height,
        config.rack_top_z,
        config.counter_height,
    ]
    assert {site.get("name") for site in counter.xml_element.findall("site")} == {
        "cupboard_1_shelf1",
        "cupboard_1_shelf2",
    }
    fixture_bounds = counter.get_bounding_box_from_config(
        np.array(counter.position), fixture_options
    )
    assert fixture_bounds == pytest.approx([0.98, -0.5, 0, 1.70, 0.5, 0.9])

    root = ET.Element("mujoco")
    world = ET.SubElement(root, "worldbody")
    world.append(counter.xml_element)
    objects = {}
    for kind, named_options in scene["objects"].items():
        for name, options in named_options.items():
            instance = registry.REGISTERED_OBJECTS[kind](name=name, options=options)
            objects[name] = instance
            region_name = next(entry[2] for entry in scene["initial_state"] if entry[1] == name)
            position = scene["regions"][region_name]["ranges"][0][:3]
            body = copy.deepcopy(instance.xml_element)
            body.set("pos", " ".join(map(str, position)))
            world.append(body)
            assert body.find("freejoint") is not None
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.njnt == 8
    assert model.body("rack_0").mass == pytest.approx(config.rack_mass)
    assert model.body("box_0").mass == pytest.approx(config.tray_mass)
    assert model.body("wiper_0").mass == pytest.approx(config.scraper_mass)
    assert model.geom("cupboard_1_counter").size[2] + data.geom("cupboard_1_counter").xpos[
        2
    ] == pytest.approx(config.counter_height)

    rack = objects["rack_0"]
    rack_geoms = rack.xml_element.findall("geom")
    assert all(geom.get("friction") == "0.6 0.005 0.003" for geom in rack_geoms)
    assert all(geom.get("condim") == "6" for geom in rack_geoms)
    assert int(model.geom_condim[model.geom("box_0_slate").id]) == 6
    assert len([geom for geom in rack_geoms if "foot" in geom.get("name")]) == 4
    surfaces = [geom for geom in rack_geoms if "foot" not in geom.get("name")]
    assert len(surfaces) > 4
    assert {geom.get("type") for geom in surfaces} == (
        {"capsule"} if surface_kind == "wire" else {"box"}
    )
    for geom in surfaces:
        if surface_kind == "wire":
            ends = np.fromstring(geom.get("fromto"), sep=" ").reshape(2, 3)
            radius = float(geom.get("size"))
            assert max(ends[:, 2]) + radius == pytest.approx(config.rack_height)
            assert min(ends[:, 0]) - radius >= -config.rack_depth / 2 - 1e-9
            assert max(ends[:, 1]) + radius <= config.rack_width / 2 + 1e-9
        else:
            size = np.fromstring(geom.get("size"), sep=" ")
            pos = np.fromstring(geom.get("pos"), sep=" ")
            assert pos[2] + size[2] == pytest.approx(config.rack_height)
            assert min(size[:2]) == pytest.approx(config.concept_web_width / 2)
    assert rack.get_bounding_box_from_config(
        np.array([1.5, 0, 0.9]), scene["objects"][RACK_TYPE]["rack_0"]
    ) == pytest.approx([1.3725, -0.1875, 0.9, 1.6275, 0.1875, 0.925])

    tray = objects["box_0"]
    assert len(tray.xml_element.findall("geom")) == 1
    assert tray.wall_thickness == 0
    assert tray.fixed is False
    assert tray.height == config.tray_thickness
    assert data.geom("box_0_slate").xpos[2] + model.geom("box_0_slate").size[2] == pytest.approx(
        config.tray_top_z
    )
    scraper = objects["wiper_0"]
    handle, blade = scraper.xml_element.findall("geom")
    handle_size = np.fromstring(handle.get("size"), sep=" ")
    blade_size = np.fromstring(blade.get("size"), sep=" ")
    assert handle_size[0] == blade_size[0]
    assert handle_size[1] > blade_size[1]
    assert 2 * blade_size[0] == config.scraper_width
    assert 2 * blade_size[1] == config.scraper_blade_thickness
    assert blade.get("priority") == "1"
    assert blade.get("friction") == "0.25 0.005 0.0001"
    assert handle.get("priority", "0") == "0"
    assert scraper.get_bounding_box_dimensions() == pytest.approx((0.25, 0.018, 0.14))
    assert scraper.get_bounding_box_from_config(
        np.zeros(3), scene["objects"][SCRAPER_TYPE]["wiper_0"]
    ) == pytest.approx([-0.125, -0.009, 0, 0.125, 0.009, 0.14])
    assert objects["cube_0"].xml_element.find("geom").get("type") == "ellipsoid"
    food = model.geom("cube_0_dumpling")
    assert int(model.geom_condim[food.id]) == 6
    assert food.friction[2] == pytest.approx(0.003)
    assert set(objects) == {"rack_0", "box_0", "wiper_0", *(f"cube_{index}" for index in range(5))}


def test_scene_initial_poses_pass_native_sampler(*, registry: Any) -> None:
    from kinder.envs.dynamic3d.objects.base import MujocoGround
    from kinder.envs.dynamic3d.placement_samplers import sample_collision_free_positions

    scene = build_scene(config=RackServingConfig())
    ground = MujocoGround(regions=scene["regions"])
    regions = {
        name: region
        for _, name, region in scene["initial_state"]
        if name not in {"cupboard_1", "robot"}
    }
    poses = sample_collision_free_positions(
        scene["objects"],
        np.random.default_rng(42),
        entity_region_names=regions,
        entity_pos_yaw_samplers={name: ground.sample_pose_in_region for name in regions},
        fail_on_exhaustion=True,
    )
    for kind, objects in poses.items():
        for name, pose in objects.items():
            assert pose["position"] == pytest.approx(
                scene["regions"][regions[name]]["ranges"][0][:3]
            )
            assert kind in {RACK_TYPE, TRAY_TYPE, SCRAPER_TYPE, DUMPLING_TYPE}


def test_custom_mass_and_friction_reach_native_geometry(*, registry: Any) -> None:
    config = RackServingConfig(rack_mass=0.8, rack_height=0.035, rack_friction=(0.2, 0.003, 0.0002))
    scene = build_scene(config=config)
    rack = registry.REGISTERED_OBJECTS[RACK_TYPE](
        name="rack_0", options=scene["objects"][RACK_TYPE]["rack_0"]
    )
    geoms = rack.xml_element.findall("geom")
    assert sum(float(geom.get("mass")) for geom in geoms) == pytest.approx(0.8)
    assert all(geom.get("friction") == "0.2 0.003 0.0002" for geom in geoms)
