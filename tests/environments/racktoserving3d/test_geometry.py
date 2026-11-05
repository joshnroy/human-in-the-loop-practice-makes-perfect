"""Native XML/registration tests, without launching a robot or stepping physics."""

import copy
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.geometry import install_geometry
from hitl_pmp.environments.racktoserving3d.potsticker import RackPotsticker
from hitl_pmp.environments.racktoserving3d.scene import (
    CORELLE_TYPE,
    COUNTER_TYPE,
    DUMPLING_TYPE,
    RACK_TYPE,
    SCRAPER_TYPE,
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


@pytest.mark.parametrize("serving_target", ["round_plate", "slate"])
@pytest.mark.parametrize("surface_kind", ["wire", "concept"])
def test_counter_and_free_objects_compile_without_hidden_support(
    *, registry: Any, surface_kind: str, serving_target: str
) -> None:
    import mujoco

    config = RackServingConfig(surface_kind=surface_kind, serving_target=serving_target)
    serving_geom = "box_0_plate" if serving_target == "round_plate" else "box_0_slate"
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
    ET.SubElement(root, "asset").extend(RackPotsticker.assets())
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
    assert int(model.geom_condim[model.geom(serving_geom).id]) == 6
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
    serving = model.geom(serving_geom)
    if serving_target == "round_plate":
        assert int(model.geom_type[serving.id]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
        assert serving.size[0] == pytest.approx(config.plate_radius)
        assert data.geom(serving_geom).xpos[2] + serving.size[1] == pytest.approx(config.tray_top_z)
        assert data.geom(serving_geom).xpos[:2] == pytest.approx([
            config.tray_center_x,
            config.tray_center_y,
        ])
        assert config.tray_center_x + serving.size[0] == pytest.approx(
            config.rack_front_x - config.gap
        )
    else:
        assert int(model.geom_type[serving.id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        assert data.geom(serving_geom).xpos[2] + serving.size[2] == pytest.approx(config.tray_top_z)
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
    assert objects["cube_0"].xml_element.find("geom").get("type") == "mesh"
    food = model.geom("cube_0_part_0")
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
            assert kind in {RACK_TYPE, CORELLE_TYPE, SCRAPER_TYPE, DUMPLING_TYPE}


def test_custom_mass_and_friction_reach_native_geometry(*, registry: Any) -> None:
    config = RackServingConfig(rack_mass=0.8, rack_height=0.035, rack_friction=(0.2, 0.003, 0.0002))
    scene = build_scene(config=config)
    rack = registry.REGISTERED_OBJECTS[RACK_TYPE](
        name="rack_0", options=scene["objects"][RACK_TYPE]["rack_0"]
    )
    geoms = rack.xml_element.findall("geom")
    assert sum(float(geom.get("mass")) for geom in geoms) == pytest.approx(0.8)
    assert all(geom.get("friction") == "0.2 0.003 0.0002" for geom in geoms)


def _compiled_scene(*, registry: Any, config: RackServingConfig) -> tuple[Any, Any]:
    """Compile the counter and every free object at its initial region, as the scene does."""
    import mujoco

    scene = build_scene(config=config)
    counter = registry.REGISTERED_FIXTURES[COUNTER_TYPE](
        name="cupboard_1",
        fixture_config=scene["fixtures"][COUNTER_TYPE]["cupboard_1"],
        position=[config.counter_center_x, 0, 0],
        yaw=0,
    )
    root = ET.Element("mujoco")
    ET.SubElement(root, "asset").extend(RackPotsticker.assets())
    world = ET.SubElement(root, "worldbody")
    world.append(counter.xml_element)
    for kind, named_options in scene["objects"].items():
        for name, options in named_options.items():
            instance = registry.REGISTERED_OBJECTS[kind](name=name, options=options)
            region_name = next(entry[2] for entry in scene["initial_state"] if entry[1] == name)
            body = copy.deepcopy(instance.xml_element)
            body.set("pos", " ".join(map(str, scene["regions"][region_name]["ranges"][0][:3])))
            world.append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _world_top(*, model: Any, data: Any, geoms: list[int]) -> float:
    return max(
        float(
            data.geom_xpos[geom][2]
            + np.abs(data.geom_xmat[geom].reshape(3, 3)[2])
            @ (
                model.geom_size[geom]
                if int(model.geom_type[geom]) == 6
                else np.array([model.geom_size[geom][0]] * 2 + [model.geom_size[geom][1]])
            )
        )
        for geom in geoms
    )


def test_lipped_platter_compiles_with_a_continuous_lip_below_the_rack_top(*, registry: Any) -> None:
    import mujoco

    config = RackServingConfig(serving_target="lipped_platter")
    model, data = _compiled_scene(registry=registry, config=config)
    platter = model.body("box_0").id
    assert model.body_mass[platter] == pytest.approx(1.5)
    assert int(model.jnt_type[model.body_jntadr[platter]]) == int(mujoco.mjtJoint.mjJNT_FREE)
    geoms = [int(geom) for geom in np.flatnonzero(model.geom_bodyid == platter)]
    names = {model.geom(geom).name for geom in geoms}
    assert names == {
        "box_0_platter_floor",
        "box_0_lip_front",
        "box_0_lip_back",
        "box_0_lip_left",
        "box_0_lip_right",
    }
    assert all(int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_BOX) for geom in geoms)
    assert all(model.geom_friction[geom] == pytest.approx(config.tray_friction) for geom in geoms)
    assert all(int(model.geom_condim[geom]) == 6 for geom in geoms)

    floor = model.geom("box_0_platter_floor").id
    lips = [model.geom(f"box_0_lip_{side}").id for side in ("front", "back", "left", "right")]
    floor_top = _world_top(model=model, data=data, geoms=[floor])
    lip_top = _world_top(model=model, data=data, geoms=lips)
    rack_geoms = [int(g) for g in np.flatnonzero(model.geom_bodyid == model.body("rack_0").id)]
    rack_top = max(
        float(data.geom_xpos[g][2] + model.geom_size[g][0])
        for g in rack_geoms
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    )
    assert floor_top - config.counter_height == pytest.approx(0.006)
    assert lip_top - floor_top == pytest.approx(0.012)
    assert lip_top < rack_top
    assert rack_top == pytest.approx(config.rack_top_z)

    lower = np.min([data.geom_xpos[g] - model.geom_size[g] for g in [floor, *lips]], axis=0)
    upper = np.max([data.geom_xpos[g] + model.geom_size[g] for g in [floor, *lips]], axis=0)
    assert upper[0] - lower[0] == pytest.approx(0.30)
    assert upper[1] - lower[1] == pytest.approx(0.45)
    assert config.rack_front_x - upper[0] == pytest.approx(0.010)
    assert (lower[1] + upper[1]) / 2 == pytest.approx(config.rack_center_y)
    for geom in lips:
        assert min(model.geom_size[geom][:2]) * 2 == pytest.approx(0.008)
        assert data.geom_xpos[geom][2] - model.geom_size[geom][2] == pytest.approx(floor_top)
    # The walls close a ring: each long wall spans the full outer length of its side.
    front, back, left, right = (model.geom_size[g] for g in lips)
    assert 2 * left[0] == pytest.approx(0.30) and 2 * right[0] == pytest.approx(0.30)
    assert 2 * front[1] + 0.0 >= 0.45 - 2 * 0.008 - 1e-9
    assert 2 * back[1] >= 0.45 - 2 * 0.008 - 1e-9


def _box_top_face(*, model: Any, data: Any, geom: int) -> np.ndarray:
    """World points of a box geom's local +z face corners."""
    rotation = data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    return np.array([
        data.geom_xpos[geom] + rotation @ (np.array([sx, sy, 1]) * size)
        for sx in (-1, 1)
        for sy in (-1, 1)
    ])


def test_corelle_plate_compiles_to_a_well_a_rising_rim_and_a_foot_ring(*, registry: Any) -> None:
    import mujoco

    config = RackServingConfig(serving_target="corelle_plate")
    model, data = _compiled_scene(registry=registry, config=config)
    plate = model.body("box_0").id
    assert model.body_mass[plate] == pytest.approx(config.corelle_mass)
    assert int(model.jnt_type[model.body_jntadr[plate]]) == int(mujoco.mjtJoint.mjJNT_FREE)
    geoms = [int(geom) for geom in np.flatnonzero(model.geom_bodyid == plate)]
    names = [model.geom(geom).name for geom in geoms]
    rims = [g for g, n in zip(geoms, names, strict=True) if n.startswith("box_0_rim_")]
    feet = [g for g, n in zip(geoms, names, strict=True) if n.startswith("box_0_foot_")]
    assert len(rims) == config.corelle_rim_segments
    assert len(feet) >= 12
    assert set(names) == {"box_0_well"} | {model.geom(g).name for g in rims + feet}
    assert all(model.geom_friction[g] == pytest.approx(config.tray_friction) for g in geoms)
    assert all(int(model.geom_condim[g]) == 6 for g in geoms)

    centre = np.array([config.tray_center_x, config.tray_center_y])
    well = model.geom("box_0_well").id
    assert int(model.geom_type[well]) == int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    assert model.geom_size[well][0] == pytest.approx(config.corelle_well_radius)
    assert 2 * model.geom_size[well][1] == pytest.approx(config.corelle_glass_thickness)
    assert data.geom_xpos[well][:2] == pytest.approx(centre)
    well_top = data.geom_xpos[well][2] + model.geom_size[well][1]
    assert well_top == pytest.approx(config.tray_top_z)
    assert data.geom_xpos[well][2] - model.geom_size[well][1] == pytest.approx(
        config.counter_height + config.corelle_foot_height
    )

    outer = config.corelle_diameter / 2
    for rim in rims:
        top = _box_top_face(model=model, data=data, geom=rim)
        radial = np.linalg.norm(top[:, :2] - centre, axis=1)
        inner_edge, outer_edge = top[radial < radial.mean()], top[radial > radial.mean()]
        # The rim's top face rises from the well's top to the published height at the edge.
        assert inner_edge[:, 2] == pytest.approx(well_top, abs=1e-6)
        assert outer_edge[:, 2] == pytest.approx(config.corelle_rim_top_z, abs=1e-6)
        assert np.linalg.norm(inner_edge[:, :2] - centre, axis=1) == pytest.approx(
            config.corelle_well_radius, abs=0.002
        )
        assert np.linalg.norm(outer_edge[:, :2] - centre, axis=1).max() <= outer + 1e-3
    rim_points = np.concatenate([_box_top_face(model=model, data=data, geom=rim) for rim in rims])
    assert np.linalg.norm(rim_points[:, :2] - centre, axis=1).max() == pytest.approx(
        outer, abs=1e-3
    )
    for foot in feet:
        assert data.geom_xpos[foot][2] - model.geom_size[foot][2] == pytest.approx(
            config.counter_height
        )
        assert data.geom_xpos[foot][2] + model.geom_size[foot][2] == pytest.approx(
            config.counter_height + config.corelle_foot_height
        )
        assert np.linalg.norm(data.geom_xpos[foot][:2] - centre) < config.corelle_well_radius
    rack_geoms = [int(g) for g in np.flatnonzero(model.geom_bodyid == model.body("rack_0").id)]
    rack_top = max(
        float(data.geom_xpos[g][2] + model.geom_size[g][0])
        for g in rack_geoms
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    )
    assert config.corelle_rim_top_z < rack_top
    assert config.rack_front_x - (centre[0] + outer) == pytest.approx(config.gap, abs=1e-6)


def test_counter_compiles_wooden_so_the_white_plate_stands_out(*, registry: Any) -> None:
    config = RackServingConfig()
    model, _ = _compiled_scene(registry=registry, config=config)
    counter = model.body("cupboard_1").id
    geoms = np.flatnonzero(model.geom_bodyid == counter)
    top = model.geom("cupboard_1_counter").id
    np.testing.assert_allclose(model.geom_rgba[top], [0.60, 0.40, 0.23, 1.0], atol=1e-6)
    # No visible counter geom is left near-white.
    assert not any(
        model.geom_rgba[geom][:3].min() > 0.6 and model.geom_rgba[geom][3] > 0 for geom in geoms
    )
    plate = model.geom("box_0_well").id
    assert model.geom_rgba[plate][:3].min() > model.geom_rgba[top][:3].max()
