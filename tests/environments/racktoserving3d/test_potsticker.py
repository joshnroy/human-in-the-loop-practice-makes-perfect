"""The rack's food is Pouring3D's frozen potsticker, collision geometry and all."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.racktoserving3d.config import RackServingConfig
from hitl_pmp.environments.racktoserving3d.potsticker import RackPotsticker

kinder_only = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="KINDER simulator"
)


def test_the_rack_potsticker_is_the_pouring_shape_turned_long_axis_across_the_rack():
    from hitl_pmp.environments.pouring3d.potsticker_shape import support_vertices

    source = support_vertices()
    turned = RackPotsticker.support_vertices()
    np.testing.assert_allclose(np.ptp(turned, axis=0), np.ptp(source, axis=0)[[1, 0, 2]])
    np.testing.assert_allclose((turned.min(axis=0) + turned.max(axis=0)) / 2, 0, atol=1e-12)
    np.testing.assert_allclose(np.ptp(turned, axis=0)[1:], [0.055, 0.02665], atol=1e-9)


def test_the_config_food_extents_are_the_potstickers_not_free_parameters():
    config = RackServingConfig()
    extents = np.ptp(RackPotsticker.support_vertices(), axis=0)
    np.testing.assert_allclose(
        [config.dumpling_depth, config.dumpling_width, config.dumpling_height], extents
    )
    assert config.dumpling_mass == 0.020


def _env(*, seed=0):
    from hitl_pmp.environments.racktoserving3d.environment import RackServingEnvironment
    from hitl_pmp.environments.racktoserving3d.tasks import RackServingTasks

    env = RackServingEnvironment()
    RackServingTasks(env=env, seed=seed).sample_train_task()
    return env


def _settle(*, env, placements):
    backend = env.backend()
    backend._commit(placements=placements)
    env.rack_controllers().shelf.hold(count=60, label="settle")
    env.current_state = env._observed_state(seed=0, steps_taken=1)


@kinder_only
def test_reset_scene_has_five_potsticker_bodies_with_its_collision_geoms():
    import mujoco

    from hitl_pmp.environments.pouring3d.potsticker_shape import shape_xml

    expected = shape_xml().findall("worldbody/body/geom")
    env = _env()
    try:
        backend = env.backend()
        model = backend.model()
        assert list(backend.cube_names) == [f"cube_{index}" for index in range(5)]
        assert not np.any(model.geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID)
        for name in backend.cube_names:
            geoms = sorted(backend.body_geoms(name=name))
            assert [int(model.geom_type[geom]) for geom in geoms] == [
                int(mujoco.mjtGeom.mjGEOM_MESH)
                if geom_xml.get("type") == "mesh"
                else int(mujoco.mjtGeom.mjGEOM_CAPSULE)
                for geom_xml in expected
            ]
            assert model.body_mass[model.body(name).id] == pytest.approx(0.020)
            assert backend.location(name=name) == "OnRack"
    finally:
        env.close()


@kinder_only
def test_potsticker_locations_on_rack_plate_well_and_counter():
    env = _env()
    try:
        backend = env.backend()
        config = backend.config
        well = backend.corelle_support()
        z = config.dumpling_height / 2 + 0.002
        upright = np.array([1.0, 0, 0, 0])
        _settle(
            env=env,
            placements={
                "cube_0": (np.array([well.center_x, well.center_y, well.top_z + z]), upright),
                "cube_1": (np.array([1.10, 0.38, config.counter_height + z]), upright),
            },
        )
        assert backend.location(name="cube_0") == "OnTray"
        assert backend.settled_on_tray(name="cube_0")
        assert backend.location(name="cube_1") == "SpilledCounter"
        for name in ("cube_2", "cube_3", "cube_4"):
            assert backend.location(name=name) == "OnRack"
        atoms = env.get_current_state().abstract_atoms
        assert ("OnTray", ("cube_0", "box_0")) in atoms
        assert ("SpilledCounter", ("cube_1",)) in atoms
        assert ("FoodAvailable", ("cube_1",)) in atoms
    finally:
        env.close()


@kinder_only
def test_batch_return_settles_every_potsticker_on_the_rack():
    env = _env(seed=3)
    try:
        backend = env.backend()
        z = backend.config.counter_height + backend.config.dumpling_height / 2 + 0.002
        _settle(
            env=env,
            placements={
                name: (np.array([1.02 + 0.12 * index, 0.38, z]), np.array([1.0, 0, 0, 0]))
                for index, name in enumerate(backend.cube_names)
            },
        )
        env.reset_movables(destination="human_batch_return")
        atoms = env.get_current_state().abstract_atoms
        for name in backend.cube_names:
            assert ("OnRack", (name, "rack_0")) in atoms
            assert ("FoodAvailable", (name,)) in atoms
    finally:
        env.close()


@kinder_only
def test_carried_potsticker_bottom_is_its_lowest_vertex_not_a_box_about_its_frame():
    from hitl_pmp.environments.racktoserving3d.food import CarriedGeom

    env = _env()
    try:
        backend = env.backend()
        model, data = backend.model(), backend.data()
        body = model.body("cube_0").id
        hull = RackPotsticker.support_vertices() @ data.xmat[body].reshape(3, 3).T
        lowest = float((hull + data.xpos[body])[:, 2].min())
        bottoms = []
        for geom in backend.body_geoms(name="cube_0"):
            center = data.geom_xpos[geom].copy()
            rotation = data.geom_xmat[geom].reshape(3, 3)
            box_center, size, bottom = CarriedGeom.bounds(
                model=model, geom=geom, center=center, rotation=rotation
            )
            assert bottom >= lowest - 1e-6
            assert np.all(np.abs(box_center - center) <= 0.03) and np.all(size > 0)
            bottoms.append(bottom)
        # The lowest part sits on the rack, within its rest penetration.
        assert min(bottoms) == pytest.approx(lowest, abs=1e-6)
        assert min(bottoms) >= backend.support(name="rack_0").top_z - 0.003
    finally:
        env.close()
