"""Live physics checks; outcome scoring is not an assertion of hardware performance."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.kitchen_recycling3d.domain import KitchenConfig, classify_zone

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="Optional Tossing3D simulation dependencies are not installed",
)


@pytest.fixture
def simulator():
    from hitl_pmp.environments.kitchen_recycling3d.simulator import RecyclingSimulator

    environment = RecyclingSimulator(config=KitchenConfig(scatter_objects=False), pretty=False)
    try:
        yield environment
    finally:
        environment.close()


def test_compiled_objects_masses_floor_zones_and_sideways_pose(*, simulator):
    config = simulator.config
    for spec in config.objects:
        body = simulator.model.body(simulator.names[spec.name]).id
        assert simulator.model.body_mass[body] == pytest.approx(spec.simulation_mass)
        observation = simulator.episode.observations[spec.name]
        assert classify_zone(config=config, observation=observation) == "kitchen"
    bottle = simulator.episode.observations["gatorade_20oz"]
    assert abs(bottle.quaternion[1]) > 0.7
    mat = simulator.model.body("sprout_mats").id
    np.testing.assert_allclose(simulator.data.xpos[mat], (2.375, 0, 0.0006), atol=1e-6)
    for index in range(simulator.model.ngeom):
        if simulator.model.geom_bodyid[index] == mat:
            assert simulator.model.geom_contype[index] == 0
            assert simulator.model.geom_conaffinity[index] == 0
    assert "sprout_mats" not in simulator.backend._state.get_object_names()
    assert not simulator.episode.solved


def test_unsafe_drive_rejected_before_physics(*, simulator):
    before = simulator.data.qpos.copy()
    action = np.zeros(simulator.backend._env.action_space.shape)
    action[0] = 1.0
    with pytest.raises(PermissionError):
        simulator.step(action=action)
    np.testing.assert_array_equal(simulator.data.qpos, before)


def test_desk_and_four_chairs_are_physical_planning_obstacles(*, simulator):
    from hitl_pmp.environments.kitchen_recycling3d.furniture import (
        CHAIR_POSITIONS,
        FURNITURE_POSITION,
    )

    expected = {"recycling_lab_desk": FURNITURE_POSITION}
    expected.update({
        f"recycling_chair_{index}": (*position, 0) for index, position in enumerate(CHAIR_POSITIONS)
    })
    names = simulator.backend._state.get_object_names()
    for name, position in expected.items():
        body = simulator.model.body(name).id
        np.testing.assert_allclose(simulator.data.xpos[body], position)
        assert simulator.model.body_jntnum[body] == 0
        assert simulator.config.mat_bounds[0] < position[0] < simulator.config.mat_bounds[2]
        assert any(value.startswith(f"collider:{name}:") for value in names)
        colliders = [
            index
            for index in range(simulator.model.ngeom)
            if simulator.model.geom_bodyid[index] == body and simulator.model.geom_contype[index]
        ]
        assert len(colliders) == 1
        collider = colliders[0]
        extent = (
            np.abs(simulator.data.geom_xmat[collider].reshape(3, 3))
            @ (simulator.model.geom_size[collider])
        )
        center = simulator.data.geom_xpos[collider]
        assert np.all(center[:2] - extent[:2] >= simulator.config.mat_bounds[:2])
        assert np.all(center[:2] + extent[:2] <= simulator.config.mat_bounds[2:])
        for geom in range(simulator.model.ngeom):
            if simulator.model.geom_bodyid[geom] != body or geom == collider:
                continue
            mesh = simulator.model.geom_dataid[geom]
            start = simulator.model.mesh_vertadr[mesh]
            count = simulator.model.mesh_vertnum[mesh]
            vertices = simulator.model.mesh_vert[start : start + count]
            world = vertices @ simulator.data.geom_xmat[geom].reshape(3, 3).T
            world += simulator.data.geom_xpos[geom]
            local = (world - simulator.data.geom_xpos[collider]) @ (
                simulator.data.geom_xmat[collider].reshape(3, 3)
            )
            assert np.all(np.abs(local) <= simulator.model.geom_size[collider] + 1e-5)


def test_rectangular_room_and_kitchen_replace_tv(*, simulator):
    room = simulator.model.body("tossing_room").id
    walls = [
        index
        for index in range(simulator.model.ngeom)
        if simulator.model.geom_bodyid[index] == room
    ]
    assert len(walls) == 4
    for geom in walls:
        np.testing.assert_allclose(simulator.data.geom_xmat[geom].reshape(3, 3), np.eye(3))
        assert simulator.model.geom_contype[geom] == 1
    assert len(simulator.scene._placement_room_halfspaces()) == 4
    geom_names = {simulator.model.geom(index).name for index in range(simulator.model.ngeom)}
    assert "living_room" not in geom_names
    assert "wall_leftcorner_visual" not in geom_names
    stove = simulator.model.body("kitchen_stove").id
    assert simulator.data.xpos[stove][0] < simulator.config.mat_bounds[0]
    for index in range(4):
        assert simulator.model.geom(f"stove_burner_{index}").id >= 0
    names = simulator.backend._state.get_object_names()
    assert any(name.startswith("collider:kitchen_stove:") for name in names)


def test_costed_near_context_and_bin_restoration_then_no_deployment_help(*, simulator):
    simulator.human(action="relocate_bin", destination=(0.65, 1.4, 0))
    assert simulator.episode.context() == "near"
    with pytest.raises(ValueError, match="restore"):
        simulator.begin_deployment()
    simulator.human(action="restore_bin")
    assert simulator.episode.context() == "far"
    assert simulator.episode.human_cost == 24
    simulator.begin_deployment()
    before = simulator.data.qpos.copy()
    for action in ("recover_object", "relocate_bin", "restore_bin"):
        with pytest.raises(PermissionError):
            simulator.human(action=action, name="food_can", destination=(0.5, 0, 0.05))
    np.testing.assert_array_equal(simulator.data.qpos, before)
    assert simulator.episode.human_cost == 24


def test_actual_sideways_bottle_pick_toss_and_observed_miss(*, simulator):
    simulator.begin_deployment()
    picked = simulator.pick(name="gatorade_20oz")
    assert picked["success"], picked
    assert picked["steps"] > 0
    assert simulator.episode.observations["gatorade_20oz"].finger_contacts == 2
    tossed = simulator.toss(name="gatorade_20oz")
    assert tossed["steps"] > 0, tossed
    assert tossed["success"] == ("gatorade_20oz" in simulator.episode.settled)
    assert simulator.episode.human_cost == 0
    assert not simulator.episode.solved
    assert simulator.episode.events[-1]["statistics_key"] == "gatorade_20oz/far/toss"
    assert not np.allclose(simulator.episode.observations["gatorade_20oz"].position[:2], (0.58, 0))


def test_toss_without_verified_grasp_and_unknown_human_action_are_rejected(*, simulator):
    with pytest.raises(ValueError, match="contact"):
        simulator.toss(name="food_can")
    with pytest.raises(ValueError, match="Unknown human"):
        simulator.human(action="dump_bin")
    assert simulator.episode.human_cost == 0


def test_autonomous_retrieval_from_reachable_bin_fixture(*, simulator):
    simulator.human(action="relocate_bin", destination=(0.65, -1.4, 0))
    state = simulator.backend._require_state().copy()
    obj = state.get_object_from_name("cube_0")
    for key, value in zip("xyz", (0.65, -1.4, 0.032), strict=True):
        state.set(obj, key, value)
    simulator.scene.set_state(state)
    simulator.backend._state = state
    simulator.idle(seconds=1)
    assert "cube_control" in simulator.episode.settled
    cost = simulator.episode.human_cost
    result = simulator.pick(name="cube_control")
    assert result["success"], result
    assert "cube_control" in simulator.episode.holding
    assert simulator.episode.human_cost == cost
