"""Neighbor contact is allowed for arm planning, never disabled in physics."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
    WIPER_OBSTACLE_MARGIN,
    KinderSkills,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed
from hitl_pmp.environments.sweeptable3d.recovery import TableFloorPicks, TableKinderSkills


def test_only_neighbor_geometries_are_filtered(*, monkeypatch):
    names = ["cube_1", "box_0", "cupboard_1", "wiper_0"]
    model = SimpleNamespace(
        geom_bodyid=np.arange(4), body=lambda index: SimpleNamespace(name=names[index])
    )
    kinder = SimpleNamespace(shelf=SimpleNamespace(model=model), contact_target="cube_0")
    monkeypatch.setattr(FloorPicks, "obstacle_geoms", lambda *args, **kwargs: [0, 1, 2, 3])
    picker = TableFloorPicks(kinder=kinder)
    assert picker.obstacle_geoms(cube="cube_0") == [1, 2, 3]
    kinder.contact_target = None
    assert picker.obstacle_geoms(cube="cube_0") == [0, 1, 2, 3]


def test_named_planning_filter_keeps_target_and_fixed_scene():
    assert TableKinderSkills.filtered_bodies(
        bodies={1, 2, 3, 4, 5},
        named={"cube_0": 1, "cube_1": 2, "box_0": 3, "cupboard_1": 4, "wiper_0": 5},
        target="cube_0",
    ) == {1, 3, 4, 5}


def test_contact_scope_restores_after_exception():
    kinder = object.__new__(TableKinderSkills)
    kinder.contact_target = None
    with pytest.raises(RuntimeError, match="test"), kinder.allow_ball_contact(cube="cube_0"):
        assert kinder.contact_target == "cube_0"
        raise RuntimeError("test")
    assert kinder.contact_target is None


@pytest.fixture
def planning_geometry(*, monkeypatch):
    bullet = SimpleNamespace(
        GEOM_BOX=3,
        GEOM_CAPSULE=7,
        createCollisionShapeArray=Mock(return_value=600),
        createCollisionShape=Mock(side_effect=[601, 602]),
        createVisualShape=Mock(return_value=603),
        createMultiBody=Mock(side_effect=[700, 701, 702]),
        resetBasePositionAndOrientation=Mock(),
        removeBody=Mock(),
    )
    monkeypatch.setitem(sys.modules, "pybullet", bullet)
    rotation = Rotation.from_euler("z", 90, degrees=True)
    position = np.array([1.5, 0.2, 0.1])
    local_positions = np.array([
        [0, 0, 0],
        [0, 0, 0.2667],
        [0, 0, 0.8],
        [0.3, 0, 0.13],
        [0.4, 0.05, 0.3],
        [0.4, 0.05, 0.35],
    ])
    local_rotations = Rotation.from_euler("y", [0, 0, 0, 15, 90, 0], degrees=True)
    model = SimpleNamespace(
        geom_type=np.array([6, 6, 5, 6, 3, 6]),
        geom_size=np.array([
            [0.55, 0.127, 0.00635],
            [0.55, 0.127, 0.00635],
            [1, 1, 1],
            [0.01, 0.127, 0.13],
            [0.01, 0.09, 0],
            [0.0075, 0.0075, 0.04],
        ]),
        geom_contype=np.array([1, 0, 0, 1, 1, 1]),
        geom_conaffinity=np.array([1, 1, 0, 1, 1, 1]),
    )
    data = SimpleNamespace(
        geom_xpos=rotation.apply(local_positions) + position,
        geom_xmat=(rotation * local_rotations).as_matrix().reshape(-1, 9),
    )
    values = dict(
        zip(("x", "y", "z", "qx", "qy", "qz", "qw"), [*position, *rotation.as_quat()], strict=True)
    )
    state = SimpleNamespace(
        get_object_from_name=lambda name: name, get=lambda obj, key: values[key]
    )
    kinder = object.__new__(TableKinderSkills)
    kinder.shelf = SimpleNamespace(
        model=model, data=data, cupboard_geoms={0, 1, 2, 3}, wiper_geoms=[4, 5]
    )
    kinder.state = lambda: state
    sim = SimpleNamespace(
        physics_client_id=42,
        _cupboard1_shelf_id=100,
        _cupboard1_surface_ids={0, 1, 2, 3},
        _static_colliders={"lab_wall": 300, "box_0": 301, "cube_0": 302, "cube_1": 303},
    )
    sim.named_collision_bodies = lambda: {
        **sim._static_colliders,
        "cupboard_1": sim._cupboard1_shelf_id,
    }
    sim.get_collision_bodies = lambda: set(sim.named_collision_bodies().values())
    monkeypatch.setattr(KinderSkills, "planning_sim", lambda self, **kwargs: sim)
    return kinder, sim, bullet, local_positions, local_rotations


def test_table_planning_replaces_only_fake_shelf_with_live_collision_geoms(*, planning_geometry):
    kinder, sim, bullet, positions, rotations = planning_geometry
    before = kinder.shelf.data.geom_xpos.copy()
    kinder.contact_target = "cube_0"
    assert kinder.planning_sim() is sim
    shape = bullet.createCollisionShapeArray.call_args.kwargs
    assert shape["shapeTypes"] == [bullet.GEOM_BOX] * 3
    np.testing.assert_allclose(shape["halfExtents"], kinder.shelf.model.geom_size[[0, 1, 3]])
    np.testing.assert_allclose(shape["collisionFramePositions"], positions[[0, 1, 3]], atol=1e-15)
    np.testing.assert_allclose(
        Rotation.from_quat(shape["collisionFrameOrientations"]).as_matrix(),
        rotations[[0, 1, 3]].as_matrix(),
        atol=1e-15,
    )
    assert shape["physicsClientId"] == 42
    body = bullet.createMultiBody.call_args.kwargs
    assert body["baseMass"] == 0
    assert body["baseCollisionShapeIndex"] == 600
    np.testing.assert_allclose(body["basePosition"], [1.5, 0.2, 0.1])
    np.testing.assert_allclose(
        Rotation.from_quat(body["baseOrientation"]).as_matrix(),
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-15,
    )
    bullet.removeBody.assert_called_once_with(100, physicsClientId=42)
    assert sim._cupboard1_shelf_id == 700
    assert sim.get_collision_bodies() == {300, 301, 302, 700}
    assert sim.named_collision_bodies()["cube_1"] == 303
    np.testing.assert_array_equal(kinder.shelf.data.geom_xpos, before)


@pytest.mark.parametrize("unsupported", [True, False])
def test_table_geometry_fails_closed_instead_of_removing_fixed_collision(
    *, planning_geometry, unsupported
):
    kinder, sim, bullet, *_ = planning_geometry
    if unsupported:
        kinder.shelf.model.geom_type[1] = 5
    else:
        kinder.shelf.cupboard_geoms.clear()
    with pytest.raises(SkillFailed, match="table"):
        kinder.planning_sim()
    bullet.removeBody.assert_not_called()
    assert sim._cupboard1_shelf_id == 100


def test_wiper_planning_uses_capsule_z_axis_and_live_rotations(*, planning_geometry):
    kinder, sim, bullet, *_ = planning_geometry
    kinder.add_wiper_obstacle(sim=sim)
    capsule, handle = [call.kwargs for call in bullet.createCollisionShape.call_args_list]
    assert capsule == {
        "shapeType": bullet.GEOM_CAPSULE,
        "radius": 0.01 + WIPER_OBSTACLE_MARGIN,
        "height": 0.18,
        "physicsClientId": 42,
    }
    assert handle["shapeType"] == bullet.GEOM_BOX
    np.testing.assert_allclose(
        handle["halfExtents"], kinder.shelf.model.geom_size[5] + WIPER_OBSTACLE_MARGIN
    )
    for geom, call in zip([4, 5], bullet.createMultiBody.call_args_list, strict=True):
        body = call.kwargs
        assert body["baseMass"] == 0
        assert body["physicsClientId"] == 42
        np.testing.assert_allclose(body["basePosition"], kinder.shelf.data.geom_xpos[geom])
        np.testing.assert_allclose(
            Rotation.from_quat(body["baseOrientation"]).as_matrix(),
            kinder.shelf.data.geom_xmat[geom].reshape(3, 3),
            atol=1e-15,
        )
    assert sim._static_colliders["wiper_0/geom4"] == 700
    assert sim._static_colliders["wiper_0/geom5"] == 701
    assert {300, 301, 302, 303, 700, 701} <= sim.get_collision_bodies()
