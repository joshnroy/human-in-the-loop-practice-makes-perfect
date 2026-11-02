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


@pytest.mark.parametrize("lean", [0.0, 15.0, 30.0])
def test_ball_standoff_places_center_inside_pad_without_hitting_support(*, lean):
    distance = TableFloorPicks.standoff(lean_deg=lean)
    radians = np.deg2rad(lean)
    lowest_pad = 0.01 + (distance - 0.042) * np.cos(radians) - 0.011 * np.sin(radians)
    assert lowest_pad >= 0.001 - 1e-12
    assert distance <= 0.038
    assert distance < FloorPicks.standoff(lean_deg=lean)


def test_ball_retry_distinguishes_yaw_offset_and_tracks_target_motion():
    picker = TableFloorPicks(kinder=SimpleNamespace(shelf=None))
    hand = np.eye(4)
    hand[:3, 3] = [1, 2, 0.04]
    centre = np.array([1, 2, 0.01])
    tried = picker.retry_key(hand=hand, centre=centre)
    rotated = hand.copy()
    rotated[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shifted = hand.copy()
    shifted[0, 3] += 0.006
    for alternative in (rotated, shifted):
        assert not picker.same_grasp(
            first=tried, second=picker.retry_key(hand=alternative, centre=centre)
        )
    motion = np.array([0.074, 0, 0])
    hand[:3, 3] += motion
    assert picker.same_grasp(
        first=tried, second=picker.retry_key(hand=hand, centre=centre + motion)
    )


def test_cube_retry_still_excludes_same_approach_yaws():
    picker = FloorPicks(kinder=SimpleNamespace(shelf=None))
    hand = np.eye(4)
    rotated = hand.copy()
    rotated[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    assert picker.same_grasp(
        first=picker.retry_key(hand=hand, centre=np.zeros(3)),
        second=picker.retry_key(hand=rotated, centre=np.zeros(3)),
    )


def test_ball_hand_candidates_use_depth_and_reject_tip_only_leans():
    centre = np.array([1, 2, 0.02])
    core = SimpleNamespace(
        get_object=lambda name: SimpleNamespace(get_pose=lambda: (centre, [1, 0, 0, 0]))
    )
    picker = TableFloorPicks(kinder=SimpleNamespace(shelf=SimpleNamespace(core=core)))
    picker.neighbour_side = lambda **kwargs: 0
    hands = picker.hands(cube="cube_0", centre=centre)
    assert {lean for lean, _ in hands} == {0, 15, 30}
    for lean, hand in hands:
        local = np.linalg.inv(hand) @ np.r_[centre, 1]
        assert local[2] == pytest.approx(TableFloorPicks.standoff(lean_deg=lean))
        assert local[2] <= 0.038


@pytest.mark.parametrize("picker_type", [FloorPicks, TableFloorPicks])
def test_only_ball_failed_executions_enter_retry_history(*, monkeypatch, picker_type):
    from contextlib import nullcontext

    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderControllers

    centre = np.array([1, 2, 0.02])
    hand = np.eye(4)
    grasp = SimpleNamespace(
        hand=hand,
        pre=hand,
        lift=hand,
        pre_conf=np.zeros(7),
        grasp_conf=np.zeros(7),
        base=np.zeros(3),
    )
    controller = SimpleNamespace()
    monkeypatch.setattr(KinderControllers, "get", lambda **kwargs: lambda **kwargs: controller)
    kinder = SimpleNamespace(
        shelf=SimpleNamespace(cube_position=lambda **kwargs: centre),
        objects=lambda *args: args,
        cube_pose=lambda **kwargs: hand,
        constant=lambda **kwargs: nullcontext(),
        pose_in=lambda **kwargs: hand,
        fixed_ik=lambda **kwargs: None,
        planning_sim=lambda: None,
        settler=lambda **kwargs: None,
        lifter=lambda **kwargs: None,
        facing_params=lambda **kwargs: None,
        drive=Mock(side_effect=SkillFailed("capture failed")),
    )
    picker = picker_type(kinder=kinder)
    picker.candidates = lambda **kwargs: iter([grasp])
    tried = []
    with pytest.raises(SkillFailed, match="capture failed"):
        picker.attempt(cube="cube_0", tried=tried, label="test")
    assert len(tried) == (1 if picker_type is TableFloorPicks else 0)
    if tried:
        np.testing.assert_allclose(tried[0], picker.retry_key(hand=hand, centre=centre))


def test_ball_floor_approach_checks_target_but_wiper_does_not(*, monkeypatch):
    settler = Mock(return_value=lambda: None)
    monkeypatch.setattr(KinderSkills, "settler", settler)
    kinder = object.__new__(TableKinderSkills)
    with kinder.allow_ball_contact(cube="cube_0"):
        kinder.settler(hand=np.eye(4), label="pick_settle")
    assert settler.call_args.kwargs["cube"] == "cube_0"
    kinder.settler(hand=np.eye(4), label="wiper_settle")
    assert settler.call_args.kwargs["cube"] is None


@pytest.mark.parametrize("loss", ["closure", "during_lift", "after_lift", None])
def test_ball_lift_checks_capture_and_restores_step_on_loss(*, monkeypatch, loss):
    kinder = object.__new__(TableKinderSkills)
    step = Mock()
    kinder.shelf = SimpleNamespace(step=step)
    kinder.held = Mock(side_effect=[loss != "closure", loss != "after_lift"])
    kinder.cube_in_jaws = Mock(return_value=loss != "during_lift")
    continued = Mock()

    def lift():
        kinder.shelf.step(arm=np.zeros(7), label="test")
        continued()

    monkeypatch.setattr(KinderSkills, "lifter", lambda *args, **kwargs: lift)
    action = kinder.lifter(pre=np.eye(4), cube="cube_0", label="test")
    with kinder.allow_ball_contact(cube="cube_0"):
        if loss is None:
            action()
        else:
            with pytest.raises(SkillFailed, match="cube_0"):
                action()
    assert kinder.shelf.step is step
    assert step.call_count == (0 if loss == "closure" else 1)
    assert continued.call_count == (0 if loss in {"closure", "during_lift"} else 1)


def test_non_floor_lift_keeps_existing_behavior(*, monkeypatch):
    kinder = object.__new__(TableKinderSkills)
    kinder.held = Mock(side_effect=AssertionError("unexpected target check"))
    lift = Mock()
    monkeypatch.setattr(KinderSkills, "lifter", lambda *args, **kwargs: lift)
    kinder.lifter(pre=np.eye(4), cube="cube_0", label="table_pick")()
    lift.assert_called_once_with()


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
