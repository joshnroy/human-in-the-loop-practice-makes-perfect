"""Rack scraper geometry and compatibility checks without physics stepping.

Mock geometry follows the kitchen prototype's bench scraper and wiper tests.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.servo import JointStep, SweepServo
from hitl_pmp.environments.racktoserving3d.tool import (
    BODY_REST,
    RackScraper,
    RackShelfSkills,
    TableWiper,
    box_overlap,
    wiper_unwatched,
)
from hitl_pmp.environments.racktoserving3d.transit import (
    EmptyHandTransit,
    geom_half_extents,
    segment_samples,
)
from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import SkillFailed


class FakeShelf:
    """A full-width scraper, counter, rack slab, robot and horizontal wire."""

    def __init__(self):
        self.model = SimpleNamespace(
            geom_type=np.array([6, 6, 6, 6, 6, 3]),
            geom_size=np.array([
                [0.125, 0.0005, 0.06],
                [0.125, 0.008, 0.008],
                [0.36, 0.5, 0.02],
                [0.1275, 0.1875, 0.003],
                [0.01, 0.01, 0.01],
                [0.0015, 0.18, 0],
            ]),
            geom_pos=np.array([
                [0, 0, 0.06],
                [0, 0, 0.128],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]),
            geom_quat=np.tile([1.0, 0, 0, 0], (6, 1)),
            geom_contype=np.ones(6, dtype=int),
            geom_conaffinity=np.ones(6, dtype=int),
            geom_rbound=np.ones(6),
            body_jntadr=np.array([0]),
            jnt_dofadr=np.array([0]),
            ngeom=6,
        )
        self.data = SimpleNamespace(
            qvel=np.zeros(6),
            contact=[],
            ncon=0,
            geom_xpos=np.array([
                [0.0, 0, 0],
                [0, 0, 0],
                [1.34, 0, 0.88],
                [1.5, 0, 0.922],
                [-1, 0, 0],
                [1.5, 0, 0.9235],
            ]),
            geom_xmat=np.tile(np.eye(3).reshape(1, 9), (6, 1)),
        )
        self.data.geom_xmat[5] = Rotation.from_euler("x", 90, degrees=True).as_matrix().ravel()
        self.core = SimpleNamespace()
        self.wiper_geoms = [0, 1]
        self.wiper_body = 0
        self.cupboard_geoms = {2, 3}
        self.box_geoms = set()
        self.robot_geoms = {4}
        self.board = {1: 0.925}
        self.front_x = 1.3725
        self.in_hand = None
        self.watch_wiper = True
        self.wiper_unheld_steps = 8
        self.pose = np.eye(4)
        self.pads_hold_wiper = Mock(return_value=False)

    def wiper_pose(self):
        return self.pose.copy()

    def support(self, *, geom=2, height=0.9):
        self.data.contact = [
            SimpleNamespace(
                geom1=0,
                geom2=geom,
                dist=0.0,
                pos=np.array([1.02, -0.34, height]),
                frame=np.array([0, 0, 1, 1, 0, 0, 0, 1, 0]),
            )
        ]
        self.data.ncon = 1


def test_counter_home_is_separate_from_rack_and_preserves_shelf():
    shelf = FakeShelf()
    board = dict(shelf.board)
    tool = RackScraper(shelf=shelf, config=SimpleNamespace(counter_height=0.9))
    shelf.pose = tool.rest_pose()
    assert (tool.top, tool.front) == (0.925, 1.3725)
    assert (tool.home_top, tool.home_front, tool.rest_y) == (0.9, 0.98, -0.34)
    assert shelf.board == board
    assert shelf.front_x == 1.3725
    np.testing.assert_allclose(tool.rest_pose()[:3, :3], BODY_REST)
    assert tool.capsule_bottom() == pytest.approx(0.9)
    assert tool.handle_end()[0] == pytest.approx(0.98 - 0.059)
    assert tool.tool_clear(pose=shelf.pose)
    assert not tool.at_home()
    shelf.support()
    assert tool.at_home()
    shelf.support(geom=3, height=0.925)
    assert not tool.at_home()


@pytest.mark.parametrize("contact_geom, expected", [(0, False), (1, True)])
def test_scraper_grasp_requires_handle_not_blade(*, contact_geom, expected):
    shelf = FakeShelf()
    shelf.pad_sides = [{10}, {11}]
    shelf.data.contact = [
        SimpleNamespace(geom1=pad, geom2=contact_geom, dist=-0.00001) for pad in (10, 11)
    ]
    shelf.data.ncon = 2
    tool = RackScraper(shelf=shelf)
    assert tool.pads_hold_grip() == expected
    np.testing.assert_allclose(tool.point, tool.end_in_body + 0.01 * tool.handle_axis)


def test_explicit_home_overrides_config_and_copies_base():
    shelf = FakeShelf()
    base = np.array([0.3, -0.4, 0])
    tool = RackScraper(
        shelf=shelf,
        config=SimpleNamespace(counter_height=0.9, counter_front_x=0.98),
        home_top=0.94,
        home_front=1.02,
        rest_y=-0.4,
        base=base,
    )
    base[:] = 0
    assert (tool.home_top, tool.home_front, tool.rest_y) == (0.94, 1.02, -0.4)
    np.testing.assert_allclose(tool.base, [0.3, -0.4, 0])
    assert tool.capsule_bottom(pose=tool.rest_pose()) == pytest.approx(0.94)
    assert tool.handle_end(pose=tool.rest_pose())[0] == pytest.approx(1.02 - 0.059)


def test_configured_blade_width_and_full_width_handle():
    shelf = FakeShelf()
    shelf.model.geom_size[:2, 0] = 0.075
    tool = RackScraper(shelf=shelf, config=SimpleNamespace(scraper_width=0.15))
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
    edge = tool.blade_edge(pose=pose)
    assert np.linalg.norm(edge[1] - edge[0]) == pytest.approx(0.15)
    shelf.model.geom_size[1, 0] = 0.01
    with pytest.raises(SkillFailed, match="full-width"):
        RackScraper(shelf=shelf)


@pytest.mark.parametrize("yaw", [-90, 0, 35, 90])
def test_physical_blade_edge_is_25cm_and_handle_tilts_45_degrees(*, yaw):
    tool = RackScraper(shelf=FakeShelf())
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=yaw)
    edge = tool.blade_edge(pose=pose)
    assert np.linalg.norm(edge[1] - edge[0]) == pytest.approx(0.25)
    np.testing.assert_allclose(edge[:, 2], tool.capsule_bottom(pose=pose))
    assert pose[2, 2] == pytest.approx(np.sqrt(0.5))


@pytest.mark.parametrize("wire", [False, True])
def test_only_shallow_top_support_is_permitted(*, wire):
    shelf = FakeShelf()
    shelf.cupboard_geoms = {5} if wire else {3}
    tool = RackScraper(shelf=shelf)
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
    pose[:3, 3] = [1.5, 0, 0]
    pose[2, 3] += tool.top - 0.002 - tool.capsule_bottom(pose=pose)
    assert tool.tool_clear(pose=pose)
    pose[2, 3] -= 0.003
    assert not tool.tool_clear(pose=pose)


@pytest.mark.parametrize("below_mm,clear", [(1.0, True), (4.0, False)])
def test_sweep_may_ride_the_plate_rim_shallowly(*, below_mm, clear):
    shelf = FakeShelf()
    shelf.cupboard_geoms, shelf.box_geoms = {2}, {3}
    # The plate rim sits lower than the rack top, so the counter/rack rule never covers it.
    shelf.data.geom_xpos[3] = [1.5, 0, 0.913]
    rim_top = 0.913 + 0.003
    tool = RackScraper(shelf=shelf)
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
    pose[:3, 3] = [1.5, 0, 0]
    pose[2, 3] += rim_top - below_mm / 1000 - tool.capsule_bottom(pose=pose)
    assert tool.tool_clear(pose=pose) is clear


@pytest.mark.parametrize("tilt_deg", [0.001, 0.1])
def test_movable_wire_support_tolerates_small_physical_tilt(*, tilt_deg):
    shelf = FakeShelf()
    shelf.cupboard_geoms = {5}
    shelf.data.geom_xmat[5] = (
        Rotation.from_euler("x", 90 + tilt_deg, degrees=True).as_matrix().ravel()
    )
    tool = RackScraper(shelf=shelf)
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
    pose[:3, 3] = [1.5, 0, 0]
    pose[2, 3] += tool.top - 0.001 - tool.capsule_bottom(pose=pose)
    assert tool.tool_clear(pose=pose)
    pose[2, 3] -= 0.006
    pose[0, 3] += 0.006
    assert not tool.tool_clear(pose=pose)


def test_clearance_checks_full_width_handle_and_live_mutable_rack():
    shelf = FakeShelf()
    shelf.cupboard_geoms = {3}
    tool = RackScraper(shelf=shelf)
    pose = np.eye(4)
    pose[:3, 3] = [1.5, 0, 1]
    assert tool.tool_clear(pose=pose)
    shelf.model.geom_size[3] = [0.004, 0.004, 0.004]
    shelf.data.geom_xpos[3] = [1.62, 0, 1.128]
    assert not tool.tool_clear(pose=pose)
    shelf.data.geom_xpos[3, 0] += 0.02
    assert tool.tool_clear(pose=pose)
    shelf.data.geom_xpos[3] = [1.62, 0, 1.04]
    assert not tool.tool_clear(pose=pose)
    shelf.model.geom_contype[3] = 0
    shelf.model.geom_conaffinity[3] = 0
    assert tool.tool_clear(pose=pose)
    shelf.model.geom_contype[3] = 1
    assert not tool.tool_clear(pose=pose)


def test_oriented_box_overlap_detects_edge_crossing_without_enclosed_corners():
    arguments = dict(
        center=np.zeros(3),
        rotation=np.eye(3),
        size=np.array([1, 0.01, 0.01]),
        other_center=np.zeros(3),
        other_rotation=Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        other_size=np.array([1, 0.01, 0.01]),
    )
    assert box_overlap(**arguments) > 0
    arguments["other_center"] = np.array([0, 0, 0.03])
    assert box_overlap(**arguments) < 0


def test_release_watch_compatibility_restores_after_nested_exception():
    shelf = FakeShelf()
    assert not hasattr(shelf, "wiper_unwatched")
    with pytest.raises(RuntimeError), wiper_unwatched(shelf=shelf):
        assert not shelf.watch_wiper
        with wiper_unwatched(shelf=shelf):
            shelf.wiper_unheld_steps = 10
        assert not shelf.watch_wiper
        assert shelf.wiper_unheld_steps == 0
        raise RuntimeError("release failed")
    assert shelf.watch_wiper
    assert shelf.wiper_unheld_steps == 0
    compatible = RackShelfSkills.__new__(RackShelfSkills)
    compatible.watch_wiper = True
    with compatible.wiper_unwatched():
        assert not compatible.watch_wiper
    assert compatible.watch_wiper


def test_initialization_is_the_only_object_pose_writer():
    shelf = FakeShelf()
    shelf.grip = 1.0
    obj = SimpleNamespace(set_pose=Mock(), set_velocity=Mock())
    shelf.core.get_object = Mock(return_value=obj)
    shelf.sim = SimpleNamespace(forward=Mock())
    shelf.hold = Mock()
    tool = RackScraper(shelf=shelf)
    tool.rest_pose()
    obj.set_pose.assert_not_called()
    tool.initialize_rest()
    assert shelf.grip == 0.0
    obj.set_pose.assert_called_once()
    position, quaternion = obj.set_pose.call_args.args
    np.testing.assert_allclose(position, tool.rest_pose()[:3, 3])
    np.testing.assert_allclose(
        Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix(), BODY_REST, atol=1e-15
    )
    obj.set_velocity.assert_called_once_with([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_joint_clearance_rejects_old_ik_collision_cost():
    shelf = FakeShelf()
    tool = RackScraper(shelf=shelf)
    shelf.ik_data = object()
    shelf.site_at = Mock(return_value=np.eye(4))
    shelf.collision_cost = Mock(return_value=0.001)
    with pytest.raises(SkillFailed, match="collide"):
        tool.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)
    shelf.ik_data = shelf.data
    with pytest.raises(SkillFailed, match="separate scratch"):
        tool.check_configuration(base=np.zeros(3), arm=np.zeros(7), holding=False)


def _planned(*, monkeypatch, tool):
    """Stand in for the PyBullet planner and the path followers: these tests are about
    the scripted contact motions after the planned carry."""
    from hitl_pmp.environments.racktoserving3d import planning

    plan = planning.ScraperPlan(
        stance=tool.base.copy(), base_path=[tool.base.copy()], arm_after_base=[np.zeros(7)]
    )
    monkeypatch.setattr(
        planning.ScraperMotionPlanner,
        "plan_return",
        staticmethod(lambda **_: (plan, np.zeros(3))),
    )
    tool.follow_arm_path = Mock()
    tool.follow_base_path = Mock()
    tool.settle_hand = Mock()


def test_return_uses_release_guard_without_requiring_new_shelf_methods(*, monkeypatch):
    shelf = FakeShelf()
    tool = RackScraper(shelf=shelf)
    _planned(monkeypatch=monkeypatch, tool=tool)
    tool.pads_hold_grip = Mock(return_value=True)
    shelf.pose = tool.rest_pose()
    shelf.site_pose = Mock(side_effect=shelf.wiper_pose)
    shelf.in_hand = np.eye(4)
    shelf.pads_hold_wiper.return_value = True
    tool.line = Mock()
    tool.move_base = Mock()
    tool.table_support = Mock(return_value=True)
    tool.at_home = Mock(return_value=True)

    def hold(*, count, label):
        assert count > 0
        assert not shelf.watch_wiper
        if label == "place_wiper_release":
            shelf.pads_hold_wiper.return_value = False

    shelf.hold = Mock(side_effect=hold)
    tool.place()
    assert shelf.in_hand is None
    assert shelf.watch_wiper
    assert shelf.grip == 0
    assert shelf.wiper_unheld_steps == 0
    assert shelf.hold.call_args_list[0].kwargs["label"] == "place_wiper_release"
    assert tool.at_home.call_count == 10
    assert [call.kwargs["label"] for call in tool.line.call_args_list] == [
        "place_wiper_lower",
        "place_wiper_retreat",
        "place_wiper_clear",
    ]


def test_return_without_support_retains_the_physical_grasp(*, monkeypatch):
    shelf = FakeShelf()
    tool = RackScraper(shelf=shelf)
    _planned(monkeypatch=monkeypatch, tool=tool)
    tool.pads_hold_grip = Mock(return_value=True)
    shelf.pose = tool.rest_pose()
    shelf.site_pose = Mock(side_effect=shelf.wiper_pose)
    shelf.in_hand = np.eye(4)
    shelf.grip = 1
    shelf.pads_hold_wiper.return_value = True
    shelf.hold = Mock()
    tool.line = Mock()
    tool.move_base = Mock()
    tool.table_support = Mock(return_value=False)
    with pytest.raises(SkillFailed, match="no board support; retaining"):
        tool.place()
    assert shelf.in_hand is not None
    assert shelf.grip == 1
    shelf.hold.assert_not_called()
    assert shelf.watch_wiper


def test_wire_capsule_bounds_include_rounded_ends():
    shelf = FakeShelf()
    shelf.model.geom_type[5] = 3
    np.testing.assert_allclose(
        geom_half_extents(model=shelf.model, geom=5), [0.0015, 0.0015, 0.1815]
    )


def test_flat_cylinder_bounds_are_its_exact_box_without_phantom_caps():
    """A 0.13 m plate is not a capsule: caps would add a 0.13 m exclusion above it."""
    shelf = FakeShelf()
    shelf.model.geom_type[5] = 5
    shelf.model.geom_size[5] = [0.13, 0.006, 0]
    np.testing.assert_allclose(geom_half_extents(model=shelf.model, geom=5), [0.13, 0.13, 0.006])


def test_transit_route_bounds_accept_mutable_wire_fixtures():
    shelf = FakeShelf()
    shelf.cupboard_geoms = {2, 5}
    shelf.ba = np.arange(3)
    shelf.data.qpos = np.zeros(3)
    transit = EmptyHandTransit(shelf=shelf)
    transit.clear = Mock(return_value=True)
    transit.segment_clear = Mock(side_effect=[False, *([True] * 5)])
    target = np.array([0.6, -0.34, 0])
    route = transit.plan(target=target, arm=np.zeros(7))
    assert route is not None
    np.testing.assert_array_equal(route[-1], target)


def test_transit_geometry_refreshes_collision_enablement():
    shelf = FakeShelf()
    shelf.model.geom_contype[5] = shelf.model.geom_conaffinity[5] = 0
    transit = EmptyHandTransit(shelf=shelf)
    assert 5 not in transit.obstacles
    shelf.model.geom_contype[5] = 1
    shelf.cupboard_geoms.add(5)
    transit.refresh_geometry()
    assert 5 in transit.obstacles
    assert 5 in transit.fixtures


def test_segment_samples_keep_unwrapped_heading_and_axis_clipped_path():
    start = np.array([0, 0, 2 * np.pi])
    target = np.array([0.02, -0.004, 0])
    samples = segment_samples(start=start, target=target)
    np.testing.assert_array_equal(samples[0], start)
    np.testing.assert_array_equal(samples[-1], target)
    assert any(np.allclose(sample, [0.01, -0.004, 2 * np.pi - 0.01]) for sample in samples)
    assert any(abs(sample[2] - np.pi) < 0.01 for sample in samples)
    with pytest.raises(ValueError, match="finite"):
        segment_samples(start=start, target=np.array([0, 0, np.nan]))


def test_servo_torque_hook_is_bounded_read_only_and_restored():
    robot = SimpleNamespace(
        _update_ctrl=Mock(),
        _compute_arm_torques=Mock(return_value=np.full(7, 100.0)),
        ARM_TORQUE_LIMITS=np.full(7, 2.0),
    )
    shelf = SimpleNamespace(
        core=SimpleNamespace(_robot_env=robot),
        model=SimpleNamespace(nv=7, site_bodyid=np.array([0])),
        data=SimpleNamespace(qpos=np.zeros(7)),
        site=0,
        qa=np.arange(7),
        arm_reference=np.zeros(7),
        ticks=0,
    )
    shelf.data.qpos.setflags(write=False)
    original = robot._update_ctrl
    servo = SweepServo(shelf=shelf, max_normal_force_n=0)
    with pytest.raises(RuntimeError, match="test"), servo:
        action = np.zeros(11)
        action.setflags(write=False)
        robot._update_ctrl(action)
        np.testing.assert_array_equal(original.call_args.args[0][3:10], np.full(7, 2.0))
        assert servo.clipped_torque_count == 7
        raise RuntimeError("test")
    assert robot._update_ctrl is original


@pytest.mark.parametrize("holding", [True, False])
def test_servo_ramps_a_held_scrapers_heading_across_the_control_period(*, holding):
    robot = SimpleNamespace(
        _update_ctrl=Mock(),
        _compute_arm_torques=Mock(return_value=np.zeros(7)),
        ARM_TORQUE_LIMITS=np.full(7, 2.0),
        control_frequency=10.0,
    )
    qpos = np.zeros(10)
    shelf = SimpleNamespace(
        core=SimpleNamespace(_robot_env=robot),
        model=SimpleNamespace(
            nv=10, site_bodyid=np.array([0]), opt=SimpleNamespace(timestep=0.025)
        ),
        data=SimpleNamespace(qpos=qpos, xmat=np.eye(3).reshape(1, 9), xpos=np.zeros((1, 3))),
        site=0,
        qa=np.arange(3, 10),
        ba=np.arange(3),
        arm_reference=np.zeros(7),
        ticks=0,
        in_hand=np.eye(4) if holding else None,
    )
    wiper = SimpleNamespace(head_geom=0, is_bench_scraper=False)
    shelf.data.geom_xpos = np.zeros((1, 3))
    with SweepServo(shelf=shelf, wiper=wiper, max_normal_force_n=0):
        action = np.r_[0.01, -0.01, 0.04, np.zeros(8)]
        for _ in range(4):
            robot._update_ctrl(action)
    sent = np.array([call.args[0][:3] for call in robot._update_ctrl.call_args_list])
    np.testing.assert_allclose(sent[:, :2], [[0.01, -0.01]] * 4)
    expected = [0.01, 0.02, 0.03, 0.04] if holding else [0.04] * 4
    np.testing.assert_allclose(sent[:, 2], expected)


def reference_tool_clear(*, tool, pose, data=None):
    """`RackScraper.tool_clear` as it was: every obstacle tested one at a time."""
    from hitl_pmp.environments.racktoserving3d.tool import GRAZE, SUPPORT_ALLOWANCE

    if not np.isfinite(pose).all():
        return False
    data = tool.data if data is None else data
    shelf, model = tool.shelf, tool.model
    for geom in (tool.head_geom, tool.handle_geom):
        center = pose[:3, :3] @ model.geom_pos[geom] + pose[:3, 3]
        rotation = pose[:3, :3] @ tool.geom_rotation(geom=geom)
        size = model.geom_size[geom]
        bottom = float(center[2] - np.abs(rotation[2]) @ size)
        for obstacle in shelf.cupboard_geoms | shelf.box_geoms:
            if not (model.geom_contype[obstacle] or model.geom_conaffinity[obstacle]):
                continue
            kind = int(model.geom_type[obstacle])
            other_center = data.geom_xpos[obstacle]
            other_rotation = data.geom_xmat[obstacle].reshape(3, 3)
            other_size = geom_half_extents(model=model, geom=obstacle)
            if np.any(
                np.abs(other_center - center)
                > np.abs(rotation) @ size + np.abs(other_rotation) @ other_size
            ):
                continue
            overlap = box_overlap(
                center=center,
                rotation=rotation,
                size=size,
                other_center=other_center,
                other_rotation=other_rotation,
                other_size=other_size,
            )
            if overlap <= GRAZE:
                continue
            support_top = float(other_center[2] + np.abs(other_rotation[2]) @ other_size)
            support = (
                obstacle in shelf.cupboard_geoms
                and (
                    (kind == 6 and abs(other_rotation[2, 2]) >= np.cos(np.deg2rad(2)))
                    or (kind in (3, 5) and abs(other_rotation[2, 2]) <= np.sin(np.deg2rad(2)))
                    or kind == 2
                )
                and min(abs(support_top - tool.home_top), abs(support_top - tool.top)) <= 0.003
                and bottom >= support_top - SUPPORT_ALLOWANCE
            )
            if not support:
                return False
    return True


def cluttered_shelf(*, rng):
    """FakeShelf's scraper amid many random rack wires, slabs, posts and meshes."""
    shelf = FakeShelf()
    count = 60
    kinds = rng.choice([2, 3, 5, 6, 7], size=count)
    sizes = np.column_stack([
        rng.uniform(0.001, 0.05, count),
        rng.uniform(0.001, 0.2, count),
        rng.uniform(0.001, 0.05, count),
    ])
    rotations = Rotation.random(count, random_state=int(rng.integers(1 << 30))).as_matrix()
    level = rng.random(count) < 0.5
    rotations[level & (kinds == 6)] = np.eye(3)
    rotations[level & np.isin(kinds, (3, 5))] = Rotation.from_euler(
        "x", 90, degrees=True
    ).as_matrix()
    centers = np.column_stack([
        rng.uniform(1.3, 1.7, count),
        rng.uniform(-0.2, 0.2, count),
        rng.uniform(0.88, 1.05, count),
    ])
    model, data = shelf.model, shelf.data
    model.geom_type = np.r_[model.geom_type, kinds]
    model.geom_size = np.r_[model.geom_size, sizes]
    model.geom_pos = np.r_[model.geom_pos, np.zeros((count, 3))]
    model.geom_quat = np.r_[model.geom_quat, np.tile([1.0, 0, 0, 0], (count, 1))]
    model.geom_contype = np.r_[model.geom_contype, (rng.random(count) < 0.9).astype(int)]
    model.geom_conaffinity = np.r_[model.geom_conaffinity, np.zeros(count, dtype=int)]
    model.geom_rbound = np.r_[model.geom_rbound, rng.uniform(0.005, 0.1, count)]
    model.ngeom += count
    data.geom_xpos = np.r_[data.geom_xpos, centers]
    data.geom_xmat = np.r_[data.geom_xmat, rotations.reshape(count, 9)]
    ids = list(range(6, 6 + count))
    shelf.cupboard_geoms = {2, 3, 5, *ids[: count * 3 // 4]}
    shelf.box_geoms = set(ids[count * 3 // 4 :])
    return shelf


@pytest.mark.parametrize("seed", range(10))
def test_tool_clear_matches_the_per_obstacle_reference(*, seed):
    rng = np.random.default_rng(seed)
    shelf = cluttered_shelf(rng=rng)
    tool = RackScraper(shelf=shelf)
    outcomes = []
    for _ in range(80):
        pose = np.eye(4)
        pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=float(rng.uniform(-90, 90)))
        pose[:3, 3] = [rng.uniform(1.2, 1.8), rng.uniform(-0.3, 0.3), 0]
        pose[2, 3] += rng.uniform(0.85, 1.1) - tool.capsule_bottom(pose=pose)
        expected = reference_tool_clear(tool=tool, pose=pose)
        assert tool.tool_clear(pose=pose) == expected
        outcomes.append(expected)
    assert any(outcomes) and not all(outcomes)


def test_tool_clear_matches_the_reference_on_the_shipped_fixtures():
    """The hand-built support cases above, swept finely through their boundaries."""
    for cupboard in ({3}, {5}, {2, 3, 5}):
        shelf = FakeShelf()
        shelf.cupboard_geoms = set(cupboard)
        tool = RackScraper(shelf=shelf)
        for yaw in (-90, 0, 35):
            for x in np.linspace(1.3, 1.7, 9):
                for lift in np.linspace(-0.01, 0.01, 41):
                    pose = np.eye(4)
                    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=yaw)
                    pose[:3, 3] = [x, 0, 0]
                    pose[2, 3] += tool.top + lift - tool.capsule_bottom(pose=pose)
                    assert tool.tool_clear(pose=pose) == reference_tool_clear(tool=tool, pose=pose)


# The captured `gather_plus_y_push` refusal: one joint needs 0.1623 rad, the rest less.
UNEQUAL_ERROR = np.array([0.1623, 0.05, -0.08, 0.02, 0.0, -0.01, 0.04])


def test_a_joint_step_scales_every_joint_together_and_keeps_its_direction():
    current = np.linspace(-0.3, 0.3, 7)
    target = JointStep.toward(current=current, target=current + UNEQUAL_ERROR)
    step = target - current
    assert np.max(np.abs(step)) == pytest.approx(JointStep.LIMIT)
    np.testing.assert_allclose(step, UNEQUAL_ERROR * (JointStep.LIMIT / 0.1623))
    near = current + 0.5 * UNEQUAL_ERROR
    np.testing.assert_allclose(JointStep.toward(current=current, target=near), near)
    wrapped = JointStep.toward(current=np.full(7, 3.1), target=np.full(7, -3.1))
    np.testing.assert_allclose(wrapped - 3.1, np.full(7, 2 * np.pi - 6.2))


def test_servo_commands_the_direction_preserving_joint_step():
    robot = SimpleNamespace(
        _update_ctrl=Mock(),
        _compute_arm_torques=Mock(return_value=np.zeros(7)),
        ARM_TORQUE_LIMITS=np.full(7, 2.0),
    )
    shelf = SimpleNamespace(
        core=SimpleNamespace(_robot_env=robot),
        model=SimpleNamespace(nv=7, site_bodyid=np.array([0])),
        data=SimpleNamespace(qpos=np.zeros(7)),
        site=0,
        qa=np.arange(7),
        arm_reference=UNEQUAL_ERROR.copy(),
        ticks=0,
    )
    compute = robot._compute_arm_torques
    with SweepServo(shelf=shelf, max_normal_force_n=0):
        robot._update_ctrl(np.zeros(11))
    np.testing.assert_allclose(
        compute.call_args.kwargs["target_positions"],
        JointStep.toward(current=np.zeros(7), target=UNEQUAL_ERROR),
    )


def _guarded_scraper(*, arm, base):
    tool = RackScraper.__new__(RackScraper)
    tool.shelf = SimpleNamespace(
        in_hand=None,
        data=SimpleNamespace(qpos=np.r_[base, arm]),
        ba=np.arange(3),
        qa=np.arange(3, 10),
        base_target=base.copy(),
        base_step_limit=0.015,
        step=Mock(),
    )
    tool.check_configuration = Mock()
    return tool


def test_the_step_guard_checks_the_command_the_servo_will_send():
    arm, base = np.linspace(-0.3, 0.3, 7), np.array([1.5, 0.85, -np.pi / 2])
    tool = _guarded_scraper(arm=arm, base=base)
    tool.step(arm=arm + UNEQUAL_ERROR, label="gather_plus_y_push")
    checked = [call.kwargs["arm"] for call in tool.check_configuration.call_args_list]
    np.testing.assert_allclose(
        checked[-1], JointStep.toward(current=arm, target=arm + UNEQUAL_ERROR)
    )
    for configuration in checked:
        moved = configuration - arm
        np.testing.assert_allclose(moved, UNEQUAL_ERROR * (moved[0] / 0.1623), atol=1e-12)
    tool.shelf.step.assert_called_once()


def _off_the_straight_path(*, arm, start, delta):
    moved = arm - start
    scale = moved[0] / delta[0]
    return not np.allclose(moved, scale * delta, atol=1e-12)


def test_an_executed_segment_is_checked_along_the_path_the_servo_follows():
    """Ours seed 0 (bulk-only-v1), 18th action: a Sweep refused 30 steps in with
    "sweep_arm_clearance: axis-clipped joint segment at 0.225 rad". The transit had
    planned its route, driven the arm to within 0.02 rad of the last waypoint, and
    re-checked the final segment from where the arm then was: the straight path was
    clear, and the axis-clipped one (which `JointStep` never drives) put one blade
    corner 0.07 mm into the rack's side."""
    arm, base = np.linspace(-0.3, 0.3, 7), np.array([1.5, 0.85, -np.pi / 2])
    delta = np.array([0.225, -0.1924, 0.225, -0.225, -0.225, -0.225, 0.225]) * 1.4
    tool = _guarded_scraper(arm=arm, base=base)
    tool.check_arm_segment(
        start=arm, target=arm + delta, base=base, holding=True, axis_clipped=False
    )
    checked = [call.kwargs["arm"] for call in tool.check_configuration.call_args_list]
    assert len(checked) > 2
    assert not any(_off_the_straight_path(arm=c, start=arm, delta=delta) for c in checked)


def test_a_planned_segment_is_still_checked_along_the_axis_clipped_path_too():
    arm, base = np.linspace(-0.3, 0.3, 7), np.array([1.5, 0.85, -np.pi / 2])
    delta = np.array([0.225, -0.1924, 0.225, -0.225, -0.225, -0.225, 0.225]) * 1.4
    tool = _guarded_scraper(arm=arm, base=base)
    tool.check_arm_segment(start=arm, target=arm + delta, base=base, holding=True)
    checked = [call.kwargs["arm"] for call in tool.check_configuration.call_args_list]
    assert any(_off_the_straight_path(arm=c, start=arm, delta=delta) for c in checked)


def test_a_transit_rechecks_each_executed_segment_along_the_servo_path_only():
    arm, base = np.linspace(-0.3, 0.3, 7), np.array([1.5, 0.85, -np.pi / 2])
    tool = _guarded_scraper(arm=arm, base=base)
    target = arm + 0.01
    tool.check_arm_segment = Mock()
    tool.settle_hand = Mock()
    planner = SimpleNamespace(
        plan=Mock(return_value=[arm + 0.005, target]),
        delta=Mock(return_value=np.zeros(7)),
        edge_checks=0,
        last_failure="",
    )
    with (
        patch.object(TableWiper, "ik", return_value=target),
        patch("hitl_pmp.environments.racktoserving3d.tool.ArmTransit", return_value=planner),
    ):
        tool.transit(
            position=np.zeros(3), rotation=np.eye(3), label="sweep_arm_clearance", holding=False
        )
    assert len(tool.check_arm_segment.call_args_list) == 2
    for call in tool.check_arm_segment.call_args_list:
        assert call.kwargs["axis_clipped"] is False


def test_a_blade_grazing_the_racks_front_face_as_it_leaves_is_not_a_collision():
    """Ours seed 0 (bulk-only-v1), 45th action: a Sweep that had moved all five foods
    was refused at step 226, "Scraper blade or handle would collide with fixture
    geometry", on its own measured pose. The blade had just left the rack's front
    edge and sagged 3.8 mm below the rack top, with its back face still less than
    0.005 mm inside the rack's front face."""
    shelf = FakeShelf()
    shelf.cupboard_geoms = {3}
    tool = RackScraper(shelf=shelf)
    pose = np.eye(4)
    pose[:3, :3] = tool.sweep_body_rotation(yaw_deg=0)
    pose[:3, 3] = [1.5, 0, 0]
    pose[2, 3] += tool.top - 0.0038 - tool.capsule_bottom(pose=pose)
    assert not tool.tool_clear(pose=pose)
    inside, outside = 1.5, 1.2
    for _ in range(40):
        middle = (inside + outside) / 2
        pose[0, 3] = middle
        if tool.tool_clear(pose=pose):
            outside = middle
        else:
            inside = middle
    pose[0, 3] = inside
    model, data, geom = shelf.model, shelf.data, tool.head_geom
    depth = box_overlap(
        center=pose[:3, :3] @ model.geom_pos[geom] + pose[:3, 3],
        rotation=pose[:3, :3] @ tool.geom_rotation(geom=geom),
        size=model.geom_size[geom],
        other_center=data.geom_xpos[3],
        other_rotation=data.geom_xmat[3].reshape(3, 3),
        other_size=model.geom_size[3],
    )
    assert depth == pytest.approx(1e-4, abs=1e-6)


def hand_settling_tool(*, ends_at: np.ndarray):
    """A tool whose hand stays `ends_at` from the origin however long it settles."""
    tool = TableWiper.__new__(TableWiper)
    pose = np.eye(4)
    pose[:3, 3] = ends_at
    tool.shelf = SimpleNamespace(site_pose=Mock(return_value=pose))
    tool.step = Mock()
    return tool


def test_a_hand_held_up_while_levelling_onto_food_is_not_a_tracking_error():
    """Ours seed 0 (bulk-only-v2), Sweep(0.06): "sweep_level: hand tracking error 5.9 mm /
    0.9 deg". The blade rested on two dumplings, so the hand stopped 5.82 mm above its
    target and under 1 mm off it horizontally, with every joint within 0.007 rad."""
    tool = hand_settling_tool(ends_at=np.array([-0.00087, -0.00076, 0.00582]))
    tool.settle_hand(
        arm=np.zeros(7),
        position=np.zeros(3),
        rotation=np.eye(3),
        label="sweep_level",
        limit=3,
        rest_above=0.010,
    )


@pytest.mark.parametrize(
    ("ends_at", "rest_above"),
    [
        ([-0.00087, -0.00076, 0.00582], 0.0),
        ([-0.00087, -0.00076, -0.00582], 0.010),
        ([-0.00600, -0.00076, 0.00582], 0.010),
        ([0.0, 0.0, 0.0200], 0.010),
    ],
    ids=["not-allowed", "below", "sideways", "far-above"],
)
def test_only_a_hand_held_just_above_its_target_is_excused(*, ends_at, rest_above):
    tool = hand_settling_tool(ends_at=np.array(ends_at))
    with pytest.raises(SkillFailed, match="hand tracking error"):
        tool.settle_hand(
            arm=np.zeros(7),
            position=np.zeros(3),
            rotation=np.eye(3),
            label="sweep_level",
            limit=3,
            rest_above=rest_above,
        )


def _line_tool():
    """A scraper at rest whose lines neither step physics nor wait for the hand."""
    tool = RackScraper.__new__(RackScraper)
    tool.shelf = SimpleNamespace(
        site_pose=Mock(return_value=np.eye(4)),
        data=SimpleNamespace(qpos=np.zeros(10)),
        qa=np.arange(3, 10),
        base_target=np.zeros(3),
        in_hand=None,
    )
    tool.step = Mock()
    tool.settle_hand = Mock()
    return tool


@pytest.mark.parametrize(("requested", "checked"), [({}, True), ({"axis_clipped": False}, False)])
def test_a_line_passes_on_which_joint_paths_its_waypoints_are_checked_along(*, requested, checked):
    """Session 10 learning curves, 29/531 Sweep tries: "sweep_level: Cartesian fraction
    0.500: axis-clipped joint segment at 0.025 rad: Scraper blade or handle would collide
    with fixture geometry". The servo follows the straight joint segment, which was clear;
    the axis-clipped one put the blade 3.13 mm under the rack top against a 3 mm allowance."""
    tool = _line_tool()
    tool.check_arm_segment = Mock()
    with patch.object(TableWiper, "ik", return_value=np.full(7, 0.01)):
        tool.line(
            position=np.array([0.02, 0.0, 0.0]),
            rotation=np.eye(3),
            label="sweep_level",
            holding=False,
            **requested,
        )
    calls = tool.check_arm_segment.call_args_list
    assert len(calls) == 2
    assert all(call.kwargs["axis_clipped"] is checked for call in calls)


def test_a_line_checked_on_the_servo_path_alone_still_refuses_a_collision_on_it():
    tool = _line_tool()
    target = np.full(7, 0.05)

    def check_configuration(*, base, arm, holding, inflate=0.0):
        if np.allclose(arm, target / 2):
            raise SkillFailed("Scraper blade or handle would collide with fixture geometry")

    tool.check_configuration = Mock(side_effect=check_configuration)
    with (
        patch.object(TableWiper, "ik", return_value=target),
        pytest.raises(SkillFailed, match=r"sweep_level: .*straight joint segment.*would collide"),
    ):
        tool.line(
            position=np.array([0.005, 0.0, 0.0]),
            rotation=np.eye(3),
            label="sweep_level",
            holding=False,
            axis_clipped=False,
        )
    tool.step.assert_not_called()
