"""Rack scraper geometry and compatibility checks without physics stepping.

Mock geometry follows the kitchen prototype's bench scraper and wiper tests.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.racktoserving3d.servo import SweepServo
from hitl_pmp.environments.racktoserving3d.tool import (
    BODY_REST,
    RackScraper,
    RackShelfSkills,
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


def test_return_uses_release_guard_without_requiring_new_shelf_methods():
    shelf = FakeShelf()
    tool = RackScraper(shelf=shelf)
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
        "place_wiper_raise",
        "place_wiper_carry",
        "place_wiper_untilt",
        "place_wiper_lower",
        "place_wiper_retreat",
        "place_wiper_clear",
    ]


def test_return_without_support_retains_the_physical_grasp():
    shelf = FakeShelf()
    tool = RackScraper(shelf=shelf)
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
