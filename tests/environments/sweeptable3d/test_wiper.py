"""Capsule wiper geometry and home classification without a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweeptable3d.wiper import BODY_REST, TableWiper


class FakeShelf:
    """Only model geometry, live poses, velocities and contacts are provided."""

    def __init__(self, *, radius: float = 0.01, handle_half_width: float = 0.0075):
        self.core = object()
        self.model = SimpleNamespace(
            geom_type=np.array([3, 6, 6, 6]),
            geom_size=np.array([
                [radius, 0.1 - radius, 0],
                [handle_half_width, handle_half_width, 0.04],
                [0.15, 0.55, 0.01],
                [0.01, 0.01, 0.01],
            ]),
            geom_pos=np.array([[0, 0, radius], [0, 0, 2 * radius + 0.04], [0, 0, 0], [0, 0, 0]]),
            geom_quat=np.array([
                [np.sqrt(0.5), 0, np.sqrt(0.5), 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ]),
            body_jntadr=np.array([0]),
            jnt_dofadr=np.array([0]),
        )
        self.data = SimpleNamespace(
            qvel=np.zeros(6),
            contact=[],
            ncon=0,
            xmat=np.eye(3).reshape(1, 9),
            xpos=np.zeros((1, 3)),
        )
        self.wiper_geoms = [0, 1]
        self.wiper_body = 0
        self.head_geom = 1
        self.handle_geom = 0
        self.cupboard_geoms = {2}
        self.robot_geoms = {3}
        self.pad_geoms = {3}
        self.board = {1: 0.30}
        self.front_x = 1.325
        self.in_hand = None
        self.recorder = object()

    def wiper_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[0].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[0]
        return pose

    def set_pose(self, *, pose):
        self.data.xmat[0] = pose[:3, :3].reshape(9)
        self.data.xpos[0] = pose[:3, 3]

    def support(self, *, height: float = 0.30):
        self.data.contact = [
            SimpleNamespace(
                geom1=0,
                geom2=2,
                dist=0.0,
                pos=np.array([1.34, -0.42, height]),
                frame=np.eye(3)[[2, 0, 1]].reshape(9),
            )
        ]
        self.data.ncon = len(self.data.contact)


@pytest.mark.parametrize("rest_y", [-0.42, -0.38])
def test_rest_pose_and_capsule_grasp_offsets(*, rest_y):
    shelf = FakeShelf()
    recorder = shelf.recorder
    wiper = TableWiper(shelf=shelf, rest_y=rest_y)
    pose = wiper.rest_pose()
    np.testing.assert_allclose(pose[:3, :3], BODY_REST, atol=1e-15)
    np.testing.assert_allclose(wiper.base, [0.60, rest_y, 0])
    np.testing.assert_allclose(wiper.point, [0, 0, 0.09])
    np.testing.assert_allclose(wiper.handle_end(pose=pose), [shelf.front_x - 0.059, rest_y, 0.31])
    assert wiper.capsule_bottom(pose=pose) == pytest.approx(0.30)
    assert wiper.handle_corners(pose=pose)[:, 2].min() == pytest.approx(0.3025)
    assert wiper.core is shelf.core
    assert wiper.model is shelf.model
    assert wiper.data is shelf.data
    assert shelf.recorder is recorder
    assert (shelf.head_geom, shelf.handle_geom) == (0, 1)


@pytest.mark.parametrize("radius,half_width", [(0.012, 0.0067), (0.01, 0.014)])
def test_rest_height_uses_measured_capsule_and_handle(*, radius, half_width):
    shelf = FakeShelf(radius=radius, handle_half_width=half_width)
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    head_bottom = wiper.capsule_bottom(pose=pose)
    handle_bottom = wiper.handle_corners(pose=pose)[:, 2].min()
    assert min(head_bottom, handle_bottom) == pytest.approx(shelf.board[1])
    assert pose[2, 3] == pytest.approx(shelf.board[1] + max(radius, half_width))
    np.testing.assert_allclose(wiper.point, [0, 0, 2 * radius + 0.07])
    assert wiper.handle_end(pose=pose)[0] == pytest.approx(shelf.front_x - 0.059)


def test_home_accepts_real_contact_with_handle_above_capsule_support():
    shelf = FakeShelf(handle_half_width=0.0067)
    wiper = TableWiper(shelf=shelf)
    shelf.set_pose(pose=wiper.rest_pose())
    shelf.support()
    assert wiper.handle_corners()[:, 2].min() - shelf.board[1] == pytest.approx(0.0033)
    assert wiper.at_home()


@pytest.mark.parametrize("height", [-0.006, 0.006])
def test_home_rejects_penetration_and_floating_even_with_stale_contact(*, height):
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    pose[2, 3] += height
    shelf.set_pose(pose=pose)
    shelf.support()
    assert not wiper.at_home()


def test_home_requires_actual_tabletop_contact():
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    shelf.set_pose(pose=wiper.rest_pose())
    assert not wiper.at_home()
    shelf.support(height=0.20)
    assert not wiper.at_home()
    shelf.support()
    shelf.data.contact[0].frame = np.eye(3).reshape(9)
    assert not wiper.at_home()


@pytest.mark.parametrize("axis,speed", [(0, 0.02), (3, 0.2)])
def test_home_requires_linear_and_angular_rest(*, axis, speed):
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    shelf.set_pose(pose=wiper.rest_pose())
    shelf.support()
    shelf.data.qvel[axis] = speed
    assert not wiper.at_home()


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_home_rejects_tilt_roll_and_wrong_outward_direction(*, axis):
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    pose[:3, :3] = Rotation.from_euler(axis, 10, degrees=True).as_matrix() @ BODY_REST
    shelf.set_pose(pose=pose)
    shelf.support()
    assert not wiper.at_home()


@pytest.mark.parametrize("shift", [[0.015, 0, 0], [-0.02, 0, 0], [0, 0.08, 0]])
def test_home_requires_safe_overhang_and_parking_location(*, shift):
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    pose = wiper.rest_pose()
    pose[:3, 3] += shift
    shelf.set_pose(pose=pose)
    shelf.support()
    assert not wiper.at_home()


def test_home_requires_gripper_clear_and_released():
    shelf = FakeShelf()
    wiper = TableWiper(shelf=shelf)
    shelf.set_pose(pose=wiper.rest_pose())
    shelf.support()
    shelf.in_hand = np.eye(4)
    assert not wiper.at_home()
    shelf.in_hand = None
    shelf.data.contact.append(SimpleNamespace(geom1=1, geom2=3, dist=0.0))
    shelf.data.ncon = 2
    assert not wiper.at_home()


@pytest.mark.parametrize("yaw_deg", [-90, -25, 0, 35, 90])
def test_sweep_rotation_is_rigid_with_45_degree_handle_tilt(*, yaw_deg):
    rotation = TableWiper.sweep_body_rotation(yaw_deg=yaw_deg)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert rotation[2, 2] == pytest.approx(np.sqrt(0.5))
    expected = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix() @ np.array([
        -np.sqrt(0.5),
        0,
        np.sqrt(0.5),
    ])
    np.testing.assert_allclose(rotation[:, 2], expected, atol=1e-15)
