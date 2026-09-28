"""Closing the gripper across a cube that is turned off the closing axis."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)


def _squeeze():
    from hitl_pmp.environments.sweep_drawer3d.primitives import SqueezeGrasp

    return SqueezeGrasp


def test_a_square_on_cube_spans_its_own_width() -> None:
    assert _squeeze().span(cube_yaw=0.0, closing_yaw=0.0) == pytest.approx(0.02)
    assert _squeeze().span(cube_yaw=0.3, closing_yaw=0.3 + np.pi / 2) == pytest.approx(0.02)


def test_a_cube_turned_45_degrees_spans_its_diagonal() -> None:
    assert _squeeze().span(cube_yaw=np.pi / 4, closing_yaw=0.0) == pytest.approx(0.02 * np.sqrt(2))


def test_the_pads_open_wider_than_the_turned_cube_and_well_inside_the_grippers_range() -> None:
    """Seed 7's floor cube by the island is turned 33 degrees off the drawer faces."""
    s = _squeeze()
    gap = s.gap(cube_yaw=np.radians(-147.2), closing_yaw=np.pi / 2)
    assert gap > s.span(cube_yaw=np.radians(-147.2), closing_yaw=np.pi / 2)
    assert gap < 0.04
    assert gap < 0.085


def test_headings_a_face_grasp_already_covers_are_not_repeated() -> None:
    headings = _squeeze().headings(face_yaws=[0.0, np.pi / 2])
    degrees = sorted(round(float(np.degrees(h)), 1) for h in headings)
    assert degrees == [22.5, 45.0, 67.5, 112.5, 135.0, 157.5]


def test_a_turned_cube_by_a_wall_is_offered_a_closing_axis_along_the_wall() -> None:
    """Its own face directions would put a finger where the wall is; along the wall, both
    fingers stay beside the cube."""
    faces = [np.radians(-147.2), np.radians(-147.2) + np.pi / 2]
    headings = _squeeze().headings(face_yaws=faces)
    assert any(abs(h - np.pi / 2) < 1e-9 for h in headings)
    assert any(abs(h) < 1e-9 for h in headings)


def test_a_hand_shut_on_a_row_stands_wider_than_one_shut_on_a_cube() -> None:
    """The check of the shut hand against the cubes beside it uses the hand as it will
    be: around one cube the planning model's fingers are as far shut as they go."""
    from hitl_pmp.environments.sweep_drawer3d.primitives import ShutHand

    one = ShutHand.state(held=("cube_2",))
    row = ShutHand.state(held=("cube_2", "cube_0"))
    assert one == pytest.approx(ShutHand.STATE)
    assert row < one
