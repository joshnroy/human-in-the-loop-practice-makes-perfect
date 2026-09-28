"""The 2D pre-filters: they may only ever reject grasps that physically cannot fit."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)


def _fp():
    from hitl_pmp.environments.sweep_drawer3d.footprints import Footprints

    return Footprints


def test_support_distance_is_half_width_face_on_and_half_diagonal_corner_on() -> None:
    fp = _fp()
    assert fp.support_distance(yaw=0.0, direction=np.array([1.0, 0.0])) == pytest.approx(0.01)
    corner = np.array([1.0, 1.0]) / np.sqrt(2)
    assert fp.support_distance(yaw=0.0, direction=corner) == pytest.approx(0.01 * np.sqrt(2))


def test_an_isolated_cube_leaves_room_for_both_fingers() -> None:
    fp = _fp()
    assert fp.finger_clearance(center=np.zeros(2), yaw=0.0, obstacles=[]) > 0.0


def test_a_neighbour_6mm_away_along_the_closing_axis_blocks_that_axis_only() -> None:
    """The measured packing in the drawer: face gaps of ~6 mm between swept cubes."""
    fp = _fp()
    neighbour = fp.cube(center=np.array([0.026, 0.0]), yaw=0.0)
    assert fp.finger_clearance(center=np.zeros(2), yaw=0.0, obstacles=[neighbour]) < 0.0
    assert fp.finger_clearance(center=np.zeros(2), yaw=np.pi / 2, obstacles=[neighbour]) > 0.0


def test_the_palm_cannot_hover_over_a_cube_pressed_to_the_drawer_front() -> None:
    """A cube 1.5 cm from the front wall (the sweep leaves them there) cannot take a
    vertical grasp -- the palm overhangs the wall -- but leaning the palm away frees it."""
    fp = _fp()
    walls = fp.drawer_walls(drawer_pos=0.25)
    wall_x = 0.872 + 0.25
    cube = np.array([wall_x - 0.03, 0.0])
    vertical = fp.palm(center=cube, yaw=np.pi / 2, lean=0.0, alpha=0.0)
    assert fp.clearance(shape=vertical, obstacles=walls) < 0.0
    # closing axis along +y, v = -x: lean +1 moves the palm toward -x, away from the wall
    leaning = fp.palm(center=cube, yaw=np.pi / 2, lean=1.0, alpha=0.7)
    assert fp.clearance(shape=leaning, obstacles=walls) > 0.0


def test_clearance_reports_overlap_as_negative() -> None:
    fp = _fp()
    a = fp.cube(center=np.zeros(2), yaw=0.0)
    b = fp.cube(center=np.array([0.01, 0.0]), yaw=0.0)
    far = fp.cube(center=np.array([0.05, 0.0]), yaw=0.0)
    assert fp.clearance(shape=a, obstacles=[b]) < 0.0
    assert fp.clearance(shape=a, obstacles=[far]) == pytest.approx(0.03)
