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


def test_by_the_island_the_fingers_and_the_palm_face_different_walls() -> None:
    """The lower drawers' faces begin 3.6 cm above the floor, at x = 0.895. A finger on a
    floor cube is below them; what it faces is the island's bottom board, 2 cm further
    back. The palm, 6.5 cm up, faces the drawer faces. Seed 7 leaves a cube 1.7 cm from
    those faces. The palm clears them only by leaning away, which points the fingertips
    toward them: past the plane of the faces, but short of the board."""
    fp = _fp()
    cube = np.array([0.9118, 0.0])
    along_the_faces, alpha = np.pi / 2, 0.7
    faces = fp.island_faces(drawer_pos=0.0)
    upright = fp.palm(center=cube, yaw=along_the_faces, lean=0.0, alpha=0.0)
    assert fp.clearance(shape=upright, obstacles=[faces]) < 0.0
    # closing axis along +y, v = -x: lean -1 carries the palm toward +x, off the faces
    leaning = fp.palm(center=cube, yaw=along_the_faces, lean=-1.0, alpha=alpha)
    assert fp.clearance(shape=leaning, obstacles=[faces]) > 0.003
    # the tilted fingers, as Primitives.grasp_candidates lays them out
    reach = 0.019 * np.sin(alpha)
    tilted = {
        "center": cube - 0.5 * reach * -1.0 * np.array([-1.0, 0.0]),
        "yaw": along_the_faces,
        "widen": reach,
    }
    assert fp.finger_clearance(**tilted, obstacles=[faces]) < 0.0
    assert fp.finger_clearance(**tilted, obstacles=[fp.island_board()]) > 0.002


def test_the_open_task_drawer_carries_the_palms_wall_out_with_it() -> None:
    fp = _fp()
    assert fp.island_faces(drawer_pos=0.25).bounds[2] == pytest.approx(0.895 + 0.25)
    assert fp.island_faces(drawer_pos=-0.001).bounds[2] == pytest.approx(0.895)
    assert fp.island_board().bounds[2] == pytest.approx(0.875)


def _scattered(*, seed: int) -> tuple[np.ndarray, np.ndarray, list]:
    """Finger placements around a cube, with neighbours and a wall crowding it."""
    fp = _fp()
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-0.01, 0.01, size=(40, 2))
    widen = rng.uniform(0.0, 0.013, size=40)
    obstacles = [
        fp.cube(center=rng.uniform(-0.06, 0.06, size=2), yaw=float(rng.uniform(0, np.pi)))
        for _ in range(4)
    ]
    obstacles += fp.drawer_walls(drawer_pos=0.25 - 0.872 - 0.04)
    return centres, widen, obstacles


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("yaw", [0.0, 0.6, np.pi / 2])
@pytest.mark.parametrize("gap", [0.032, 0.0343])
def test_clearances_for_many_placements_agree_with_the_one_at_a_time_clearance(
    *, seed: int, yaw: float, gap: float
) -> None:
    """The grasp search tries some 600 finger placements per cube; computed one by one
    through shapely's affine transforms they took 0.18 s a cube and 48 s a nudge."""
    fp = _fp()
    centres, widen, obstacles = _scattered(seed=seed)
    many = fp.finger_clearances(centres=centres, yaw=yaw, obstacles=obstacles, gap=gap, widen=widen)
    assert many.shape == (40,)
    for k in range(40):
        one = fp.finger_clearance(
            center=centres[k], yaw=yaw, obstacles=obstacles, gap=gap, widen=float(widen[k])
        )
        if one > 0:
            assert many[k] == pytest.approx(one, abs=1e-9)
        else:
            # overlapping: the scalar reports how much, the grasp search only that it does
            assert many[k] == 0.0


def test_a_neighbour_in_the_pads_way_as_they_close_blocks_the_grasp() -> None:
    """Seed 16: three counter cubes in a touching row. Closing across the row, the open
    fingers cleared the next cube's corner by 3 mm -- and the pads, closing from 1.6 cm
    to the cube's face at 1.0 cm, took that corner with them. The neighbour was lifted
    too, fell on the robot's base, and was thrown 1.9 m when the base next moved."""
    fp = _fp()
    neighbour = fp.cube(center=np.array([0.019, 0.003]), yaw=0.0)
    across_the_row = np.pi / 2
    where_they_start = fp.finger_clearances(
        centres=np.zeros((1, 2)),
        yaw=across_the_row,
        obstacles=[neighbour],
        gap=0.032,
        widen=np.zeros(1),
    )
    assert where_they_start[0] > 0.002
    what_they_sweep = fp.finger_clearances(
        centres=np.zeros((1, 2)),
        yaw=across_the_row,
        obstacles=[neighbour],
        gap=0.032,
        widen=np.zeros(1),
        inner=0.01,
    )
    assert what_they_sweep[0] == 0.0
    # held 5 mm off the cube's middle, away from the neighbour, the pads pass it by 2 mm
    aside = fp.finger_clearances(
        centres=np.array([[-0.005, 0.0]]),
        yaw=across_the_row,
        obstacles=[neighbour],
        gap=0.032,
        widen=np.zeros(1),
        inner=0.01,
    )
    assert aside[0] == pytest.approx(0.002)


def test_with_nothing_around_every_placement_is_clear() -> None:
    fp = _fp()
    many = fp.finger_clearances(
        centres=np.zeros((3, 2)), yaw=0.0, obstacles=[], gap=0.032, widen=np.zeros(3)
    )
    assert many.tolist() == [1.0, 1.0, 1.0]


def test_clearance_reports_overlap_as_negative() -> None:
    fp = _fp()
    a = fp.cube(center=np.zeros(2), yaw=0.0)
    b = fp.cube(center=np.array([0.01, 0.0]), yaw=0.0)
    far = fp.cube(center=np.array([0.05, 0.0]), yaw=0.0)
    assert fp.clearance(shape=a, obstacles=[b]) < 0.0
    assert fp.clearance(shape=a, obstacles=[far]) == pytest.approx(0.03)
