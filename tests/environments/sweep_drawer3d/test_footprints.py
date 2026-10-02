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


def _heading(*, quaternion: np.ndarray) -> float:
    from hitl_pmp.environments.sweep_drawer3d.session import CubeHeading

    return CubeHeading.of(quaternion=quaternion)


def _shadow(*, quaternion: np.ndarray, center: np.ndarray):
    """What the cube covers on the ground: its eight corners, turned and dropped. No
    angle convention enters it."""
    import itertools

    from scipy.spatial.transform import Rotation
    from shapely.geometry import MultiPoint

    corners = np.array(list(itertools.product((-0.01, 0.01), repeat=3)))
    world = Rotation.from_quat(quaternion).apply(corners)[:, :2] + center
    return MultiPoint([tuple(p) for p in world]).convex_hull


FACES_DOWN = {
    "the face it started on": (0.0, 0.0),
    "rolled a quarter turn about x": (90.0, 0.0),
    "rolled the other way about x": (-90.0, 0.0),
    "rolled a quarter turn about y": (0.0, 90.0),
    "rolled the other way about y": (0.0, -90.0),
    "upside down": (180.0, 0.0),
}


@pytest.mark.parametrize("face", sorted(FACES_DOWN))
@pytest.mark.parametrize("turn", [0.0, 18.5, 37.0, -44.0, 75.0])
def test_a_cubes_footprint_is_drawn_where_the_cube_is_whichever_face_it_rests_on(
    *, face: str, turn: float
) -> None:
    """The sweep tips cubes over: at the start of the reset 64 of the 160 cubes in seeds
    0 to 31 rested on another face than at the start of the episode. The first angle of
    a z-y-x Euler split is the footprint's turn only for a cube on its start face, and
    drew 57 of the 99 swept cubes' footprints off by more than 5% of their area."""
    from scipy.spatial.transform import Rotation

    fp = _fp()
    about_x, about_y = FACES_DOWN[face]
    onto_the_face = Rotation.from_euler("x", about_x, degrees=True) * Rotation.from_euler(
        "y", about_y, degrees=True
    )
    q = (Rotation.from_euler("z", turn, degrees=True) * onto_the_face).as_quat()
    center = np.array([0.87, 0.05])
    drawn = fp.cube(center=center, yaw=_heading(quaternion=q))
    real = _shadow(quaternion=q, center=center)
    assert drawn.symmetric_difference(real).area < 0.001 * real.area


def test_the_cube_that_was_lifted_with_its_neighbour_is_drawn_where_it_was() -> None:
    """Seed 4, cube_0, as logged when the reset began: on its side, its footprint turned
    18.5 degrees. Drawn square to the axes, the grasp filter passed a grasp whose pads
    overlapped the cube beside it, and the pick lifted both."""
    from scipy.spatial.transform import Rotation

    fp = _fp()
    q = Rotation.from_euler("zyx", [180.0, -18.5, 89.8], degrees=True).as_quat()
    heading = np.degrees(_heading(quaternion=q))
    assert (heading + 45.0) % 90.0 - 45.0 == pytest.approx(-18.5, abs=0.3)
    center = np.array([0.8685, 0.0530])
    drawn = fp.cube(center=center, yaw=_heading(quaternion=q))
    real = _shadow(quaternion=q, center=center)
    assert drawn.symmetric_difference(real).area < 0.02 * real.area


def test_a_cube_nobody_tipped_keeps_the_yaw_it_has() -> None:
    """Both of its horizontal axes lie flat; taking the second would report the same
    footprint a quarter turn on, and a cube put back to a given yaw would be turned."""
    from scipy.spatial.transform import Rotation

    for yaw in (-2.9, -0.4, 0.0, 0.3, 1.2, 3.0):
        q = Rotation.from_euler("z", yaw).as_quat()
        assert _heading(quaternion=q) == pytest.approx(yaw, abs=1e-9)


def test_resting_jitter_does_not_flip_the_heading_by_a_quarter_turn() -> None:
    """A resting cube's rotation wanders by tenths of a degree from tick to tick."""
    from scipy.spatial.transform import Rotation

    rest = Rotation.from_euler("z", 0.3)
    for wobble in ((0.2, -0.1), (-0.2, 0.1), (0.05, 0.3), (0.3, 0.05)):
        q = (rest * Rotation.from_euler("xy", wobble, degrees=True)).as_quat()
        assert _heading(quaternion=q) == pytest.approx(0.3, abs=0.01)


def test_two_headings_a_quarter_turn_apart_are_the_same_footprint() -> None:
    from hitl_pmp.environments.sweep_drawer3d.session import CubeHeading

    apart = CubeHeading.quarter_turns_apart
    assert apart(a=0.3, b=0.3 + np.pi / 2) == pytest.approx(0.0, abs=1e-9)
    assert apart(a=0.3, b=0.3 - np.pi) == pytest.approx(0.0, abs=1e-9)
    assert apart(a=0.1, b=-0.1) == pytest.approx(0.2)
    assert apart(a=0.0, b=np.radians(50.0)) == pytest.approx(np.radians(40.0))
