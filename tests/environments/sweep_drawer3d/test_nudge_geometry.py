"""Where the closed fingertips go to push a cube, and how much room that takes."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("shapely") is None, reason="shapely ships with KINDER"
)

SUPPORT = 0.2275
ACROSS, ALONG, DEPTH = 0.009, 0.011, 0.048


def _geometry():
    from hitl_pmp.environments.sweep_drawer3d.repositioning import NudgeGeometry

    return NudgeGeometry


def _tip_corners(*, position: np.ndarray, quat: tuple[float, ...]) -> np.ndarray:
    """The four corners of the closed pads' end face, from the measured pad box: 1.8 cm
    along the closing axis (x), 2.2 cm across it (y), 4.8 cm down the approach axis (z)."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_quat(quat).as_matrix()
    local = [(sx * ACROSS, sy * ALONG, DEPTH) for sx in (-1, 1) for sy in (-1, 1)]
    return np.array([position + r @ np.array(p) for p in local])


@pytest.mark.parametrize("alpha", [0.0, 0.45, 0.7, 0.9])
def test_the_fingertip_starts_clear_of_the_floor_and_behind_the_cube(*, alpha: float) -> None:
    g = _geometry()
    centre, push = np.array([1.05, 0.10]), np.array([-1.0, 0.0])
    pose = g.start(centre=centre, support=SUPPORT, rear=0.01, direction=push, alpha=alpha)
    corners = _tip_corners(position=np.array(pose.position), quat=pose.orientation)
    assert corners[:, 2].min() == pytest.approx(SUPPORT + g.FLOOR_CLEARANCE, abs=1e-6)
    # the cube's rear face, seen from the push direction, is at x = 1.06
    assert corners[:, 0].min() > 1.06


@pytest.mark.parametrize("alpha", [0.45, 0.7, 0.9])
def test_a_tilted_finger_does_not_overhang_the_cube_it_is_about_to_push(*, alpha: float) -> None:
    """The leading face leans over the cube; at the cube's top it must still be short of
    the rear face, or the start pose is already in contact."""
    g = _geometry()
    pose = g.start(
        centre=np.zeros(2), support=0.0, rear=0.01, direction=np.array([1.0, 0.0]), alpha=alpha
    )
    corners = _tip_corners(position=np.array(pose.position), quat=pose.orientation)
    low = corners[np.argmin(corners[:, 2])]
    at_cube_top = low[0] + (0.02 - low[2]) * np.tan(alpha)
    assert at_cube_top == pytest.approx(-0.01 - g.CONTACT_MARGIN, abs=1e-6)


def test_the_palm_leans_toward_the_push_so_it_is_away_from_a_wall_behind() -> None:
    from scipy.spatial.transform import Rotation

    g = _geometry()
    push = np.array([0.0, 1.0])
    pose = g.start(centre=np.zeros(2), support=0.0, rear=0.01, direction=push, alpha=0.7)
    approach = Rotation.from_quat(pose.orientation).as_matrix()[:, 2]
    # the approach axis runs palm -> tip: it points down and against the push
    assert approach[2] < 0
    assert approach[:2] @ push == pytest.approx(-np.sin(0.7))


def test_the_push_runs_across_the_closing_axis_where_the_gripper_is_slim() -> None:
    """Along the closing axis the knuckles flare to 6.5 cm; across it the closed gripper
    stays within 2 cm until the palm."""
    from scipy.spatial.transform import Rotation

    g = _geometry()
    push = np.array([-1.0, 0.0])
    pose = g.start(centre=np.zeros(2), support=0.0, rear=0.01, direction=push, alpha=0.0)
    closing = Rotation.from_quat(pose.orientation).as_matrix()[:, 0]
    assert abs(closing[:2] @ push) == pytest.approx(0.0, abs=1e-9)


def test_a_vertical_finger_needs_under_three_centimetres_behind_the_cube() -> None:
    g = _geometry()
    assert g.reach_behind(alpha=0.0) == pytest.approx(g.CONTACT_MARGIN + 0.022)
    assert g.reach_behind(alpha=0.0) < 0.03


def test_tilting_costs_room_behind_the_cube_but_never_more_than_four_centimetres() -> None:
    g = _geometry()
    reaches = [g.reach_behind(alpha=a) for a in (0.0, 0.45, 0.7, 0.9)]
    assert reaches == sorted(reaches)
    assert reaches[-1] < 0.04


def test_no_fingertip_fits_behind_a_floor_cube_that_lies_against_a_drawer_face() -> None:
    """Seed 7 leaves floor cubes with their rear faces 0.3 and 1.0 cm from the lower
    drawers' faces. Upright or tilted, the fingertip takes more room than that, which is
    why such a cube is pushed along the face instead."""
    g = _geometry()
    assert min(g.reach_behind(alpha=a) for a in (0.0, 0.45, 0.7, 0.9)) > 0.010


def test_the_stroke_ends_with_the_cube_moved_by_the_asked_distance() -> None:
    g = _geometry()
    push = np.array([1.0, 0.0])
    start = g.start(centre=np.zeros(2), support=0.0, rear=0.01, direction=push, alpha=0.45)
    stop = g.stop(start=start, direction=push, distance=0.06)
    travel = np.array(stop.position) - np.array(start.position)
    assert travel[2] == pytest.approx(0.0)
    assert travel[0] == pytest.approx(0.06 + g.CONTACT_MARGIN)
    assert stop.orientation == start.orientation


def _beside_a_wall(**overrides: object) -> dict[str, object]:
    """A push along +y past a wall on the -x side of a floor cube."""
    spec: dict[str, object] = {
        "centre": np.array([0.912, 0.0]),
        "support": 0.0,
        "rear": 0.01,
        "direction": np.array([0.0, 1.0]),
        "away": np.array([1.0, 0.0]),
        "alpha": 0.7,
        "lift": 0.004,
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize("direction", [np.array([0.0, 1.0]), np.array([0.0, -1.0])])
def test_beside_a_wall_the_palm_leans_away_from_it_whichever_way_the_push_runs(
    *,
    direction: np.ndarray,
) -> None:
    from scipy.spatial.transform import Rotation

    g = _geometry()
    pose = g.side_start(**_beside_a_wall(direction=direction))
    r = Rotation.from_quat(pose.orientation).as_matrix()
    approach, closing = r[:, 2], r[:, 0]
    # palm -> tip points down and toward the wall, so the palm is on the far side
    assert approach[2] < 0
    assert approach[0] == pytest.approx(-np.sin(0.7))
    # and the push runs along the closing axis: one finger's outer face does the pushing
    assert abs(closing[:2] @ direction) == pytest.approx(1.0)


def test_beside_a_wall_the_fingertip_is_centred_on_the_cube_and_clear_of_the_floor() -> None:
    g = _geometry()
    pose = g.side_start(**_beside_a_wall())
    corners = _tip_corners(position=np.array(pose.position), quat=pose.orientation)
    assert corners[:, 2].min() == pytest.approx(0.004, abs=1e-6)
    assert corners[:, 0].mean() == pytest.approx(0.912, abs=1e-6)
    # behind the cube along the push, by the contact margin and the finger's own reach
    assert corners[:, 1].max() < -0.01


def test_a_tilted_fingertip_beside_a_cube_flush_with_a_drawer_face_clears_that_face() -> None:
    """The lower drawers' faces start 3.6 cm above the floor, at x = 0.895; below them
    the island's bottom board stands 2 cm back, at 0.875. A cube flush with the face has
    its centre at 0.905, and nothing fits behind it -- but a fingertip beside it does."""
    g = _geometry()
    pose = g.side_start(**_beside_a_wall(centre=np.array([0.905, 0.0])))
    corners = _tip_corners(position=np.array(pose.position), quat=pose.orientation)
    assert corners[:, 0].min() > 0.875 + 0.003
    wall_side = corners[np.argmin(corners[:, 0])]
    assert wall_side[2] < 0.036
    # the finger rises from that edge leaning away from the wall
    at_face_bottom = wall_side[0] + (0.036 - wall_side[2]) * np.tan(0.7)
    assert at_face_bottom > 0.895 + 0.003


def test_shifting_the_fingertip_away_from_the_wall_moves_it_by_exactly_that() -> None:
    g = _geometry()
    a = g.side_start(**_beside_a_wall())
    b = g.side_start(**_beside_a_wall(shift=0.004))
    assert np.array(b.position) - np.array(a.position) == pytest.approx([0.004, 0.0, 0.0])
