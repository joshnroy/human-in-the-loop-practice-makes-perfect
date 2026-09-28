"""Simulator-backed checks of the repositioning moves and the grasps they make room for.

Each test lays the cubes out by hand in a state the sweep is known to produce -- the
numbers are from the 32-seed measurement -- so it needs no sweep and tests one move.
"""

import importlib.util
from typing import Any

import numpy as np
import pytest

needs_kinder = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="KINDER is an optional extra",
)

DRAWER_FLOOR = 0.2275
FRONT_WALL = 0.872


def _lay_out(
    *, session: Any, drawer: float = 0.0, cubes: dict[str, tuple[float, float, float, float]]
) -> None:
    """Put the drawer at `drawer` and each named cube at (x, y, z, yaw in degrees)."""
    from scipy.spatial.transform import Rotation

    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    # the drawer first, the cubes once it is there: set together, the cubes are put down
    # before the drawer is under them
    state = session.state.copy()
    state.set(state.get_object_from_name(S.DRAWER), "pos", drawer)
    session.restore(state=state)
    assert session.drawer_pos() == pytest.approx(drawer, abs=0.005), (
        "the drawer did not stay where it was put: is the robot standing in its way?"
    )
    state = session.state.copy()
    for name, (x, y, z, yaw) in cubes.items():
        obj = state.get_object_from_name(name)
        q = Rotation.from_euler("z", yaw, degrees=True).as_quat()
        for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), (x, y, z, *q), strict=True):
            state.set(obj, key, float(value))
    session.restore(state=state)
    for _ in range(5):
        session.step(action=np.zeros(11))


def _reset(*, seed: int = 6, without: tuple[str, ...] = ()) -> tuple[Any, Any]:
    """Seed 6 starts the robot 1.69 m out, clear of the drawer even when it is open;
    seed 1 starts it where an opened drawer would be inside its chassis."""
    from hitl_pmp.environments.sweep_drawer3d.self_reset import SweepDrawerSelfReset
    from hitl_pmp.environments.sweep_drawer3d.session import SweepDrawerSession

    session = SweepDrawerSession(seed=seed)
    return session, SweepDrawerSelfReset(session=session, without=frozenset(without))


@needs_kinder
def test_sharp_closing_strokes_shake_cubes_off_the_drawers_front_wall() -> None:
    """Seed 5: the sweep leaves cubes 1.5 to 4.0 cm from the front wall, where the palm
    cannot come down. The robot holds the handle and slams the drawer shut; the cubes
    slide on toward the back."""
    session, reset = _reset()
    try:
        z = DRAWER_FLOOR + 0.01
        wall = FRONT_WALL + 0.25
        _lay_out(
            session=session,
            drawer=0.25,
            cubes={
                "cube_3": (wall - 0.015, 0.045, z, 0.0),
                "cube_1": (wall - 0.019, 0.077, z, 0.0),
                "cube_4": (wall - 0.039, 0.105, z, 0.0),
            },
        )
        rep = reset.repositioning
        before = rep.wall_gaps()
        assert set(before) == {"cube_1", "cube_3", "cube_4"}
        assert min(before.values()) < 0.02
        rep.wiggle_drawer()
        after = rep.wall_gaps()
        assert set(after) == set(before), "a cube left the drawer"
        assert min(after.values()) >= 0.06
        assert session.drawer_pos() > 0.2, "the drawer is left open for the picks"
        assert session.finger_closure() < 0.1, "the handle is let go"
    finally:
        session.close()


@needs_kinder
def test_a_closing_stroke_shorter_than_the_grips_slack_still_shuts_the_drawer() -> None:
    """Seed 1: the sweep leaves the drawer 1.5 cm open. A stroke of exactly 1.5 cm moved
    it 4 mm and the reset ended with it 1.01 cm open, over its own 1 cm criterion."""
    from hitl_pmp.environments.sweep_drawer3d.primitives import DrawerStroke

    session, reset = _reset()
    try:
        _lay_out(session=session, drawer=0.015, cubes={})
        assert session.drawer_pos() == pytest.approx(0.015, abs=0.003)
        reset.primitives.move_drawer(target=0.0)
        assert session.drawer_pos() < DrawerStroke.CLOSED_TOLERANCE
    finally:
        session.close()


@needs_kinder
def test_asking_whether_a_pick_is_feasible_moves_nothing() -> None:
    session, reset = _reset()
    try:
        base, arm, ticks = session.base(), session.arm(), session.ticks
        cubes = {c: session.position(name=c) for c in session.locations()}
        assert all(reset.primitives.pick_feasible(cube=c) for c in cubes)
        assert session.ticks == ticks
        assert session.base() == base
        assert np.allclose(session.arm(), arm)
        assert all(np.allclose(session.position(name=c), p) for c, p in cubes.items())
    finally:
        session.close()


@needs_kinder
def test_a_cube_under_the_countertop_in_the_shut_drawer_has_no_feasible_pick() -> None:
    session, reset = _reset()
    try:
        _lay_out(
            session=session,
            drawer=0.0,
            cubes={"cube_4": (FRONT_WALL - 0.03, 0.0, DRAWER_FLOOR + 0.01, 0.0)},
        )
        assert session.location(cube="cube_4") == "drawer"
        assert not reset.primitives.pick_feasible(cube="cube_4")
    finally:
        session.close()


@needs_kinder
def test_a_turned_floor_cube_against_a_drawer_face_is_picked_by_closing_along_the_face() -> None:
    """Seed 7: a floor cube 1.7 cm from the lower drawers' faces, under a handle, turned
    33 degrees off them. Along its own faces a finger would have to go where the drawer
    face is; closing along the face, the pads turn the cube square and lift it."""
    session, reset = _reset()
    try:
        _lay_out(session=session, cubes={"cube_4": (0.9118, 0.0046, 0.01, -147.2)})
        assert session.location(cube="cube_4") == "floor"
        prims = reset.primitives
        found = prims.find_pick(cube="cube_4")
        closing = np.degrees(found.grasp.yaw) % 180
        assert 60 < closing < 120, "the closing axis runs along the drawer faces"
        assert found.grasp.approach[0] < -0.3, "the palm leans out from the island"
        prims.pick(cube="cube_4")
        assert session.position(name="cube_4")[2] > 0.07
    finally:
        session.close()


@needs_kinder
def test_without_the_squeeze_grasp_that_cube_has_no_pick() -> None:
    """The counterfactual: the same cube, the same scene, the closing axis taken only
    from the cube's own faces."""
    session, reset = _reset(without=("squeeze",))
    try:
        _lay_out(session=session, cubes={"cube_4": (0.9118, 0.0046, 0.01, -147.2)})
        assert not reset.primitives.pick_feasible(cube="cube_4")
    finally:
        session.close()


@needs_kinder
def test_without_the_other_drawers_the_planner_reaches_through_a_drawer_face() -> None:
    """A fingertip placed 3 cm inside the lower middle drawer's face collides with
    nothing in a scene that models only the task's drawer."""
    from pybullet_helpers.geometry import Pose

    from hitl_pmp.environments.sweep_drawer3d.planning_scene import Orientations
    from hitl_pmp.environments.sweep_drawer3d.types import SweepDrawerScene as S

    inside = Pose(
        (0.865 + 0.048, 0.2, 0.10),
        Orientations.from_axes(
            closing=np.array([0.0, 1.0, 0.0]), approach=np.array([-1.0, 0.0, 0.0])
        ),
    )
    verdicts = {}
    for without in ((), ("drawers",)):
        session, reset = _reset(without=without)
        try:
            scene = reset._scene
            scene.sync(base=(1.55, 0.2, float(np.pi)))
            joints = scene.ik(pose=inside, seed=S.HOME)
            assert joints is not None
            verdicts[without] = scene.in_collision(
                joints=scene.fingers(arm=joints, state=0.8), bodies=scene.bodies()
            )
        finally:
            session.close()
    assert verdicts[()] is True
    assert verdicts[("drawers",)] is False


@needs_kinder
def test_two_floor_cubes_a_finger_cannot_part_are_picked_without_being_pushed() -> None:
    """Seed 11: two floor cubes 6 mm apart, 6 cm from the island. Along their own faces
    one closing axis puts a finger between them and the other puts the knuckles in the
    drawer face. A closing axis taken from the surroundings fits, so nothing is pushed."""
    session, reset = _reset()
    try:
        _lay_out(
            session=session,
            cubes={"cube_3": (0.9531, 0.1541, 0.01, 0.0), "cube_4": (0.9612, 0.1295, 0.01, 90.0)},
        )
        prims, rep = reset.primitives, reset.repositioning
        assert prims.pick_feasible(cube="cube_3")
        assert prims.pick_feasible(cube="cube_4")
        assert rep.nudge_candidates(cube="cube_3") == []
    finally:
        session.close()


def _cluster() -> dict[str, tuple[float, float, float, float]]:
    """Seed 5's cubes as the wiggle left them: 7 to 10 cm off the front wall, in two rows
    with 0 to 9 mm between neighbours."""
    z = DRAWER_FLOOR + 0.01
    return {
        "cube_0": (1.024, 0.105, z, 0.0),
        "cube_1": (1.045, 0.077, z, 0.0),
        "cube_2": (1.015, 0.056, z, 0.0),
        "cube_3": (1.044, 0.046, z, 0.0),
        "cube_4": (1.022, 0.079, z, 0.0),
    }


@needs_kinder
def test_a_fingertip_push_parts_a_cluster_the_wiggle_left_packed() -> None:
    """The wiggle slides every cube the same way, so it takes a cluster off the wall
    without parting it. One push with the closed fingertips does."""
    session, reset = _reset()
    try:
        _lay_out(session=session, drawer=0.241, cubes=_cluster())
        prims, rep = reset.primitives, reset.repositioning
        before = [c for c in _cluster() if prims.pick_feasible(cube=c)]
        assert len(before) == 1
        plans = {c: rep.nudge_candidates(cube=c) for c in _cluster()}
        cube = max((c for c in plans if plans[c]), key=lambda c: plans[c][0].score)
        assert plans[cube][0].kind == "push"
        rep.nudge(cube=cube)
        assert all(session.location(cube=c) == "drawer" for c in _cluster())
        after = [c for c in _cluster() if prims.pick_feasible(cube=c)]
        assert len(after) > len(before)
    finally:
        session.close()


@needs_kinder
def test_a_block_no_push_is_expected_to_part_is_stirred() -> None:
    """Four cubes packed square, faces 0.5 mm apart: in the rigid 2D model every push
    moves the block whole, so none promises a grasp. Real cubes scatter, so a cube with
    no grasp is offered the push anyway, ranked below any push that does promise one."""
    session, reset = _reset()
    try:
        _lay_out(
            session=session,
            cubes={
                "cube_1": (1.05, 0.10, 0.01, 0.0),
                "cube_2": (1.0705, 0.10, 0.01, 0.0),
                "cube_3": (1.05, 0.1205, 0.01, 0.0),
                "cube_4": (1.0705, 0.1205, 0.01, 0.0),
            },
        )
        prims, rep = reset.primitives, reset.repositioning
        block = ("cube_1", "cube_2", "cube_3", "cube_4")
        assert not any(prims.graspable(cube=c) for c in block)
        for cube in block:
            plans = rep.nudge_candidates(cube=cube)
            assert plans
            assert {p.kind for p in plans} == {"stir"}
            assert all(p.score < 0 for p in plans)
    finally:
        session.close()


@needs_kinder
def test_letting_go_of_a_cube_does_not_shove_the_cube_in_the_next_slot() -> None:
    """Seed 5: opened fully to let go, a pad swung out 5 cm and pushed the cube placed
    4.5 cm away to x = 0.826, outside the pile region's 0.825."""
    session, reset = _reset()
    try:
        _lay_out(
            session=session,
            cubes={
                "cube_1": (0.806, -0.080, 0.47, 0.0),
                "cube_2": (1.05, 0.10, 0.01, 0.0),
                "cube_0": (0.70, -0.15, 0.47, 0.0),
                "cube_3": (0.66, -0.05, 0.47, 0.0),
                "cube_4": (0.66, -0.15, 0.47, 0.0),
            },
        )
        prims = reset.primitives
        neighbour = session.position(name="cube_1").copy()
        held, _ = prims.pick(cube="cube_2")
        prims.place(cube="cube_2", target_xy=(0.761, -0.080), ee_to_cube=held)
        assert session.in_pile(cube="cube_2")
        moved = float(np.linalg.norm(session.position(name="cube_1")[:2] - neighbour[:2]))
        assert moved < 0.003
        assert session.in_pile(cube="cube_1")
    finally:
        session.close()
