"""A cube returned to the shelf is let go over a free spot, not one fixed point.

Before, every ReturnCubeToShelf* let go with the hand at `SHELF_PLACE_XY` (1.45, 0),
which lands the cube at about (1.496, 0). A cube already there is landed on: in the
new-scene skill benchmark's floor-out-right-mid case (cube_4), cube_2 lay at
(1.500, -0.003) and cube_4 came to rest on top of it, 309 mm up instead of on the board.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    Setups,
    StateCapture,
    StateSet,
)

STATES = Path(__file__).resolve().parents[3] / "benchmarks/sweepoffshelf3d/states.jsonl"
CUBES = [f"cube_{i}" for i in range(5)]
# The new-scene benchmark's floor-out-right-mid (867abd9e6, kindergarden 490831fa5):
# each cube's free-joint pose (x, y, z, qw, qx, qy, qz) and the base (x, y, yaw). Laid
# over main's state of the same name, whose wiper is at main's home.
NEW_SCENE_CUBES = {
    "cube_0": (1.490269, 0.138694, 0.289399, 0.397938, 0.0, 0.0, 0.917412),
    "cube_1": (1.528681, 0.212546, 0.289399, 0.913466, 0.0, 0.0, 0.406916),
    "cube_2": (1.499701, -0.003062, 0.289399, -0.991547, 0.0, 0.0, 0.129751),
    "cube_3": (1.478991, -0.13259, 0.289399, -0.429105, 0.0, 0.0, 0.903255),
    "cube_4": (1.235081, -0.279909, 0.009892, 1.0, 0.0, 0.0, 0.0),
}
NEW_SCENE_BASE = (6.925524e-05, 0.027063, 0.0)

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_shelf_spread"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    # The benchmark states were captured in the lab2 scene; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _restore_case(*, env, states):
    state = states["floor-out-right-mid"]
    env.reset_to_seed(seed=state.seed)
    m = env.backend().model()
    qpos = list(state.qpos)
    qvel = list(state.qvel)
    for cube, pose in NEW_SCENE_CUBES.items():
        joint = m.joint(f"{cube}_joint")
        qpos[m.jnt_qposadr[joint.id] : m.jnt_qposadr[joint.id] + 7] = pose
        qvel[m.jnt_dofadr[joint.id] : m.jnt_dofadr[joint.id] + 6] = [0.0] * 6
    joints = ("robot_joint_x", "robot_joint_y", "robot_joint_th")
    for name, value in zip(joints, NEW_SCENE_BASE, strict=True):
        qpos[m.jnt_qposadr[m.joint(name).id]] = value
    StateCapture.restore(env=env, state=state.model_copy(update={"qpos": qpos, "qvel": qvel}))


def _kinder(*, env):
    kinder = env.controllers()._kinder  # noqa: SLF001
    kinder.sync()
    return kinder


def _act(*, env, skill, cube):
    action = np.array([float(SKILL_IDS[skill]), float(cube.split("_")[1]), 0.0, 0.0])
    env.take_action(action=action)
    return env.get_current_state().abstract_atoms


def _position(*, env, cube):
    return np.asarray(env.backend().core().get_object(cube).get_pose()[0], dtype=float)


def _board(*, env):
    return float(_kinder(env=env).shelf.board[1])


def _assert_resting_on_the_board(*, env, cube):
    z = float(_position(env=env, cube=cube)[2])
    assert abs(z - _board(env=env) - O.cube_half_size) < 0.004, (cube, z)


def _region(*, env):
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
        SHELF_DROP_X_MARGINS,
        SHELF_DROP_Y_MARGIN,
    )

    x0, y0, x1, y1 = _kinder(env=env).shelf_region()
    near, far = SHELF_DROP_X_MARGINS
    return x0 + near, y0 + SHELF_DROP_Y_MARGIN, x1 - far, y1 - SHELF_DROP_Y_MARGIN


def _move_to_floor(*, env, cube, x, y):
    Setups.teleport(env=env, name=cube, position=[x, y, O.cube_half_size], quat=[1, 0, 0, 0])
    Setups.settle(env=env)
    env.current_state = env._observed_state(seed=0, steps_taken=0)  # noqa: SLF001


def test_the_case_reproduces_the_new_scene_layout(*, env, states):
    _restore_case(env=env, states=states)
    assert np.allclose(_position(env=env, cube="cube_2")[:2], (1.4997, -0.0031), atol=1e-3)
    atoms = env.get_current_state().abstract_atoms
    assert ("OnFloor", ("cube_4",)) in atoms
    assert all(("OnShelf", (c,)) in atoms for c in CUBES[:4])


def test_a_cube_returned_from_the_floor_lands_on_the_board_not_on_cube_2(*, env, states):
    _restore_case(env=env, states=states)
    atoms = _act(env=env, skill="ReturnCubeToShelfFromFloor", cube="cube_4")
    assert ("OnShelf", ("cube_4",)) in atoms, env.last_skill_error()
    for cube in CUBES:
        _assert_resting_on_the_board(env=env, cube=cube)
    assert all(("OnShelf", (c,)) in atoms for c in CUBES)


def test_the_drop_points_are_free_spots_inside_the_drop_region(*, env, states):
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import (
        SHELF_DROP_CLEARANCE,
        SHELF_DROP_TARGET_SPACING,
        SHELF_DROP_TARGETS,
    )

    _restore_case(env=env, states=states)
    kinder = _kinder(env=env)
    points = kinder.shelf_drop_points()
    assert points == kinder.shelf_drop_points()  # deterministic
    assert 1 <= len(points) <= SHELF_DROP_TARGETS
    x0, y0, x1, y1 = _region(env=env)
    shelf = [_position(env=env, cube=c)[:2] for c in CUBES[:4]]
    for x, y in points:
        assert x0 - 1e-9 <= x <= x1 + 1e-9
        assert y0 - 1e-9 <= y <= y1 + 1e-9
        for c in shelf:
            assert np.hypot(x - c[0], y - c[1]) >= SHELF_DROP_CLEARANCE - 1e-9
    for i, (ax, ay) in enumerate(points):
        for bx, by in points[i + 1 :]:
            assert np.hypot(ax - bx, ay - by) >= SHELF_DROP_TARGET_SPACING - 1e-9
    # The first is as far from every shelf cube as the drop region's grid allows.
    grid = [(x, y) for x in kinder.grid(low=x0, high=x1) for y in kinder.grid(low=y0, high=y1)]

    def free(*, point):
        return min(np.hypot(point[0] - c[0], point[1] - c[1]) for c in shelf)

    assert free(point=points[0]) >= max(free(point=p) for p in grid) - 1e-9


def test_an_empty_shelf_takes_its_first_cube_at_the_drop_regions_centre(*, env, states):
    _restore_case(env=env, states=states)
    for i, cube in enumerate(CUBES[:4]):
        _move_to_floor(env=env, cube=cube, x=0.3, y=-0.6 + 0.3 * i)
    x0, y0, x1, y1 = _region(env=env)
    first = _kinder(env=env).shelf_drop_points()[0]
    assert np.allclose(first, ((x0 + x1) / 2, (y0 + y1) / 2), atol=1e-9)


def test_a_full_shelf_offers_no_drop_point(*, env, states):
    """No spot keeps the clearance: the return refuses rather than stacking."""
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import SHELF_DROP_CLEARANCE

    _restore_case(env=env, states=states)
    kinder = _kinder(env=env)
    x0, y0, x1, y1 = _region(env=env)
    step = SHELF_DROP_CLEARANCE * 0.9
    spots = [(x, y) for x in np.arange(x0, x1 + 1e-9, step) for y in np.arange(y0, y1 + 1e-9, step)]
    original = kinder.shelf_cube_xys
    kinder.shelf_cube_xys = lambda: [np.asarray(s) for s in spots]
    try:
        assert kinder.shelf_drop_points() == []
    finally:
        kinder.shelf_cube_xys = original


def test_cubes_returned_one_after_another_do_not_stack(*, env, states):
    """Main's own floor-out-right-mid, where the first return lands clear, then two more
    cubes returned in turn from the same spot on the floor. At the fixed place point,
    the second landed on the first."""
    StateCapture.restore(env=env, state=states["floor-out-right-mid"])
    floor = NEW_SCENE_CUBES["cube_4"][:2]
    for cube in ("cube_4", "cube_3", "cube_0"):
        if cube != "cube_4":
            _move_to_floor(env=env, cube=cube, x=floor[0], y=floor[1])
        atoms = _act(env=env, skill="ReturnCubeToShelfFromFloor", cube=cube)
        assert ("OnShelf", (cube,)) in atoms, (cube, env.last_skill_error())
    for cube in CUBES:
        _assert_resting_on_the_board(env=env, cube=cube)
    xys = [_position(env=env, cube=c)[:2] for c in CUBES]
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(xys) for b in xys[i + 1 :]]
    assert min(gaps) > 2 * np.sqrt(2) * O.cube_half_size, gaps
