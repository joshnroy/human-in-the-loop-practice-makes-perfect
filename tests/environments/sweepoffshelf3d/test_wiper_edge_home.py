"""The wiper's home on top of the cupboard, its handle over the open front edge, and the
one grasp it is picked, swept with and put back by (Josh, 2026-10-21)."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from tests.environments.sweepoffshelf3d.test_kinder_skills import _force_one_reseat

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_wiper_edge_home"),
]

# The cupboard's top board: the board above the third shelf.
TOP_BOARD = 3
HOME = ("WiperHome", ("wiper_0",))
HELD = ("HoldingWiper", ("robot", "wiper_0"))
EMPTY = ("HandEmpty", ("robot",))


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=False)
    try:
        yield env
    finally:
        env.close()


def _wiper(*, env):
    """(body position, centre-of-mass x, handle tip x, tilt in degrees) as it lies now."""
    backend = env.backend()
    model, data = backend.model(), backend.data()
    body = model.body("wiper_0").id
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
    handle = max(geoms, key=lambda g: float(model.geom_size[g][0]))
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    rotation = data.geom_xmat[handle].reshape(3, 3)
    corners = (signs * model.geom_size[handle]) @ rotation.T + data.geom_xpos[handle]
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[body].reshape(3, 3)[2, 2], -1, 1))))
    return (
        data.xpos[body].copy(),
        float(data.subtree_com[body][0]),
        float(corners[:, 0].min()),
        tilt,
    )


def _place(*, env, position, yaw_deg=180.0, tilt_deg=0.0, settle=0):
    from scipy.spatial.transform import Rotation

    quat = Rotation.from_euler("zy", [yaw_deg, tilt_deg], degrees=True).as_quat()
    backend = env.backend()
    obj = backend.core().get_object("wiper_0")
    obj.set_pose(list(position), [quat[3], quat[0], quat[1], quat[2]])
    obj.set_velocity([0.0] * 3, [0.0] * 3)
    backend.sim().forward()
    for _ in range(settle):
        backend.sim().step()


def _act(*, env, skill, height=0.005, speed=0.05):
    env.take_action(action=np.array([float(SKILL_IDS[skill]), 0.0, height, speed]))
    return env.get_current_state().abstract_atoms


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_wiper_starts_home_on_top_of_the_cupboard_with_its_handle_over_the_front(*, env, seed):
    env.reset_to_seed(seed=seed)
    core = env.backend().core()
    position, com_x, tip_x, tilt = _wiper(env=env)
    front = core.shelf_front_x()
    assert abs(position[2] - core.shelf_top_z(shelf=TOP_BOARD)) < 0.003
    assert tilt < 5.0
    assert com_x > front + 0.005
    assert tip_x < front - 0.05
    assert HOME in env.backend().abstract_atoms()


def test_the_wiper_in_its_old_slot_on_the_third_shelf_is_not_home(*, env):
    env.reset_to_seed(seed=0)
    board = env.backend().core().shelf_top_z(shelf=2)
    _place(env=env, position=[1.512, 0.038, board + 0.001], yaw_deg=90.0, settle=300)
    atoms = env.backend().abstract_atoms()
    assert HOME not in atoms
    assert ("WiperOnFloor", ("wiper_0",)) in atoms


def test_the_wiper_lying_wholly_on_the_top_board_is_not_home(*, env):
    """Its handle no longer overhangs the front: the hand could not reach the grasp from
    outside the cupboard."""
    env.reset_to_seed(seed=0)
    position, *_ = _wiper(env=env)
    _place(env=env, position=[position[0] + 0.15, position[1], position[2]], settle=300)
    _, _, tip_x, _ = _wiper(env=env)
    assert tip_x > env.backend().core().shelf_front_x()
    assert HOME not in env.backend().abstract_atoms()


def test_the_wiper_tilted_at_its_home_is_not_home(*, env):
    env.reset_to_seed(seed=0)
    position, *_ = _wiper(env=env)
    _place(env=env, position=[position[0], position[1], position[2] + 0.02], tilt_deg=10.0)
    assert HOME not in env.backend().abstract_atoms()


def test_reset_wiper_teleports_a_fallen_wiper_to_its_edge_home(*, env):
    env.reset_to_seed(seed=0)
    start, *_ = _wiper(env=env)
    _place(env=env, position=[0.6, 0.6, 0.05], settle=500)
    assert HOME not in env.backend().abstract_atoms()
    env.backend().reset_wiper()
    for _ in range(500):
        env.backend().sim().step()
    position, *_ = _wiper(env=env)
    assert float(np.linalg.norm(position - start)) < 0.002
    assert HOME in env.backend().abstract_atoms()


def test_one_grasp_picks_sweeps_and_returns_the_wiper(*, env, monkeypatch):
    """PickWiper, SweepShelf and ReturnWiper all hold the wiper at the one sweep grasp:
    every grasp taken is it, and nothing is ever set down on the floor to regrasp. One
    reseat is forced, since seed 0's own strokes never turn the wiper far enough to
    reseat and a check over no grasps at all would pass vacuously."""
    env.reset_to_seed(seed=0)
    shelf = env.controllers()._skills  # noqa: SLF001
    atoms = _act(env=env, skill="PickWiper")
    assert HELD in atoms, env.last_skill_error()
    _force_one_reseat(monkeypatch=monkeypatch)
    atoms = _act(env=env, skill="SweepShelf")
    assert ("AnyCubeOffShelf", ()) in atoms, env.last_skill_error()
    assert HELD in atoms
    atoms = _act(env=env, skill="ReturnWiper")
    assert {HOME, EMPTY} <= atoms, env.last_skill_error()
    shelf = env.controllers()._skills  # noqa: SLF001
    home = float(np.linalg.norm(shelf.wiper_pose()[:3, 3] - shelf.wiper_start[:3, 3]))
    assert home < 0.01
    # The pick closes through KINDER's controller; any grasp the scripted skills take
    # after it is a reseat of the same grasp on the cube shelf mid-sweep.
    assert len(shelf.grasps_taken) >= 1, shelf.notes
    assert all(grasp["label"].endswith("_reseat") for grasp in shelf.grasps_taken)
    assert all("kind" not in grasp for grasp in shelf.grasps_taken)


def test_reset_wiper_while_holding_it_takes_it_out_of_the_hand_to_its_home(*, env):
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickWiper")
    assert HELD in atoms, env.last_skill_error()
    env.backend().reset_wiper()
    env.controllers().on_objects_moved()
    for _ in range(300):
        env.backend().sim().step()
    atoms = env.backend().abstract_atoms()
    assert {HOME, EMPTY} <= atoms
    assert env.controllers()._skills.in_hand is None  # noqa: SLF001


def test_no_two_grasp_machinery_is_left():
    from hitl_pmp.environments.sweepoffshelf3d import scripted_skills
    from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills
    from hitl_pmp.environments.sweepoffshelf3d.scripted_skills import ShelfSkills

    for name in (
        "leaned_head_grasp",
        "regrasp_for_sweep",
        "set_wiper_down_on_floor",
        "take_wiper_from_floor",
    ):
        assert not hasattr(ShelfSkills, name), name
    assert not hasattr(KinderSkills, "nearest_wiper_grasp")
    for name in ("SHELF_HEAD_OFFSET", "FLOOR_HEAD_XY"):
        assert not hasattr(scripted_skills, name), name


def test_shelf_skills_hold_one_fixed_wiper_grasp_and_no_grasp_kind(*, env):
    env.reset_to_seed(seed=0)
    shelf = env.controllers()._skills  # noqa: SLF001
    assert not hasattr(shelf, "grasp_kind")
    assert np.asarray(shelf.grasp_in_wiper).shape == (4, 4)
    assert np.allclose(shelf.wiper_grasp(), shelf.wiper_pose() @ shelf.grasp_in_wiper)
