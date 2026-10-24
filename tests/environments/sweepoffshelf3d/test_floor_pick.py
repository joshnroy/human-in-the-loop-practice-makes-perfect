"""The floor and box picks (`floor_pick.py`) on states the skill benchmark restores:
cubes beside the box, beside the cupboard's side panel and inside the box, where
upstream's four vertical grasps all collide."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.backend import WIDE_BOX_TASK_ID
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import HOME_ARM, RETRACT_TOLERANCE
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet

STATES = Path(__file__).resolve().parent / "data/wide_box_benchmark_states.jsonl"

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_floor_pick"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene with the 50 cm box; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True, task_id=WIDE_BOX_TASK_ID)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _restore(*, env, states, state_id):
    StateCapture.restore(env=env, state=states[state_id])


def _act(*, env, skill, cube):
    action = np.array([float(SKILL_IDS[skill]), float(cube.split("_")[1]), 0.0, 0.0])
    env.take_action(action=action)
    return env.get_current_state().abstract_atoms


def _arm_from_home(*, env):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    joints = [model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
    arm = data.qpos[model.jnt_qposadr[joints]]
    return float(np.max(np.abs((arm - HOME_ARM + np.pi) % (2 * np.pi) - np.pi)))


@pytest.mark.parametrize(
    ("state_id", "cube"),
    [
        ("floor-out-near-centre", "cube_0"),  # 2 cm in front of the box's near wall
        ("floor-out-left-mid", "cube_3"),  # 2 cm beside the box's side wall
        ("floor-panel-left-front", "cube_3"),  # 2 cm beside the cupboard's side panel
        ("floor-panel-right-mid", "cube_1"),
    ],
)
def test_a_cube_on_the_floor_beside_a_wall_is_picked_up(*, env, states, state_id, cube):
    _restore(env=env, states=states, state_id=state_id)
    atoms = _act(env=env, skill="PickCubeFromFloor", cube=cube)
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_cube_beside_the_box_is_returned_to_the_shelf(*, env, states):
    _restore(env=env, states=states, state_id="floor-out-right-near")
    atoms = _act(env=env, skill="ReturnCubeToShelfFromFloor", cube="cube_1")
    assert ("OnShelf", ("cube_1",)) in atoms, env.last_skill_error()
    assert ("HandEmpty", ("robot",)) in atoms


@pytest.mark.parametrize(
    ("state_id", "cube"),
    [
        # A real practice state: 4-6 cm from the box's far wall.
        ("harvest-ReturnCubeToShelfFromBox-18", "cube_1"),
        # The same cube, with cube_0 on the shelf where a fixed place point let go: it
        # was dropped onto cube_0 until the drop point moved to a free spot.
        ("harvest-ReturnCubeToShelfFromBox-17", "cube_1"),
        # The one box state upstream's pick already returned.
        ("container-in-mid-centre", "cube_4"),
    ],
)
def test_a_cube_in_the_box_is_returned_to_the_shelf(*, env, states, state_id, cube):
    _restore(env=env, states=states, state_id=state_id)
    atoms = _act(env=env, skill="ReturnCubeToShelfFromBox", cube=cube)
    assert ("OnShelf", (cube,)) in atoms, env.last_skill_error()
    assert ("HandEmpty", ("robot",)) in atoms


def test_a_cube_face_to_face_with_another_is_not_taken_with_it(*, env, states):
    # A real practice state: cube_2 rests against cube_1's side in the box. Pads closing
    # on cube_1 also close on cube_2's edge, and carried both out together. No centred
    # grasp keeps upstream's clearance from cube_2 (once refused outright); a grasp
    # shifted away from cube_2 does, and returns cube_1 alone.
    _restore(env=env, states=states, state_id="harvest-ReturnCubeToShelfFromBox-19")
    atoms = _act(env=env, skill="ReturnCubeToShelfFromBox", cube="cube_1")
    assert ("InBox", ("cube_2", "box_0")) in atoms
    assert ("OnShelf", ("cube_1",)) in atoms, env.last_skill_error()


def test_a_cube_wedged_in_a_box_corner_is_refused_without_moving_it(*, env, states):
    # 4 mm from two walls: no pad fits between the cube and either wall.
    _restore(env=env, states=states, state_id="container-in-near-right")
    before = env.backend().core().get_object("cube_0").get_pose()[0].copy()
    atoms = _act(env=env, skill="ReturnCubeToShelfFromBox", cube="cube_0")
    assert env.last_skill_error() is not None
    assert ("InBox", ("cube_0", "box_0")) in atoms
    after = env.backend().core().get_object("cube_0").get_pose()[0]
    assert np.linalg.norm(after - before) < 0.005
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


@pytest.mark.parametrize(
    ("state_id", "cube"),
    [("harvest-ReturnCubeToShelfFromBox-17", "cube_1"), ("floor-out-left-near", "cube_0")],
)
def test_every_candidate_keeps_upstreams_clearance_from_the_box_and_cupboard(
    *, env, states, state_id, cube
):
    """Measured independently of the search's own check: at the grasp, the pre-grasp and
    the end of the lift, no arm or gripper geom comes within upstream's
    `PickCubeController.GRASP_OBSTACLE_CLEARANCE` of the box, the cupboard or another cube."""
    from kinder_models.dynamic3d.tossing.parameterized_skills import PickCubeController

    from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks

    _restore(env=env, states=states, state_id=state_id)
    kinder = env.controllers()._kinder  # noqa: SLF001
    shelf = kinder.shelf
    m, d = shelf.model, mujoco.MjData(shelf.model)

    def collides(*, g):
        return bool(m.geom_contype[g] or m.geom_conaffinity[g])

    others = {
        g
        for g in range(m.ngeom)
        if str(m.body(m.geom_bodyid[g]).name).startswith("cube_")
        and m.body(m.geom_bodyid[g]).name != cube
    }
    fixed = [g for g in sorted(shelf.cupboard_geoms | shelf.box_geoms | others) if collides(g=g)]
    arm = [
        g
        for g in sorted(shelf.robot_geoms)
        if collides(g=g) and m.body(m.geom_bodyid[g]).name != "robot_base_link"
    ]
    clearance = PickCubeController.GRASP_OBSTACLE_CLEARANCE
    seen = 0
    for grasp in FloorPicks(kinder=kinder).candidates(cube=cube):
        for conf in (grasp.grasp_conf, grasp.pre_conf, grasp.lift_conf):
            d.qpos[:] = shelf.data.qpos
            d.qpos[shelf.ba] = grasp.base
            d.qpos[shelf.qa] = conf
            mujoco.mj_kinematics(m, d)
            nearest = min(
                mujoco.mj_geomDistance(m, d, g, f, clearance, None) for g in arm for f in fixed
            )
            assert nearest >= clearance - 1e-4
        seen += 1
        if seen >= 5:
            break
    assert seen > 0
