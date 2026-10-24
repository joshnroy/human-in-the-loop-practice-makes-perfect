"""Where a sweep that misses a narrow box drops its cubes: on the floor in the pocket
between the box's outer side wall and the cupboard's bottom board, under the cube
shelf's front edge. In the smallbox PoC (2026-10-23) the floor pick refused 16/16 such
cubes, and a miss ended the task.

The benchmark's pocket family (`Placements.pockets`) slides the default box sideways to
make the same pocket. These two states are refused on main 8ff23ce2c ("no grasp
planned", 0 control steps): the open jaw's outer pad and linkage, the wrist under the
shelf's edge, and upstream's 1 cm reserve left no grasp."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import HOME_ARM, RETRACT_TOLERANCE
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet

STATES = Path(__file__).resolve().parents[3] / "benchmarks/sweepoffshelf3d/states.jsonl"

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_floor_pick_pockets"),
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


def _arm_from_home(*, env):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    joints = [model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
    arm = data.qpos[model.jnt_qposadr[joints]]
    return float(np.max(np.abs((arm - HOME_ARM + np.pi) % (2 * np.pi) - np.pi)))


@pytest.mark.parametrize(
    "state_id",
    [
        "floor-pocket-right-w40-b5",  # 4 cm from the box's wall, 5 mm from the board
        "floor-pocket-right-w40-b25",  # 4 cm from the wall, 2.5 cm from the board
    ],
)
def test_a_cube_in_the_pocket_beside_the_box_is_picked_up(*, env, states, state_id):
    state = states[state_id]
    StateCapture.restore(env=env, state=state)
    cube = state.cases[0].cube
    assert cube is not None
    action = np.array([float(SKILL_IDS["PickCubeFromFloor"]), float(cube.split("_")[1]), 0, 0])
    env.take_action(action=action)
    atoms = env.get_current_state().abstract_atoms
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE
