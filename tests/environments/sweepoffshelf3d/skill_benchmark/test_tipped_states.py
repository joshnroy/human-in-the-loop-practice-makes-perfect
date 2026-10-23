"""The tipped-cube benchmark family: cubes lying on a side face, which no other family
produces, so the benchmark can see a skill that reads a tipped cube's pose wrong."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    StateCapture,
    StateFamilies,
    StateSet,
    TippedCubes,
)

TIPPED_STATES = (
    Path(__file__).resolve().parents[4] / "benchmarks/sweepoffshelf3d/tipped_states.jsonl"
)
AXES = {"x": 0, "-x": 0, "y": 1}

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [pytest.mark.xdist_group("sweepoffshelf3d_skill_benchmark_tipped_states")]
NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None or importlib.util.find_spec("kinder_models") is None,
    reason="needs KINDER",
)


def _up_axis(*, wxyz) -> tuple[int, float]:
    """(index of the cube's own axis nearest vertical, its z component)."""
    matrix = Rotation.from_quat(np.asarray(wxyz)[[1, 2, 3, 0]]).as_matrix()
    up = int(np.argmax(np.abs(matrix[2, :])))
    return up, float(matrix[2, up])


@pytest.mark.parametrize("index", range(12))
def test_a_tipped_cube_has_the_named_axis_vertical(*, index):
    up, yaw = TippedCubes.pose(index=index)
    axis, z = _up_axis(wxyz=TippedCubes.quat(up=up, yaw_deg=yaw))
    assert axis == AXES[up]
    assert abs(z) > 0.9999
    assert (z < 0) == up.startswith("-")


def test_two_of_every_three_tipped_cubes_are_at_gimbal_lock():
    ups = [TippedCubes.pose(index=i)[0] for i in range(9)]
    assert sum(u in {"x", "-x"} for u in ups) == 6


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@NEEDS_KINDER
def test_the_family_covers_box_shelf_and_floor_cases_for_each_tipped_skill(*, env):
    specs = list(StateFamilies.tipped_specs(env=env))
    skills = {case.skill for spec in specs for case in spec.cases}
    assert skills == {
        "ReturnCubeToShelfFromBox",
        "PickCubeFromShelf",
        "PickCubeFromFloor",
        "ReturnCubeToShelfFromFloor",
    }
    assert all(spec.family == "tipped" and spec.state_id.startswith("tipped-") for spec in specs)
    assert len({spec.state_id for spec in specs}) == len(specs)


@NEEDS_KINDER
def test_every_shipped_tipped_state_restores_with_its_cube_still_on_a_side_face(*, env):
    """The cube settles on the face it was put on: still tipped when the case starts."""
    states = StateSet.load(path=TIPPED_STATES)
    assert states
    for state in states:
        StateCapture.restore(env=env, state=state)
        cube = state.cases[0].cube
        assert cube is not None
        data = env.backend().data()
        axis, z = _up_axis(wxyz=np.asarray(data.body(cube).xquat))
        assert axis != 2 and abs(z) > 0.99, (state.state_id, axis, z)
