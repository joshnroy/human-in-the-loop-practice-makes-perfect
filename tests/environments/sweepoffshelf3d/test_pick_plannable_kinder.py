"""`PickPlannable` on the live scene: the environment's dry run of the floor and box pick
agrees with the dispatched skill, and gates the operators that start with that pick."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.objects import SweepOffShelf3DObjects as O
from hitl_pmp.environments.sweepoffshelf3d.predicates import PICK_PLANNABLE
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_provider import SweepOffShelf3DSkillProvider
from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.planning.grounding import SkillGrounder

STATES = Path(__file__).resolve().parents[3] / "benchmarks/sweepoffshelf3d/states.jsonl"
GATED = (
    S.PICK_CUBE_FROM_FLOOR,
    S.RETURN_CUBE_TO_SHELF_FROM_FLOOR,
    S.RETURN_CUBE_TO_SHELF_FROM_BOX,
)

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_pick_plannable_kinder"),
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


def _cube(*, name):
    return next(c for c in O.cubes if c.name == name)


def _atoms(*, env):
    provider = SweepOffShelf3DSkillProvider.model_construct(env=env)
    return SkillGrounder.abstract_state(
        state=env.get_current_state(), objects=O.all_objects(), predicates=provider.predicates()
    )


def _gated_for(*, env, cube):
    return [
        g.skill.name
        for g in SkillGrounder.applicable_ground_skills(
            skills=GATED, objects=O.all_objects(), true_atoms=_atoms(env=env)
        )
        if g.objects[1] == cube
    ]


def test_a_wedged_cube_the_pick_refuses_is_not_plannable_and_cannot_be_picked(*, env, states):
    # 1 cm from the narrow box's outer wall, 5 mm from the cupboard's bottom board:
    # refused at dispatch ("no grasp planned", 0 control steps) by the grid-36 pick.
    state = states["floor-pocket-right-w10-b5"]
    StateCapture.restore(env=env, state=state)
    cube = _cube(name=state.cases[0].cube)
    atoms = _atoms(env=env)
    assert GroundAtom(predicate=PICK_PLANNABLE, objects=(O.robot, cube)) not in atoms
    assert _gated_for(env=env, cube=cube) == []
    env.take_action(
        action=np.array([float(SKILL_IDS["PickCubeFromFloor"]), O.cubes.index(cube), 0, 0])
    )
    assert env.last_controller_steps() == 0, env.last_skill_error()
    assert ("OnFloor", (cube.name,)) in env.get_current_state().abstract_atoms


def test_a_cube_on_open_floor_is_plannable(*, env, states):
    state = states["floor-out-near-centre"]
    StateCapture.restore(env=env, state=state)
    cube = _cube(name=state.cases[0].cube)
    assert GroundAtom(predicate=PICK_PLANNABLE, objects=(O.robot, cube)) in _atoms(env=env)
    assert _gated_for(env=env, cube=cube) == ["PickCubeFromFloor", "ReturnCubeToShelfFromFloor"]


def test_after_reset_cubes_the_wedged_cube_is_back_on_the_shelf(*, env, states):
    state = states["floor-pocket-right-w10-b5"]
    StateCapture.restore(env=env, state=state)
    cube = _cube(name=state.cases[0].cube)
    env.reset_movables(destination="reset_cubes")
    atoms = env.get_current_state().abstract_atoms
    assert ("OnShelf", (cube.name,)) in atoms
    assert ("PickPlannable", (O.robot.name, cube.name)) in atoms


def test_the_dry_run_leaves_no_note_in_the_skill_trace(*, env, states):
    StateCapture.restore(env=env, state=states["floor-pocket-right-w10-b5"])
    assert env.controllers().notes() == []
