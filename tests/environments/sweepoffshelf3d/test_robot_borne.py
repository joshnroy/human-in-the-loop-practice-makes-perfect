"""Objects the robot itself carries -- held by the pads, or resting on its chassis --
are not obstacles to its own base planner, and a pick that closes on two cubes is a
miss. The states are the 2026-10-21 refusal diagnosis's stuck states from launch
ea9b45b15 (`data/refusal_states.jsonl`): case 2 (ees-always-ask/ees/1) left cube_1 on
the robot's chassis, and case 3 (cccc-determinized-always-ask/pomdp/3) closed one shelf
pick on cube_4 and cube_1 together. Both were refused at 0 control steps ever after."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import KinderSkills
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateCapture, StateSet

STATES = Path(__file__).resolve().parent / "data/refusal_states.jsonl"

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_robot_borne"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    # The states were captured in the lab2 scene; a state restores only there.
    env = SweepOffShelf3DEnvironment(scene_bg=True)
    try:
        yield env
    finally:
        env.close()


@pytest.fixture(name="states", scope="module")
def _states():
    return {state.state_id: state for state in StateSet.load(path=STATES)}


def _restore(*, env, states, state_id):
    StateCapture.restore(env=env, state=states[state_id])
    kinder = env.controllers()._kinder  # noqa: SLF001
    kinder.sync()
    return kinder


def _act(*, env, skill, cube):
    action = np.array([float(SKILL_IDS[skill]), float(cube.split("_")[1]), 0.0, 0.0])
    env.take_action(action=action)
    return env.get_current_state().abstract_atoms


def _backed_off(*, kinder, by=0.3):
    """A base pose straight behind the current one: open floor in every state here."""
    base = kinder.shelf.data.qpos[kinder.shelf.ba].copy()
    base[:2] -= by * np.array([np.cos(base[2]), np.sin(base[2])])
    return base


def test_a_cube_resting_on_the_robot_does_not_block_base_planning(*, env, states):
    kinder = _restore(env=env, states=states, state_id="refusal-case2-eesaa1-cube-on-robot")
    assert "cube_1" in kinder.robot_borne()
    assert kinder.base_drivable(base=_backed_off(kinder=kinder))


def test_a_cube_resting_on_the_robot_has_floor_and_shelf_pick_candidates(*, env, states):
    """The diagnosis counted 0 candidates for both before, 161 and 246 with cube_1 left
    out of the base planner's obstacles."""
    from hitl_pmp.environments.sweepoffshelf3d.floor_pick import FloorPicks

    kinder = _restore(env=env, states=states, state_id="refusal-case2-eesaa1-cube-on-robot")
    assert next(FloorPicks(kinder=kinder).candidates(cube="cube_1"), None) is not None
    centre = kinder.shelf.cube_position(cube="cube_0")
    assert next(kinder.shelf_candidates(cube="cube_0", centre=centre, tried=[]), None) is not None


@pytest.mark.parametrize(
    ("skill", "cube"),
    [("PickCubeFromFloor", "cube_1"), ("PickCubeFromShelf", "cube_0")],
)
def test_with_a_cube_resting_on_the_robot_both_refused_picks_now_pick(*, env, states, skill, cube):
    """The run refused these 47,486 and 47,758 times, at 0 control steps each."""
    _restore(env=env, states=states, state_id="refusal-case2-eesaa1-cube-on-robot")
    atoms = _act(env=env, skill=skill, cube=cube)
    assert ("HoldingCube", ("robot", cube)) in atoms, env.last_skill_error()


def test_two_held_cubes_are_both_tracked_and_neither_blocks_the_base(*, env, states):
    kinder = _restore(env=env, states=states, state_id="refusal-case3-pomdp3-two-cubes-held")
    assert kinder.held_cubes == frozenset({"cube_1", "cube_4"})
    assert {"cube_1", "cube_4"} <= set(kinder.robot_borne())
    assert kinder.base_drivable(base=_backed_off(kinder=kinder))


@pytest.mark.parametrize("cube", ["cube_1", "cube_4"])
def test_either_of_two_held_cubes_is_placed_in_the_box(*, env, states, cube):
    """Refused 259 times as "no placement planned" (cube_1) and 9 as "not holding
    cube_4"; the diagnosis's live counterfactual put both in the box in 742 steps."""
    _restore(env=env, states=states, state_id="refusal-case3-pomdp3-two-cubes-held")
    atoms = _act(env=env, skill="PlaceCubeInBox", cube=cube)
    assert ("InBox", (cube, "box_0")) in atoms, env.last_skill_error()
    assert env.last_controller_steps() > 0


def test_a_pick_that_closes_on_two_cubes_is_a_miss(*, env, states, monkeypatch):
    """From case 3's start, the run's own PickCubeFromShelf(cube_4) missed once, then
    closed on cube_4 and cube_1 together and reported success. Now the pick opens and
    tries again, and never ends with a second cube in the hand."""
    # The neighbour refusals (pad sweep, hand touching a cube) now prevent this double
    # grasp; both off, so this still covers the miss handling behind them.
    monkeypatch.setattr(KinderSkills, "neighbour_in_sweep", lambda self, *, path, cube: None)
    monkeypatch.setattr(KinderSkills, "touches_other_cube", lambda self, *, base, arm, cube: False)
    # The resting wiper as a planning obstacle reroutes this approach off the double
    # grasp; off too, since this tests the miss handling, not the planning scene.
    monkeypatch.setattr(KinderSkills, "add_wiper_obstacle", lambda self, *, sim: None)
    kinder = _restore(
        env=env, states=states, state_id="refusal-case3-pomdp3-start-before-double-grasp"
    )
    atoms = _act(env=env, skill="PickCubeFromShelf", cube="cube_4")
    held = {a[1][1] for a in atoms if a[0] == "HoldingCube"}
    assert held <= {"cube_4"}, held
    assert kinder.held_cubes <= {"cube_4"}
    reasons = [n.reason for n in env.controllers().notes()]
    assert any("cube_1" in r and "two cubes" in r for r in reasons), reasons


def test_the_neighbour_sweep_refusal_now_prevents_that_double_grasp(*, env, states):
    """The same start with the neighbour-sweep refusal on: every centred grasp on
    cube_4 would sweep cube_1, and none is offered. An off-centre one clears it
    (`shelf_offsets`): cube_4 is taken alone, cube_1 left on the shelf where it was."""
    kinder = _restore(
        env=env, states=states, state_id="refusal-case3-pomdp3-start-before-double-grasp"
    )
    before = kinder.shelf.cube_position(cube="cube_1").copy()
    atoms = _act(env=env, skill="PickCubeFromShelf", cube="cube_4")
    assert {a[1][1] for a in atoms if a[0] == "HoldingCube"} == {"cube_4"}, env.last_skill_error()
    assert kinder.held_cubes == {"cube_4"}
    moved = float(np.linalg.norm(kinder.shelf.cube_position(cube="cube_1") - before))
    assert moved < 0.005, moved
