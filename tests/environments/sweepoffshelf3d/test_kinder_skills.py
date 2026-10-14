"""SweepOffShelf3D's skills on KINDER's controllers, run through `take_action`."""

import importlib.util

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.controllers import ControllerFailure
from hitl_pmp.environments.sweepoffshelf3d.environment import (
    SKILL_IDS,
    SweepOffShelf3DEnvironment,
)
from hitl_pmp.environments.sweepoffshelf3d.kinder_skills import HOME_ARM, RETRACT_TOLERANCE

# Module-scoped fixtures: keep them on one xdist worker (see tests/test_xdist_groups.py).
pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("kinder") is None
        or importlib.util.find_spec("kinder_models") is None,
        reason="needs KINDER",
    ),
    pytest.mark.xdist_group("sweepoffshelf3d_test_kinder_skills"),
]


@pytest.fixture(name="env", scope="module")
def _env():
    env = SweepOffShelf3DEnvironment(scene_bg=False)
    try:
        yield env
    finally:
        env.close()


def _act(*, env, skill, cube=0, height=0.005, speed=0.05):
    env.take_action(action=np.array([float(SKILL_IDS[skill]), float(cube), height, speed]))
    return env.get_current_state().abstract_atoms


def _arm_from_home(*, env):
    backend = env.backend()
    model, data = backend.model(), backend.data()
    joints = [model.joint(f"robot_joint_{i}").id for i in range(1, 8)]
    arm = data.qpos[model.jnt_qposadr[joints]]
    return float(np.max(np.abs((arm - HOME_ARM + np.pi) % (2 * np.pi) - np.pi)))


def _teleport(*, env, name, position):
    obj = env.backend().core().get_object(name)
    obj.set_pose(list(position), [1.0, 0.0, 0.0, 0.0])
    obj.set_velocity([0.0] * 3, [0.0] * 3)
    env.backend().sim().forward()
    for _ in range(20):
        env.backend().env().step(np.zeros(11))


def test_pick_wiper_takes_it_off_its_shelf_and_retracts_the_arm(*, env):
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_cube_on_the_open_floor_is_picked_up(*, env):
    # Behind the robot's start, where the scripted floor pick found no IK solution.
    env.reset_to_seed(seed=0)
    _teleport(env=env, name="cube_1", position=[0.60, -0.55, 0.011])
    assert ("OnFloor", ("cube_1",)) in env.backend().abstract_atoms()
    atoms = _act(env=env, skill="PickCubeFromFloor", cube=1)
    assert ("HoldingCube", ("robot", "cube_1")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_cube_picked_off_the_floor_is_placed_in_the_box(*, env):
    env.reset_to_seed(seed=0)
    _teleport(env=env, name="cube_2", position=[0.95, 0.45, 0.011])
    atoms = _act(env=env, skill="PickCubeFromFloor", cube=2)
    assert ("HoldingCube", ("robot", "cube_2")) in atoms, env.last_skill_error()
    atoms = _act(env=env, skill="PlaceCubeInBox", cube=2)
    assert {("InBox", ("cube_2", "box_0")), ("HandEmpty", ("robot",))} <= atoms, (
        env.last_skill_error()
    )
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


@pytest.mark.parametrize("cube", [0, 1])
def test_pick_cube_from_shelf_takes_a_cube_out_of_the_cupboard(*, env, cube):
    # cube_1 rests on the board so exactly that upstream's retract plan, holding it,
    # started in collision and never planned.
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickCubeFromShelf", cube=cube)
    assert ("HoldingCube", ("robot", f"cube_{cube}")) in atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_wiper_lost_from_the_hand_is_not_still_believed_held(*, env):
    """The absorbing failure behind CCCC's PickWiper tally: a skill that loses the
    wiper used to leave the controllers believing it was in hand, so every later
    hand-empty skill refused without moving."""
    env.reset_to_seed(seed=0)
    atoms = _act(env=env, skill="PickWiper")
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms, env.last_skill_error()
    # The wiper slips out of the hand onto the floor, and nobody puts it back.
    _teleport(env=env, name="wiper_0", position=[0.55, -0.6, 0.02])
    env.current_state = env._observed_state(seed=0, steps_taken=1)  # noqa: SLF001
    assert ("HandEmpty", ("robot",)) in env.get_current_state().abstract_atoms
    _act(env=env, skill="PickCubeFromShelf", cube=1)
    assert "not empty" not in (env.last_skill_error() or "")
    assert env.last_controller_steps() > 0


@pytest.mark.main_only
def test_the_wiper_sweeps_the_shelf_and_goes_back(*, env):
    env.reset_to_seed(seed=0)
    assert ("HoldingWiper", ("robot", "wiper_0")) in _act(env=env, skill="PickWiper")
    atoms = _act(env=env, skill="SweepShelf", height=0.005, speed=0.05)
    # Whether every cube lands in the box is the sweep's competence; the wiper staying
    # in hand through the retract that follows is not.
    assert not any(name == "OnShelf" for name, _ in atoms), env.last_skill_error()
    assert ("HoldingWiper", ("robot", "wiper_0")) in atoms
    atoms = _act(env=env, skill="ReturnWiper")
    assert {("WiperHome", ("wiper_0",)), ("HandEmpty", ("robot",))} <= atoms, env.last_skill_error()
    assert _arm_from_home(env=env) < RETRACT_TOLERANCE


def test_a_cube_returned_from_the_box_rests_on_the_shelf_or_the_skill_says_it_failed(*, env):
    env.reset_to_seed(seed=3)
    box = env.backend().core().get_object("box_0").get_pose()[0]
    _teleport(env=env, name="cube_3", position=[box[0], box[1], 0.06])
    try:
        env.controllers()._run(skill="return_cube_to_shelf", cube="cube_3")  # noqa: SLF001
    except ControllerFailure:
        assert ("OnShelf", ("cube_3",)) not in env.backend().abstract_atoms()
    else:
        assert ("OnShelf", ("cube_3",)) in env.backend().abstract_atoms()
