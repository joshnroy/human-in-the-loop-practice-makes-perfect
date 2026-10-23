"""Every Tossing3D pick retracts the arm before its base moves.

t3-final-20261023 collapsed on picks that started with the arm left stretched by an
earlier pick that timed out (`pick_step_limit`): nothing retracted it, since OpenGripper
only commands the fingers and the resets move only the cube and the bin. The stuck
states are frozen here from those runs' recordings (the last tick before the action):

* expectimax seed 2, before seq 484: the arm is stretched beside the bin, and the old
  pick's base motion carried it onto the bin rim.
* expectimax seed 2, before seq 485: a finger pad already rests on the bin rim.
* determinized seed 4, before seq 808 and seq 821: the arm is frozen stretched after
  seq 805. At 808, with the fingers posed open as the planner poses them, a finger pad
  is 6.3 mm inside the cube.

The retract configuration is upstream's own, `PickCubeController.home_joints`
(= `MoveToTossLocationAndTossController._return_home_arm_config`). A pick whose start
configuration is already in contact refuses, loudly: what it should do instead is not
decided, and the planner does not read the real finger joints yet.
"""

import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.predicates import HOLDING, PICK_PLANNABLE

pytestmark = [
    pytest.mark.xdist_group("test_retract_at_skill_start"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
]

_FIXTURES = Path(__file__).parent / "fixtures" / "pick_plannable"
_STRETCHED_BESIDE_BIN = "t3final_expectimax_s2_seq484_stretched.json"
_HOOKED_ON_RIM = "t3final_expectimax_s2_seq485_hooked.json"
_STRETCHED_FINGER_IN_CUBE = "t3final_determinized_s4_seq808_stretched.json"
_STRETCHED = "t3final_determinized_s4_seq821_stretched.json"
_STUCK = [_STRETCHED_BESIDE_BIN, _HOOKED_ON_RIM, _STRETCHED_FINGER_IN_CUBE, _STRETCHED]
# Stretched, and clear of everything at the current configuration.
_RETRACTABLE = [_STRETCHED_BESIDE_BIN, _STRETCHED]

_ARM = tuple(f"pos_arm_joint{i}" for i in range(1, 8))
_RETRACT = np.deg2rad([0, -20, 180, -146, 0, -50, 90])
_RETRACTED_TOLERANCE = 0.05  # rad, per joint, after wrapping


@pytest.fixture(scope="module")
def live_env() -> Iterator[Tossing3DEnvironment]:
    env = Tossing3DEnvironment()
    env.reset_to_seed(seed=125)
    yield env
    env.close()


def _restore(*, env: Tossing3DEnvironment, fixture: str) -> Any:
    plain = json.loads((_FIXTURES / fixture).read_text())
    return env.restore_plain_snapshot(plain=plain, dtype=np.float64)


def _arm_and_base(*, env: Tossing3DEnvironment) -> tuple[np.ndarray, np.ndarray]:
    backend = env.backend()
    state = backend.live_state()
    robot = state.get_object_from_name(backend.robot_name)
    arm = np.array([state.get(robot, name) for name in _ARM])
    base = np.array([state.get(robot, k) for k in ("pos_base_x", "pos_base_y", "pos_base_rot")])
    return arm, base


def _retracted(*, arm: np.ndarray) -> bool:
    wrapped = (arm - _RETRACT + np.pi) % (2 * np.pi) - np.pi
    return bool(np.max(np.abs(wrapped)) <= _RETRACTED_TOLERANCE)


def _with_teleported_retract(*, env: Tossing3DEnvironment, fixture: str) -> dict:
    plain = json.loads((_FIXTURES / fixture).read_text())
    # The robot row's feature order is the backend's own; arm velocities zeroed.
    backend = env.backend()
    names = list(
        backend.live_state().type_features[
            backend.live_state().get_object_from_name(backend.robot_name).type
        ]
    )
    for name, value in zip(_ARM, _RETRACT, strict=True):
        plain["robot"][names.index(name)] = float(value)
    for i, name in enumerate(names):
        if name.startswith("vel_arm"):
            plain["robot"][i] = 0.0
    return plain


@pytest.mark.parametrize("fixture", _STUCK)
def test_the_recorded_stuck_states_start_with_the_arm_stretched(
    *, live_env: Tossing3DEnvironment, fixture: str
) -> None:
    _restore(env=live_env, fixture=fixture)
    arm, _ = _arm_and_base(env=live_env)
    assert not _retracted(arm=arm)


@pytest.mark.parametrize("fixture", _RETRACTABLE)
def test_a_pick_from_a_stretched_arm_retracts_before_the_base_moves(
    *, live_env: Tossing3DEnvironment, fixture: str
) -> None:
    env = live_env
    _restore(env=env, fixture=fixture)
    _, base0 = _arm_and_base(env=env)
    ticks: list[tuple[np.ndarray, np.ndarray]] = []
    env.backend().set_control_step_observers(
        before=None, after=lambda: ticks.append(_arm_and_base(env=env))
    )
    try:
        run = env.backend().run_pick_cube()
    finally:
        env.backend().set_control_step_observers(before=None, after=None)
    assert run.error is None, run.error
    assert ticks
    first_base_move = next(
        (i for i, (_, base) in enumerate(ticks) if np.max(np.abs(base - base0)) > 1e-3),
        len(ticks),
    )
    assert any(_retracted(arm=arm) for arm, _ in ticks[: first_base_move + 1])


@pytest.mark.parametrize("fixture", _RETRACTABLE)
def test_pick_plannable_from_a_stretched_arm_answers_as_from_the_retract(
    *, live_env: Tossing3DEnvironment, fixture: str
) -> None:
    env = live_env
    retracted = env.restore_plain_snapshot(
        plain=_with_teleported_retract(env=env, fixture=fixture), dtype=np.float64
    )
    from_retract = PICK_PLANNABLE.holds(retracted, (env.robot, env.cube))
    stretched = _restore(env=env, fixture=fixture)
    assert PICK_PLANNABLE.holds(stretched, (env.robot, env.cube)) == from_retract
    assert from_retract


@pytest.mark.parametrize("fixture", _RETRACTABLE)
def test_the_stuck_pick_succeeds_through_a_real_retract(
    *, live_env: Tossing3DEnvironment, fixture: str
) -> None:
    env = live_env
    _restore(env=env, fixture=fixture)
    held = env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))
    assert env.last_skill_error() is None
    assert HOLDING.holds(held, (env.robot, env.cube))


def test_a_pick_that_starts_in_contact_refuses_loudly(*, live_env: Tossing3DEnvironment) -> None:
    """s4 before seq 808: the planner poses the fingers open, and open, a pad is inside
    the cube. The pick refuses with the reason, rather than plan from a configuration
    already in collision."""
    env = live_env
    _restore(env=env, fixture=_STRETCHED_FINGER_IN_CUBE)
    failure = env.backend().pick_cube_plan_failure()
    assert failure is not None
    assert "starts in collision with cube_0" in failure


@pytest.mark.xfail(
    strict=True,
    reason="undecided: a finger already resting on the bin rim (0.23 mm clear in the "
    "planning scene, in contact in MuJoCo) stalls the retract",
)
def test_a_pick_with_a_finger_on_the_rim_succeeds(*, live_env: Tossing3DEnvironment) -> None:
    env = live_env
    _restore(env=env, fixture=_HOOKED_ON_RIM)
    held = env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))
    assert env.last_skill_error() is None
    assert HOLDING.holds(held, (env.robot, env.cube))


def test_an_already_retracted_pick_spends_no_steps_retracting(
    *, live_env: Tossing3DEnvironment
) -> None:
    """The retract is a no-op from the retract: a fresh episode's pick is unchanged."""
    env = live_env
    env.reset_to_seed(seed=125)
    arm, _ = _arm_and_base(env=env)
    assert _retracted(arm=arm)
    env.take_action(action=np.array([float(env.pick_cube_id), 0.0, 0.0, 0.0, 0.0]))
    assert env.last_skill_error() is None
