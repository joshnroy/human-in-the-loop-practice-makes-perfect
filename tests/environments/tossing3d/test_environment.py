"""The parts of `Tossing3DEnvironment` that do not touch MuJoCo: the `State` layout,
the action decoding, and `set_state`'s seed contract. Anything that steps the simulator
lives in `test_kinder_fidelity.py` and is skipped without `kindergarden`."""

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment

from .conftest import GOAL_REGION, build_state

_ENV = Tossing3DEnvironment


class _RecordingBackend:
    """Stands in for `KinderBackend`, recording what `take_action` asked the simulator
    to do. This is the seam the whole domain is designed around -- decoding an action
    is separable from executing it, so it can be tested with no MuJoCo present."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.features = {
            "robot": (0.0, 0.0, 0.0, 0.0),
            "cube": (0.65, -0.1, 0.025),
            "bin": (2.2305, 0.0, 0.0),
            "barrier": (1.3, 0.0, 0.05),
        }

    def execute_pick(self, *, distance: float, rot: float) -> bool:
        self.calls.append(("pick", distance, rot))
        return True

    def execute_move_to_throw_pose(self, *, distance: float) -> bool:
        self.calls.append(("move", distance))
        return True

    def execute_toss(self, *, swing: float) -> bool:
        self.calls.append(("toss", swing))
        return True

    def read_features(self) -> dict[str, tuple[float, ...]]:
        return self.features

    def reset(self, *, seed: int) -> dict[str, tuple[float, ...]]:
        self.calls.append(("reset", seed))
        return self.features

    def goal_region_bounds(self) -> tuple[float, ...]:
        return GOAL_REGION


def _wired() -> tuple[Tossing3DEnvironment, _RecordingBackend]:
    env = Tossing3DEnvironment()
    backend = _RecordingBackend()
    env._backend = backend  # type: ignore[assignment]
    env.set_state(state=build_state(env=env, seed=7))
    backend.calls.clear()
    return env, backend


def test_state_declares_every_object_the_symbolic_layer_grounds_over() -> None:
    state = build_state()
    assert set(state.data) == {
        _ENV.robot,
        _ENV.cube,
        _ENV.bin_object,
        _ENV.barrier,
        _ENV.goal_region,
        _ENV.scene,
    }


def test_build_state_places_the_goal_region_box_in_the_state() -> None:
    state = build_state()
    bounds = tuple(
        state.get(obj=_ENV.goal_region, feature_name=name)
        for name in ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")
    )
    assert bounds == GOAL_REGION


def test_take_action_routes_pick_to_the_backend_with_both_parameters() -> None:
    env, backend = _wired()
    env.take_action(action=np.array([float(_ENV.SKILL_PICK), 0.57, -0.3]))
    assert backend.calls == [("pick", pytest.approx(0.57), pytest.approx(-0.3))]


def test_take_action_routes_toss_with_the_swing_dial() -> None:
    env, backend = _wired()
    env.take_action(action=np.array([float(_ENV.SKILL_TOSS), 0.75, 0.0]))
    assert backend.calls == [("toss", pytest.approx(0.75))]


def test_move_to_throw_pose_uses_the_fixed_standoff_not_the_action_parameter() -> None:
    """The standoff is a ClassVar precisely so AtThrowPose can read it; a per-action
    override would let the base stop somewhere the predicate then denies."""
    env, backend = _wired()
    env.take_action(action=np.array([float(_ENV.SKILL_MOVE_TO_THROW_POSE), 99.0, 0.0]))
    assert backend.calls == [("move", _ENV.throw_standoff)]


def test_take_action_is_total_over_the_whole_action_space() -> None:
    """A raw Box contains +-inf and every unknown skill id. None of it may raise, and
    none of it may reach the simulator."""
    env, backend = _wired()
    for action in (
        np.array([np.inf, 0.0, 0.0]),
        np.array([np.nan, 0.0, 0.0]),
        np.array([float(_ENV.SKILL_TOSS), np.inf, 0.0]),
        np.array([float(_ENV.SKILL_PICK), 0.5, np.nan]),
        np.array([7.0, 0.0, 0.0]),
        np.array([-3.0, 0.0, 0.0]),
    ):
        state = env.take_action(action=action)
        assert state is not None
    assert backend.calls == []


def test_take_action_preserves_the_scene_seed() -> None:
    """The seed is what makes a State restorable; losing it mid-episode would make
    reset_to_task silently reset to the wrong episode."""
    env, _ = _wired()
    state = env.take_action(action=np.array([float(_ENV.SKILL_PICK), 0.55, 0.0]))
    assert state.get(obj=_ENV.scene, feature_name="seed") == 7


def test_set_state_reinstalls_by_re_running_that_states_kinder_seed() -> None:
    env = Tossing3DEnvironment()
    backend = _RecordingBackend()
    env._backend = backend  # type: ignore[assignment]
    env.set_state(state=build_state(env=env, seed=1234))
    assert backend.calls == [("reset", 1234)]


def test_hard_reset_uses_the_canonical_seed() -> None:
    env = Tossing3DEnvironment(canonical_seed=42)
    backend = _RecordingBackend()
    env._backend = backend  # type: ignore[assignment]
    env.hard_reset()
    assert backend.calls == [("reset", 42)]
    assert env.get_current_state().get(obj=_ENV.scene, feature_name="seed") == 42


def test_get_valid_actions_is_empty_because_two_skills_are_continuous() -> None:
    assert Tossing3DEnvironment().get_valid_actions() == []
