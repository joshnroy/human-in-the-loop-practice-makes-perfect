"""Post-toss settling must stop on rest, remain bounded, and hold the robot still."""

from types import SimpleNamespace

import numpy as np
import pytest

from hitl_pmp.environments.tossing3d.kinder_backend import KinderBackend, TossSettlingController


class _State:
    def __init__(self, *, speed=0.0, angular_speed=0.0):
        self.speed = speed
        self.angular_speed = angular_speed

    def get_object_from_name(self, name):  # noqa: PLR0917 (KINDER signature)
        return name

    def get(self, obj, feature):  # noqa: PLR0917 (KINDER signature)
        if obj == "cube_0":
            return {"vx": self.speed, "wx": self.angular_speed}.get(feature, 0.0)
        return 0.4 if feature == "pos_gripper" else 0.2


class _Toss:
    def __init__(self):
        self.done = False

    def reset(self, state, params):  # noqa: PLR0917 (KINDER signature)
        del state, params
        self.done = False

    def step(self):
        return np.full(11, 0.1)

    def observe(self, state):  # noqa: PLR0917 (KINDER signature)
        del state
        self.done = True

    def terminated(self):
        return self.done


def _controller(*, delta=True, dimensions=11):
    robot_env = SimpleNamespace(control_frequency=10, act_delta=delta)
    backend = SimpleNamespace(
        robot_name="robot",
        cube_name="cube_0",
        _env=SimpleNamespace(action_space=SimpleNamespace(shape=(dimensions,))),
        _object_centric=lambda: SimpleNamespace(_robot_env=robot_env),
    )
    controller = TossSettlingController(backend=backend, controller=_Toss())
    controller.reset(_State(), None)
    np.testing.assert_array_equal(controller.step(), np.full(11, 0.1))
    controller.observe(_State())
    assert not controller.terminated()
    return controller


def test_settling_needs_three_consecutive_quiet_ticks():
    controller = _controller()
    for speed in (0.0, 0.0, 0.2, 0.0, 0.0):
        controller.step()
        controller.observe(_State(speed=speed))
        assert not controller.terminated()
    controller.step()
    controller.observe(_State())
    assert controller.terminated()


@pytest.mark.parametrize("angular", [False, True])
def test_unsettled_cube_exhausts_two_seconds_without_an_unbounded_wait(*, angular):
    controller = _controller()
    for index in range(20):
        controller.step()
        controller.observe(
            _State(speed=0.0 if angular else 0.2, angular_speed=1.0 if angular else 0.0)
        )
        assert controller.terminated() == (index == 19)


@pytest.mark.parametrize("delta", [False, True])
@pytest.mark.parametrize("dimensions", [11, 18])
def test_wait_holds_pose_and_preserves_gripper_command(*, delta, dimensions):
    controller = _controller(delta=delta, dimensions=dimensions)
    action = controller.step()
    assert action.shape == (dimensions,)
    np.testing.assert_allclose(action[:10], 0.0 if delta else 0.2)
    assert action[10] == 0.4
    if dimensions == 18:
        np.testing.assert_array_equal(action[11:], np.zeros(7))


def test_backend_wraps_the_native_toss_controller(*, monkeypatch):
    toss = _Toss()
    api = SimpleNamespace(
        tossing_controllers=lambda *args, **kwargs: {
            "move_to_toss_location_and_toss": SimpleNamespace(ground=lambda objects: toss)
        },
        shelf_controllers=None,
    )
    backend = KinderBackend()
    backend._env = SimpleNamespace(action_space=None)
    monkeypatch.setattr(KinderBackend, "api", lambda self: api)  # noqa: PLR0917
    controller = backend._ground_controller(
        module="tossing",
        key="move_to_toss_location_and_toss",
        object_names=("robot",),
        state=_State(),
    )
    assert isinstance(controller, TossSettlingController)
    assert controller.controller is toss


def test_upstream_failure_is_reported_without_a_settling_delay():
    controller = _controller()
    controller.controller.failure = "aborted toss"
    assert controller.failure == "aborted toss"
    assert controller.terminated()


def test_timeout_hook_remains_visible_to_the_execution_loop():
    controller = _controller()
    called = []
    controller.controller.on_step_limit = lambda: called.append(True)
    controller.on_step_limit()
    assert called == [True]
