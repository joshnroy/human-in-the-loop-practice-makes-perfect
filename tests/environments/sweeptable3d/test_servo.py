"""Fast sweep feedback checks using mocks, without simulator stepping."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hitl_pmp.environments.sweeptable3d.servo import SweepServo


class FakeShelf:
    """Expose disjoint position/velocity indices and mutable feedback state."""

    def __init__(self):
        self.qa = np.arange(4, 11)
        self.va = np.arange(3, 10)
        self.site = 1
        self.ticks = 0
        self.arm_reference = np.zeros(7)
        self.model = SimpleNamespace(nv=16, site_bodyid=np.array([0, 2]))
        self.data = SimpleNamespace(
            qpos=np.zeros(17),
            qvel=np.zeros(16),
            qfrc_bias=np.zeros(16),
            xpos=np.array([[0.0, 0, 0], [0, 0, 0], [1, 2, 3]]),
            xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1)),
            geom_xpos=np.array([[1.2, 2.3, 3.4]]),
        )
        self.robot = SimpleNamespace(
            _update_ctrl=Mock(),
            _compute_arm_torques=Mock(
                side_effect=lambda target_positions: (
                    10 * (target_positions - self.data.qpos[self.qa])
                    - 2 * self.data.qvel[self.va]
                    + self.data.qfrc_bias[self.va]
                )
            ),
            ARM_TORQUE_LIMITS=np.arange(1.0, 8.0),
        )
        self.core = SimpleNamespace(_robot_env=self.robot)
        self.mujoco = SimpleNamespace(
            mj_jac=Mock(side_effect=lambda *args: self.fill_jacobian(args=args))
        )

    def fill_jacobian(self, *, args):
        args[2][:] = 1000
        args[2][:, self.va] = 0
        args[2][2, self.va] = [1, -1, 0.5, -0.5, 0, 0, 0]


def test_latches_wrapped_clipped_target_and_refreshes_feedback():
    shelf = FakeShelf()
    shelf.robot.ARM_TORQUE_LIMITS[:] = 100
    shelf.data.qpos[shelf.qa] = [np.pi - 0.02, -np.pi + 0.02, 0, 0, 0, 0, 0]
    shelf.arm_reference = np.array([-np.pi + 0.02, np.pi - 0.02, 1, -1, 0.03, 0, 0])
    initial = shelf.data.qpos[shelf.qa].copy()
    expected = initial + [0.04, -0.04, 0.1, -0.1, 0.03, 0, 0]
    action = np.array([1.2, -2.3, 0.4, 99, 99, 99, 99, 99, 99, 99, 204.0])
    action.setflags(write=False)
    original = shelf.robot._update_ctrl
    compute = shelf.robot._compute_arm_torques
    with SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0)) as servo:
        assert servo.normal_force_n == 0
        shelf.robot._update_ctrl(action)
        np.testing.assert_allclose(compute.call_args.kwargs["target_positions"], expected)
        np.testing.assert_allclose(original.call_args.args[0][3:10], 10 * (expected - initial))
        shelf.data.qpos[shelf.qa] += 0.01
        shelf.data.qvel[shelf.va] = 0.02
        shelf.data.qfrc_bias[shelf.va] = 0.03
        shelf.arm_reference[:] = 0.8
        shelf.robot._update_ctrl(action)
        np.testing.assert_allclose(compute.call_args.kwargs["target_positions"], expected)
        np.testing.assert_allclose(
            original.call_args.args[0][3:10], 10 * (expected - initial - 0.01) - 0.01
        )
        shelf.ticks += 1
        current = shelf.data.qpos[shelf.qa].copy()
        delta = (shelf.arm_reference - current + np.pi) % (2 * np.pi) - np.pi
        shelf.robot._update_ctrl(action)
        np.testing.assert_allclose(
            compute.call_args.kwargs["target_positions"], current + np.clip(delta, -0.1, 0.1)
        )
        assert compute.call_count == servo.servo_updates == 3
        assert servo.clipped_torque_count == 0
        assert not shelf.mujoco.mj_jac.called
    assert shelf.robot._update_ctrl is original
    for call in original.call_args_list:
        forwarded = call.args[0]
        assert not np.shares_memory(forwarded, action)
        np.testing.assert_array_equal(forwarded[[0, 1, 2, 10]], action[[0, 1, 2, 10]])
    np.testing.assert_array_equal(action[3:10], np.full(7, 99))


def test_preload_uses_rigid_robot_head_point_and_clamps_arm_torques():
    shelf = FakeShelf()
    shelf.data.xmat[2] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).reshape(9)
    original = shelf.robot._update_ctrl
    with SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0)) as servo:
        servo.normal_force_n = 100
        assert servo.normal_force_n == servo.max_normal_force_n == 15
        shelf.data.geom_xpos[0] = [100, 200, 300]
        shelf.data.xmat[2] = np.eye(3).reshape(9)
        shelf.data.xpos[2] = [4, 5, 6]
        shelf.robot._update_ctrl(np.zeros(11))
        jac_args = shelf.mujoco.mj_jac.call_args.args
        assert jac_args[0] is shelf.model
        assert jac_args[1] is shelf.data
        assert jac_args[3] is None
        np.testing.assert_allclose(jac_args[4], [4.3, 4.8, 6.4])
        assert jac_args[5] == 2
        np.testing.assert_allclose(original.call_args.args[0][3:10], [-1, 2, -3, 4, 0, 0, 0])
        assert servo.clipped_torque_count == 4
        servo.normal_force_n = 0.5
        shelf.ticks += 1
        shelf.robot._update_ctrl(np.zeros(11))
        np.testing.assert_allclose(
            original.call_args.args[0][3:10], [-0.5, 0.5, -0.25, 0.25, 0, 0, 0]
        )
        assert servo.clipped_torque_count == 4
        assert servo.servo_updates == 2


def test_callback_does_not_write_simulator_state():
    shelf = FakeShelf()
    for value in vars(shelf.data).values():
        value.setflags(write=False)
    with SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0)) as servo:
        servo.normal_force_n = 2
        shelf.robot._update_ctrl(np.zeros(11))
    assert servo.servo_updates == 1


@pytest.mark.parametrize("failure", ["body", "compute", "jacobian", "callback"])
@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt])
def test_restores_exact_prior_bound_callable_after_exception(*, failure, exception_type):
    shelf = FakeShelf()
    original = Mock(side_effect=exception_type("failure")) if failure == "callback" else [].append
    shelf.robot._update_ctrl = original
    servo = SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0))
    with pytest.raises(exception_type, match="failure"), servo:
        if failure == "body":
            raise exception_type("failure")
        if failure == "compute":
            shelf.robot._compute_arm_torques.side_effect = exception_type("failure")
        if failure == "jacobian":
            servo.normal_force_n = 1
            shelf.mujoco.mj_jac.side_effect = exception_type("failure")
        shelf.robot._update_ctrl(np.zeros(11))
    assert shelf.robot._update_ctrl is original


def test_instance_local_hook_reentry_guard_and_reuse():
    shelf = FakeShelf()
    other = FakeShelf()
    other_update = other.robot._update_ctrl
    servo = SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0))
    original = shelf.robot._update_ctrl
    for target in [0.03, -0.02]:
        shelf.arm_reference[:] = target
        with servo:
            assert other.robot._update_ctrl is other_update
            with pytest.raises(RuntimeError, match="already installed"):
                servo.__enter__()
            shelf.robot._update_ctrl(np.zeros(11))
            np.testing.assert_allclose(original.call_args.args[0][3:10], 10 * target)
        assert shelf.robot._update_ctrl is original


def test_force_bounds_are_configurable_and_downward_only():
    servo = SweepServo(shelf=FakeShelf(), wiper=SimpleNamespace(head_geom=0), max_normal_force_n=3)
    servo.normal_force_n = 4
    assert servo.normal_force_n == 3
    servo.normal_force_n = -1
    assert servo.normal_force_n == 0


@pytest.mark.parametrize("force", [np.nan, np.inf, -np.inf])
def test_nonfinite_force_is_rejected(*, force):
    servo = SweepServo(shelf=FakeShelf(), wiper=SimpleNamespace(head_geom=0))
    with pytest.raises(ValueError, match="finite"):
        servo.normal_force_n = force
    assert servo.normal_force_n == 0


@pytest.mark.parametrize("force", [-1, np.nan, np.inf, -np.inf])
def test_invalid_force_limit_is_rejected(*, force):
    shelf = FakeShelf()
    original = shelf.robot._update_ctrl
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0), max_normal_force_n=force)
    assert shelf.robot._update_ctrl is original


def test_control_update_applies_new_force_in_same_tick_without_relatching_target():
    shelf = FakeShelf()
    original = shelf.robot._update_ctrl
    forces = iter([0.5, 0.0])
    update = Mock(side_effect=lambda: setattr(servo, "normal_force_n", next(forces)))
    with SweepServo(
        shelf=shelf, wiper=SimpleNamespace(head_geom=0), control_update=update
    ) as servo:
        shelf.robot._update_ctrl(np.zeros(11))
        np.testing.assert_allclose(
            original.call_args.args[0][3:10], [-0.5, 0.5, -0.25, 0.25, 0, 0, 0]
        )
        shelf.arm_reference[:] = 1.0
        shelf.robot._update_ctrl(np.zeros(11))
        np.testing.assert_allclose(original.call_args.args[0][3:10], np.zeros(7))
        assert update.call_count == servo.servo_updates == 2
    assert shelf.robot._update_ctrl is original


def test_control_update_exception_restores_original_callback():
    shelf = FakeShelf()
    original = shelf.robot._update_ctrl
    update = Mock(side_effect=RuntimeError("contact feedback failed"))
    with (
        pytest.raises(RuntimeError, match="contact feedback failed"),
        SweepServo(shelf=shelf, wiper=SimpleNamespace(head_geom=0), control_update=update),
    ):
        shelf.robot._update_ctrl(np.zeros(11))
    assert shelf.robot._update_ctrl is original
    shelf.robot._compute_arm_torques.assert_not_called()
    original.assert_not_called()
