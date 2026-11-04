"""The joint-5 posture term compares angles modulo a turn, not absolute angles.

Joint 5 is continuous, so an arm wound one full turn from neutral is the same arm. An
absolute posture residual would pull such an arm ~2*pi back toward the posture on every
IK call, reconfiguring the other joints to keep the hand pose.
"""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.native_controller import NativePouringController

pytestmark = pytest.mark.xdist_group("kitchen_posture_wrap")

TURN = 2 * np.pi
POSTURE_JOINT = 4


class Arm:
    """A kitchen arm with joint 5 optionally wound by whole turns, plus a nearby target."""

    @staticmethod
    def kitchen(*, turns: float) -> KitchenController:
        c = KitchenEnvironment(requested_count=1).controller()
        c.data.qpos[c.arm_q[POSTURE_JOINT]] += turns * TURN
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        return c

    @staticmethod
    def solve(*, c: KitchenController) -> np.ndarray:
        hand, rotation = c.hand_pose()
        return c.inverse_kinematics(position=hand + np.array([0.0, 0.0, 0.03]), rotation=rotation)

    @staticmethod
    def solve_hand(*, c: KitchenController, q: np.ndarray) -> np.ndarray:
        c.scratch.qpos[:] = c.data.qpos
        c.scratch.qpos[c.arm_q] = q
        mujoco.mj_kinematics(c.model, c.scratch)
        return c.scratch.site_xpos[c.site].copy()


@pytest.mark.parametrize("angle", [0.0, 0.3, -1.2, 3.0, -3.0, 3.1415])
def test_the_offset_is_plain_subtraction_within_half_a_turn(*, angle):
    for posture in (0.0, 0.4, 1.0):
        if abs(angle - posture) < np.pi:
            assert NativePouringController.posture_offset(angle=angle, posture=posture) == (
                angle - posture
            )


@pytest.mark.parametrize("turns", [-2, -1, 1, 3])
def test_the_offset_ignores_whole_turns(*, turns):
    for angle in (0.2, -2.5, 2.9):
        wound = NativePouringController.posture_offset(angle=angle + turns * TURN, posture=0.5)
        assert wound == pytest.approx(angle - 0.5, abs=1e-9)
        assert abs(wound) <= np.pi


@pytest.mark.parametrize("recovery", [False, True])
def test_a_wound_arm_solves_like_an_unwound_one(*, monkeypatch, recovery):
    if recovery:
        # Force PotstickerController's bounded-recovery IK, the second copy of the term.
        def infeasible(self, **_):  # noqa: PLR0917 -- patched bound method
            raise ValueError("IK infeasible (forced by test)")

        monkeypatch.setattr(NativePouringController, "inverse_kinematics", infeasible)
    flat, wound = Arm.kitchen(turns=0), Arm.kitchen(turns=-1)
    q_flat, q_wound = Arm.solve(c=flat), Arm.solve(c=wound)
    shift = np.zeros(7)
    shift[POSTURE_JOINT] = -TURN
    # Same solution one turn over: no other joint reconfigures to fight the posture term.
    np.testing.assert_allclose(q_wound, q_flat + shift, atol=1e-3)
    np.testing.assert_allclose(
        Arm.solve_hand(c=wound, q=q_wound), Arm.solve_hand(c=flat, q=q_flat), atol=1e-4
    )


@pytest.mark.parametrize("recovery", [False, True])
def test_an_unwound_arm_solves_bit_identically_to_the_absolute_residual(*, monkeypatch, recovery):
    if recovery:

        def infeasible(self, **_):  # noqa: PLR0917 -- patched bound method
            raise ValueError("IK infeasible (forced by test)")

        monkeypatch.setattr(NativePouringController, "inverse_kinematics", infeasible)
    wrapped = Arm.solve(c=Arm.kitchen(turns=0))
    monkeypatch.setattr(
        NativePouringController,
        "posture_offset",
        staticmethod(lambda *, angle, posture: angle - posture),
    )
    absolute = Arm.solve(c=Arm.kitchen(turns=0))
    np.testing.assert_array_equal(wrapped, absolute)
