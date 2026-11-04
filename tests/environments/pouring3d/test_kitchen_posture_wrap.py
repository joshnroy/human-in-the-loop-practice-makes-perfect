"""Joint 5's posture attractor sits on the turn nearest the arm when a skill starts.

Joint 5 is continuous, so an arm wound one full turn from neutral is the same arm. An
absolute posture residual pulls such an arm ~2*pi back toward the posture on every IK
call, reconfiguring the other joints to keep the hand pose. Wrapping the residual modulo
a turn on every call fixes that but flips the attractor to the next turn the moment a
skill carries joint 5 past posture - pi, after which it winds on. So the turn is chosen
once, at the start of each skill, and the residual within the skill is plain subtraction.
"""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.native_controller import NativePouringController

pytestmark = pytest.mark.xdist_group("kitchen_posture_wrap")

TURN = 2 * np.pi
POSTURE_JOINT = 4


class Arm:
    """A kitchen arm with joint 5 optionally wound by whole turns, plus a nearby target."""

    @staticmethod
    def kitchen(*, turns: float, posture: float = 0.0) -> KitchenController:
        c = KitchenEnvironment(requested_count=1).controller()
        c.posture = c.posture_goal = posture
        Arm.wind(c=c, by=turns * TURN)
        c.choose_posture_turn()
        return c

    @staticmethod
    def wind(*, c: KitchenController, by: float) -> None:
        c.data.qpos[c.arm_q[POSTURE_JOINT]] += by
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()

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


def force_bounded_recovery(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Route IK through PotstickerController's bounded recovery, the term's second copy."""

    def infeasible(self, **_):  # noqa: PLR0917 -- patched bound method
        raise ValueError("IK infeasible (forced by test)")

    monkeypatch.setattr(NativePouringController, "inverse_kinematics", infeasible)


@pytest.mark.parametrize("turns", [-2, -1, 0, 1, 3])
def test_the_chosen_turn_is_the_one_nearest_the_arm(*, turns):
    c = Arm.kitchen(turns=turns, posture=1.0)
    assert c.posture_turn == pytest.approx(turns * TURN, abs=1e-12)


@pytest.mark.parametrize("recovery", [False, True])
def test_a_wound_arm_solves_like_an_unwound_one(*, monkeypatch, recovery):
    if recovery:
        force_bounded_recovery(monkeypatch=monkeypatch)
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
def test_crossing_posture_minus_pi_within_a_skill_does_not_flip_the_attractor(
    *, monkeypatch, recovery
):
    # After a pour the posture rests at pour_posture = 1.0 rad. A later skill starts
    # unwound, then carries joint 5 past posture - pi; the posture term must keep
    # pulling it back up toward +1.0, not on down toward 1.0 - 2*pi.
    if recovery:
        force_bounded_recovery(monkeypatch=monkeypatch)
    c = Arm.kitchen(turns=0, posture=1.0)
    assert c.posture_turn == 0.0
    Arm.wind(c=c, by=1.0 - np.pi - 0.3 - c.data.qpos[c.arm_q[POSTURE_JOINT]])
    before = float(c.last_q[POSTURE_JOINT])
    after = float(Arm.solve(c=c)[POSTURE_JOINT])
    assert after > before + 1e-4


def test_each_skill_rechooses_the_turn_from_the_arm_it_starts_from():
    env = KitchenEnvironment(requested_count=1)
    provider = KitchenSkillProvider(env=env)
    c = env.controller()
    Arm.wind(c=c, by=-TURN)
    wait = provider.ground(name="Wait", request=1)
    env.take_action(action=provider.encode(ground_skill=wait, params=np.zeros(0)))
    assert env.last_info["success"], env.last_info
    assert c.posture_turn == pytest.approx(-TURN, abs=1e-12)


@pytest.mark.parametrize("recovery", [False, True])
def test_an_unwound_arm_solves_bit_identically_to_the_absolute_residual(*, monkeypatch, recovery):
    if recovery:
        force_bounded_recovery(monkeypatch=monkeypatch)
    chosen = Arm.solve(c=Arm.kitchen(turns=0, posture=0.4))
    monkeypatch.setattr(
        NativePouringController,
        "posture_residual",
        lambda self, *, angle: angle - self.posture,
    )
    absolute = Arm.solve(c=Arm.kitchen(turns=0, posture=0.4))
    np.testing.assert_array_equal(chosen, absolute)
