"""Continuous arm joints are unwound by whole turns before every empty-hand pick."""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.potsticker_controller import PotstickerController
from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

pytestmark = pytest.mark.xdist_group("kitchen_unwind")

TURN = 2 * np.pi
CONTINUOUS = [0, 2, 4, 6]


class Wound:
    """A kitchen whose shoulder joint has ratcheted two whole turns from neutral."""

    @staticmethod
    def kitchen(*, turns: dict[int, int]) -> KitchenController:
        env = KitchenEnvironment(requested_count=1)
        c = env.controller()
        for joint, n in turns.items():
            c.data.qpos[c.arm_q[joint]] += n * TURN
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.pause(phase="settle wound arm", seconds=1, grip=0)
        return c

    @staticmethod
    def entry(*, monkeypatch, owner: type, name: str) -> list[tuple[np.ndarray, np.ndarray]]:
        """Record arm joints and hand position where the wrapped pick begins, then stop."""
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def record(self, **_):  # noqa: PLR0917 -- patched bound method
            seen.append((self.data.qpos[self.arm_q].copy(), self.hand_pose()[0]))

        monkeypatch.setattr(owner, name, record)
        return seen


def test_the_continuous_joints_are_the_ones_unwound():
    c = KitchenEnvironment(requested_count=1).controller()
    assert np.flatnonzero(c.lower == -12).tolist() == CONTINUOUS
    np.testing.assert_array_equal(c.arm_neutral_q, c.data.qpos[c.arm_q])


@pytest.mark.parametrize(
    ("owner", "name", "pick"),
    [
        (PracticePourController, "pick_pitcher", lambda c: c.pick_pitcher()),
        (
            PotstickerController,
            "pick_payload_with_retry",
            lambda c: c.pick_shrimp(index=2, location="tray"),
        ),
    ],
)
def test_a_pick_starts_within_half_a_turn_of_neutral_with_the_hand_where_it_was(
    *, monkeypatch, owner, name, pick
):
    c = Wound.kitchen(turns={0: 2, 4: -1})
    before_q, before_hand = c.data.qpos[c.arm_q].copy(), c.hand_pose()[0]
    seen = Wound.entry(monkeypatch=monkeypatch, owner=owner, name=name)
    pick(c)
    (q, hand) = seen[0]
    offset = q - c.arm_neutral_q
    assert np.all(np.abs(offset[CONTINUOUS]) <= np.pi)
    # Whole turns only: every joint ends where it was, modulo a turn, and so does the hand.
    wrapped = np.arctan2(np.sin(q - before_q), np.cos(q - before_q))
    assert np.max(np.abs(wrapped)) < 0.03
    assert np.linalg.norm(hand - before_hand) < 0.01


def test_an_unwound_arm_is_left_alone(*, monkeypatch):
    c = Wound.kitchen(turns={})
    steps = c.steps
    seen = Wound.entry(monkeypatch=monkeypatch, owner=PracticePourController, name="pick_pitcher")
    c.pick_pitcher()
    assert seen and c.steps == steps


def test_nothing_is_unwound_while_holding():
    c = Wound.kitchen(turns={0: 2})
    c.held = "pitcher"
    with pytest.raises(ValueError, match="empty hand"):
        c.unwind_continuous_joints()


def test_a_held_motion_near_the_ik_bound_fails_loudly():
    c = Wound.kitchen(turns={})
    c.held = "pitcher"
    c.last_q = c.data.qpos[c.arm_q].copy()
    c.last_q[0] = c.upper[0] - 0.1
    hand, rotation = c.hand_pose()
    with pytest.raises(ValueError, match="IK bound"):
        c.inverse_kinematics(position=hand, rotation=rotation)
