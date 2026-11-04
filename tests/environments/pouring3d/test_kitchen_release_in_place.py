"""A dumpling skill that fails holding its dumpling releases it where it is.

Without this, a failed carry left the hand closed on a dumpling, and every skill
that could recover (another pick, the jug pick, the human pour-back) requires an
empty hand: the run stranded. The skill still reports failure.
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_release_in_place")

DUMPLING = 2
PICK = np.array([SKILL_NAMES.index("PickShrimpFromTray"), DUMPLING, 1, 0, 0, 0], dtype=float)
DROP = np.array([SKILL_NAMES.index("DropShrimpIntoPitcher"), DUMPLING, 1, 0, 0, 0], dtype=float)
# The recorded stranding (2026-11-03, cycle 3 of a four-cycle kitchen sequence).
RAISE_ERROR = (
    "IK infeasible in raise held shrimp in aisle: residual 0.0081; "
    "joint degrees [53.0, 29.9, 96.9, -40.7, 2.6, -119.7, 158.3]"
)
LIFT_ERROR = "IK infeasible in lift shrimp: residual 0.0081; joint degrees []"


class Scene:
    @staticmethod
    def dumpling_in_pan() -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=1)
        c = env.controller()
        m, d = c.model, c.data
        joint = int(m.jnt_qposadr[m.body(c.scene.cube_names[DUMPLING]).jntadr[0]])
        pan = d.xpos[m.body("tray").id]
        d.qpos[joint : joint + 3] = [pan[0], pan[1], pan[2] + 0.03]
        d.qpos[joint + 3 : joint + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        c.pause(phase="settle test dumpling", seconds=10, grip=0)
        env.current_state = env.observe()
        assert c.scene.cube_locations()[DUMPLING] == "tray"
        return env

    @staticmethod
    def fail_motion(*, monkeypatch, error: str, when) -> None:
        """Make the arm/base motion matching `when(controller, phase)` infeasible."""
        move, move_base = KitchenController.move, KitchenController.move_base

        def failing_move(self, **kwargs):  # noqa: PLR0917 -- patched method
            if when(self, kwargs["phase"]):
                raise ValueError(error)
            return move(self, **kwargs)

        def failing_move_base(self, **kwargs):  # noqa: PLR0917 -- patched method
            if when(self, kwargs["phase"]):
                raise ValueError(error)
            return move_base(self, **kwargs)

        monkeypatch.setattr(KitchenController, "move", failing_move)
        monkeypatch.setattr(KitchenController, "move_base", failing_move_base)

    @staticmethod
    def assert_released_and_recoverable(*, env: KitchenEnvironment, error: str) -> None:
        c = env.controller()
        name = c.scene.cube_names[DUMPLING]
        assert env.last_info["success"] is False, env.last_info
        assert env.last_info["error"] == error
        assert c.held is None
        assert not c.finger_contacts(name=name)
        # Located by the physics on something the robot or the human can recover from.
        assert c.scene.cube_locations()[DUMPLING] in {"floor", "surface", "tray"}
        atoms = env.observe().abstract_atoms
        assert ("HandEmpty", ("robot",)) in atoms
        assert not any(a[0] == "HoldingShrimp" for a in atoms)
        assert env.reset_movables(destination=HUMAN_RESET) is True
        assert c.held is None
        assert c.scene.cube_locations() == ["pitcher"] * len(c.scene.cube_names)


def test_a_drop_that_cannot_carry_its_dumpling_releases_it_in_place(*, monkeypatch):
    env = Scene.dumpling_in_pan()
    c = env.controller()
    env.take_action(action=PICK)
    assert env.last_info["success"] is True, env.last_info
    assert c.held == c.scene.cube_names[DUMPLING]
    # Every transport motion with the dumpling in hand is infeasible, as at the
    # recorded stranding; motions after the release are left alone.
    Scene.fail_motion(
        monkeypatch=monkeypatch,
        error=RAISE_ERROR,
        when=lambda self, phase: self.held in self.scene.cube_names,
    )
    env.take_action(action=DROP)
    Scene.assert_released_and_recoverable(env=env, error=RAISE_ERROR)


def test_a_pick_that_cannot_lift_its_dumpling_releases_it_in_place(*, monkeypatch):
    env = Scene.dumpling_in_pan()
    Scene.fail_motion(
        monkeypatch=monkeypatch,
        error=LIFT_ERROR,
        when=lambda self, phase: phase == "lift shrimp",
    )
    env.take_action(action=PICK)
    Scene.assert_released_and_recoverable(env=env, error=LIFT_ERROR)
