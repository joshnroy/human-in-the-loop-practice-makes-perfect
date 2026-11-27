"""The kitchen drop's settle wait and `ready()` judge rest by position, not by speed.

A flat dumpling balanced on one mesh contact rocks in place: KinDER's speed rule
(`cubes_settled`) calls it moving while its centre stays put. In the 2026-11-04 Della
run that failed 5/60 drops with "did not settle" and refused the next PickPitcher.
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController, KitchenScene
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider
from hitl_pmp.environments.pouring3d.practice.controller import PracticePourController

pytestmark = pytest.mark.xdist_group("kitchen_drop_settle")

SETTLE_PHASE = "settle returned potstickers"


class Rocking:
    """KinDER's speed rule as it reads a dumpling rocking on one contact point."""

    @staticmethod
    def never_settled(self) -> bool:  # noqa: PLR0917 -- patched method
        del self
        return False


class Released:
    """The base drop with the release already done: the dumpling lies in the jug."""

    @staticmethod
    def drop_shrimp(self, *, index: int) -> None:  # noqa: PLR0917 -- patched method
        del index
        self.held = None
        self.pause(phase="release shrimp into pitcher", seconds=0.5, grip=0)


class Withdrawn:
    """Skip the vertical withdraw: from the episode-start posture it is out of reach."""

    @staticmethod
    def move_except_withdraw(*, original):
        def move(self, **kwargs) -> None:  # noqa: PLR0917 -- patched method
            if kwargs.get("phase") == "withdraw empty gripper above pitcher":
                return
            original(self, **kwargs)

        return move


class Sliding:
    """Drive one dumpling along an 8 mm-amplitude path during the settle wait."""

    @staticmethod
    def pause_moving_piece(*, original):
        origin: list[np.ndarray] = []

        def pause(  # noqa: PLR0917 -- patched method
            self, *, phase: str, seconds: float, grip: float | None = None
        ) -> None:
            original(self, phase=phase, seconds=seconds, grip=grip)
            if phase != SETTLE_PHASE:
                return
            piece = self.scene.get_object(self.scene.cube_names[0])
            if not origin:
                origin.append(self.scene.payload_positions()[0])
            offset = 0.008 * np.sin(2 * np.pi * self.data.time / 4.0)
            piece.set_pose(origin[0] + [offset, 0, 0], [1, 0, 0, 0])
            piece.set_velocity([0.0] * 3, [0.0] * 3)
            mujoco.mj_forward(self.model, self.data)

        return pause


def _rested_env(*, monkeypatch) -> KitchenEnvironment:
    env = KitchenEnvironment(requested_count=1)
    env.controller().pause(phase="observe rest", seconds=2.1)
    assert env.controller().scene.payload_at_rest()
    monkeypatch.setattr(KitchenScene, "cubes_settled", Rocking.never_settled)
    return env


def test_a_dumpling_rocking_in_place_leaves_the_scene_ready(*, monkeypatch):
    env = _rested_env(monkeypatch=monkeypatch)
    try:
        assert env.controller().ready()
        # ReadyToPour is gone from the kitchen; PickPitcher reads the same facts.
        pick = KitchenSkillProvider(env=env).ground(name="PickPitcher")
        assert {
            (a.predicate.name, tuple(o.name for o in a.objects)) for a in pick.preconditions
        } <= env.observe().abstract_atoms
    finally:
        env.close()


def test_the_drop_settle_wait_returns_while_a_dumpling_rocks_in_place(*, monkeypatch):
    monkeypatch.setattr(PracticePourController, "drop_shrimp", Released.drop_shrimp)
    monkeypatch.setattr(
        KitchenController, "move", Withdrawn.move_except_withdraw(original=KitchenController.move)
    )
    env = _rested_env(monkeypatch=monkeypatch)
    try:
        c = env.controller()
        c.held = c.scene.cube_names[0]
        c.drop_shrimp(index=0)
        assert c.ready()
    finally:
        env.close()


def test_the_drop_settle_wait_does_not_return_while_a_dumpling_still_moves(*, monkeypatch):
    monkeypatch.setattr(PracticePourController, "drop_shrimp", Released.drop_shrimp)
    monkeypatch.setattr(
        KitchenController, "move", Withdrawn.move_except_withdraw(original=KitchenController.move)
    )
    monkeypatch.setattr(
        KitchenController, "pause", Sliding.pause_moving_piece(original=KitchenController.pause)
    )
    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        c.held = c.scene.cube_names[0]
        with pytest.raises(ValueError, match="did not settle within 20 seconds"):
            c.drop_shrimp(index=0)
        assert not c.scene.payload_at_rest()
    finally:
        env.close()


def test_the_human_reset_still_reports_a_ready_scene():
    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        piece = c.scene.get_object(c.scene.cube_names[0])
        pan = c.data.xpos[c.model.body("tray").id].copy()
        piece.set_pose(pan + [0, 0, 0.009], [1, 0, 0, 0])
        piece.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        assert env.reset_movables(destination=HUMAN_RESET)
        assert env.last_info["success"]
        # The respawn is a jump the position history saw, so the speed rule carries it.
        assert not c.scene.payload_at_rest()
        assert c.ready()
    finally:
        env.close()
