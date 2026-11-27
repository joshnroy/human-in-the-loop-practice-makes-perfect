"""A drop whose last centring move over the jug falls short still releases into it.

Pick and place are assumed to work in this project (Josh, 2026-11-08). In the p = 0
Curtis run on agni the drop missed centring over the jug by 15 mm; the skill failed
and let the dumpling go elsewhere. Close enough over the opening, it now lets go there.
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_drop_releases_over_jug")

PICK = np.array([SKILL_NAMES.index("PickShrimpFromTray"), 0, 1, 0, 0, 0], dtype=float)
DROP = np.array([SKILL_NAMES.index("DropShrimpIntoPitcher"), 0, 1, 0, 0, 0], dtype=float)
CENTRE = "center held shrimp over pitcher"


def _holding_a_dumpling() -> KitchenEnvironment:
    env = KitchenEnvironment(requested_count=1)
    c = env.controller()
    name = c.scene.cube_names[0]
    tray, _ = c.body_pose(name="tray")
    c.scene.get_object(name).set_pose(tray + [0.0, 0.0, 0.03], [1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(c.model, c.data)
    c.pause(phase="settle in tray", seconds=2)
    env.take_action(action=PICK)
    assert c.held == name, env.last_info
    return env


def _short_centring(*, monkeypatch, short_by: float) -> None:
    original = KitchenController.move

    def move(self, **kwargs):  # noqa: PLR0917 -- patched method
        if kwargs.get("phase") != CENTRE:
            return original(self, **kwargs)
        kwargs["position"] = np.asarray(kwargs["position"]) + [short_by, 0.0, 0.0]
        original(self, **kwargs)
        raise ValueError(f"IK infeasible in {CENTRE}: residual {short_by:.4f}")

    monkeypatch.setattr(KitchenController, "move", move)


def test_a_drop_15_mm_short_over_the_jug_releases_into_it(*, monkeypatch):
    env = _holding_a_dumpling()
    try:
        _short_centring(monkeypatch=monkeypatch, short_by=0.015)
        env.take_action(action=DROP)
        c = env.controller()
        assert c.held is None
        assert c.scene.cube_in_vessel(c.scene.cube_names[0], "pitcher")
        assert env.last_info["success"], env.last_info
    finally:
        env.close()


def test_a_drop_far_from_the_jug_does_not_release_over_it(*, monkeypatch):
    env = _holding_a_dumpling()
    try:
        _short_centring(monkeypatch=monkeypatch, short_by=0.08)
        env.take_action(action=DROP)
        assert not env.last_info["success"], env.last_info
    finally:
        env.close()
