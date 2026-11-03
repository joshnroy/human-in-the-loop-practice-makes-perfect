"""The held kitchen jug must not creep round the pad-to-pad axis while it is carried.

Before the fix the handle's 0.005 torsional coefficient let MuJoCo's soft friction creep
under the jug's ~0.2 N·m gravity torque: ~0.2 deg/s upright, so after two pours without
setting the jug down it sat 28-37 deg rolled in the hand and no upright place plan
existed (plans exist up to ~27 deg).
"""

import importlib.util

import numpy as np
import pytest

# Imported inside each test: collecting this file (as tests/test_main_only_marker.py does
# on every CI job) must not import KinDER's OpenGL stack.
pytestmark = [
    pytest.mark.xdist_group("kitchen_grip_slip"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]

HANDLE_GEOMS = ("pitcher_handle_upper", "pitcher_handle_grip")
# Comfortably under the ~27 deg roll at which the upright place plan stops existing.
MAX_IN_HAND_ROTATION_DEG = 5.0
MAX_AFTER_TWO_POURS_DEG = 8.0
# A run of the stranded curtis/pomdp/2 episode: two failing-then-delivering pours,
# back to back, with the jug never set down in between.
CURTIS_POUR_1 = (1.0, 0.0, 1.0, 78.18302721227074, 0.017913403446968946, 18.4079314001052)
CURTIS_POUR_2 = (1.0, 0.0, 2.0, 91.11100710807476, 0.014526104879165048, 13.19366702629978)
PLACE_PITCHER_AT_HOME = (2.0, 0.0, 3.0, 0.0, 0.0, 0.0)
PICK_PITCHER = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


class InHandRotation:
    """A static-method container, never instantiated."""

    @staticmethod
    def hand_in_jug(*, controller):
        import mujoco

        mujoco.mj_forward(controller.model, controller.data)
        _, jug = controller.body_pose(name="pitcher")
        _, hand = controller.hand_pose()
        return jug.inv() * hand

    @staticmethod
    def degrees(*, controller, reference) -> float:
        relative = InHandRotation.hand_in_jug(controller=controller) * reference.inv()
        return float(np.degrees(relative.magnitude()))


def _picked():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    env.reset_to_seed(seed=0, requested=1)
    env.take_action(action=np.array(PICK_PITCHER))
    assert env.last_info["success"], env.last_info
    c = env.controller()
    assert c.held == "pitcher"
    return env, c, InHandRotation.hand_in_jug(controller=c)


def test_only_the_kitchen_jug_handle_takes_the_raised_torsional_friction():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenScene
    from hitl_pmp.environments.pouring3d.practice.environment import FrozenPourEnvironment

    assert KitchenScene.HANDLE_TORSIONAL_FRICTION >= 0.02
    env = KitchenEnvironment(requested_count=1)
    try:
        m = env.controller().model
        for name in HANDLE_GEOMS:
            g = m.geom(name).id
            assert m.geom_friction[g][1] == KitchenScene.HANDLE_TORSIONAL_FRICTION
            # Torsional friction acts only at condim >= 4; sliding and rolling unchanged.
            assert m.geom_condim[g] == 4
            assert m.geom_friction[g][0] == pytest.approx(0.7)
            assert m.geom_friction[g][2] == pytest.approx(1e-4)
        # Every other jug geom keeps its own coefficient.
        others = [
            g
            for g in range(m.ngeom)
            if m.body(int(m.geom_bodyid[g])).name == "pitcher"
            and m.geom(g).name not in HANDLE_GEOMS
        ]
        assert others
        assert {round(float(m.geom_friction[g][1]), 6) for g in others} == {0.005}
    finally:
        env.close()
    legacy = FrozenPourEnvironment(layout="stations", payload_count=3, scene_bg=False)
    try:
        m = legacy.controller().model
        for name in HANDLE_GEOMS:
            assert m.geom_friction[m.geom(name).id][1] == pytest.approx(0.005)
    finally:
        legacy.close()


@pytest.mark.main_only
def test_the_held_jug_does_not_creep_in_the_hand_over_a_minute_upright():
    env, c, reference = _picked()
    try:
        c.pause(phase="hold upright", seconds=60, grip=1)
        assert c.held == "pitcher"
        rotation = InHandRotation.degrees(controller=c, reference=reference)
        assert rotation < MAX_IN_HAND_ROTATION_DEG, rotation
    finally:
        env.close()


@pytest.mark.main_only
def test_two_pours_without_setting_down_still_leave_an_upright_place():
    env, c, reference = _picked()
    try:
        for action in (CURTIS_POUR_1, CURTIS_POUR_2):
            env.take_action(action=np.array(action))
            assert env.last_info.get("error") is None, env.last_info
            assert c.held == "pitcher"
        rotation = InHandRotation.degrees(controller=c, reference=reference)
        assert rotation < MAX_AFTER_TWO_POURS_DEG, rotation
        env.take_action(action=np.array(PLACE_PITCHER_AT_HOME))
        assert env.last_info["success"], env.last_info
        assert c.held is None
    finally:
        env.close()
