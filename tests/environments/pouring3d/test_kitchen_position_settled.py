"""Kitchen pours are judged settled by position; legacy pouring keeps KinDER's rule."""

import importlib.util

import numpy as np
import pytest

# Imported inside each test: collecting this file (as tests/test_main_only_marker.py does
# on every CI job) must not import KinDER's OpenGL stack.
pytestmark = [
    pytest.mark.xdist_group("kitchen_position_settled"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]


def test_gates_keep_kinders_rule_and_only_kitchen_outcomes_use_position():
    from kinder.envs.dynamic3d.pitcher_pour3d import ObjectCentricPitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenShrimpScene
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenScene
    from hitl_pmp.environments.pouring3d.potsticker_native import PotstickerScene
    from hitl_pmp.environments.pouring3d.practice.native import PracticeThreeScene

    kinder_rule = ObjectCentricPitcherPour3DEnv.physical_snapshot
    for scene in (PotstickerScene, PracticeThreeScene, FiveFrozenShrimpScene, KitchenScene):
        assert scene.cubes_settled is ObjectCentricPitcherPour3DEnv.cubes_settled
    for legacy in (PotstickerScene, PracticeThreeScene, FiveFrozenShrimpScene):
        assert not hasattr(legacy, "payload_at_rest")
    assert KitchenScene.physical_snapshot is not kinder_rule


def test_construction_is_unchanged_and_an_unobserved_outcome_is_not_settled():
    import mujoco

    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        # The start state the controllers were validated on: the base second of
        # settling only, gated by KinDER's speed rule, no extra hold.
        assert c.data.time == pytest.approx(1.0)
        assert c.ready()
        # No history yet, so no outcome can be called settled.
        assert not c.scene.payload_at_rest()
        assert not c.scene.physical_snapshot()["settled"]
        c.pause(phase="observe rest", seconds=2.1)
        assert c.scene.payload_at_rest()
        assert c.scene.physical_snapshot()["settled"]
        # A teleport the history did not see counts as movement.
        piece = c.scene.get_object(c.scene.cube_names[0])
        pan = c.data.xpos[c.model.body("tray").id].copy()
        piece.set_pose(pan + [0, 0, 0.009], [1, 0, 0, 0])
        piece.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        assert not c.outcome_settled()
        assert ("SplitCorrect", ("request_1",)) not in env.observe().abstract_atoms
        # Placed flat on the pan it tumbles about 2 cm and comes to rest after ~7 s.
        for _ in range(12):
            c.pause(phase="observe rest", seconds=1.0)
            if c.outcome_settled():
                break
        assert c.outcome_settled()
        assert ("SplitCorrect", ("request_1",)) in env.observe().abstract_atoms
    finally:
        env.close()


def _pick_and_pour(*, requested: int, params: tuple[float, ...]) -> dict:
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

    env = KitchenEnvironment(requested_count=requested)
    try:
        provider = KitchenSkillProvider(env=env)
        for name, p in (("PickPitcher", ()), (f"Pour{requested}", params)):
            g = provider.ground(name=name, request=requested, shrimp=0)
            env.take_action(action=provider.encode(ground_skill=g, params=np.array(p)))
            info = dict(env.last_info)
            assert info.get("error") is None, info.get("error")
        return info
    finally:
        env.close()


@pytest.mark.main_only
@pytest.mark.parametrize("requested", [1, 2, 3])
def test_canonical_pours_are_settled_successes(*, requested):
    # The closed-loop Pour's one parameter, its tilt ramp rate: 1 deg/s delivered Pour1
    # and Pour2 exactly on seeds 0-3 in the 2026-11-04 prototype.
    info = _pick_and_pour(requested=requested, params=(1.0,))
    assert info["mode"] == "closed_loop"
    assert info["delivered"] == requested
    assert info["settled"]
    assert info["success"]


@pytest.mark.main_only
def test_a_fast_ramp_overshoots_pour2_on_the_canonical_seed():
    # The trade-off the learned rate faces: at 2 deg/s the third dumpling is already
    # on its way out when the second leaves, and the cutoff cannot take it back.
    info = _pick_and_pour(requested=2, params=(2.0,))
    assert info["delivered"] == 3
    assert not info["success"]
