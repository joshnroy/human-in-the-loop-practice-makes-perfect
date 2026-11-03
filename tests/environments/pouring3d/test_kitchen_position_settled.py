"""Kitchen pours are judged settled by position; legacy pouring keeps KinDER's rule."""

import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import HUMAN_RESET, KitchenEnvironment
from hitl_pmp.environments.pouring3d.kitchen.native import KitchenScene
from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

pytestmark = pytest.mark.xdist_group("kitchen_position_settled")


def test_only_the_kitchen_scene_overrides_kinders_rule():
    from kinder.envs.dynamic3d.pitcher_pour3d import ObjectCentricPitcherPour3DEnv

    from hitl_pmp.environments.pouring3d.five_frozen import FiveFrozenShrimpScene
    from hitl_pmp.environments.pouring3d.potsticker_native import PotstickerScene
    from hitl_pmp.environments.pouring3d.practice.native import PracticeThreeScene

    kinder_rule = ObjectCentricPitcherPour3DEnv.cubes_settled
    for legacy in (PotstickerScene, PracticeThreeScene, FiveFrozenShrimpScene):
        assert legacy.cubes_settled is kinder_rule
    assert KitchenScene.cubes_settled is not kinder_rule


def test_episode_start_and_human_reset_wait_until_rest_is_observed():
    import mujoco

    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        # Construction holds still until a full window of rest has been recorded.
        assert c.scene.cubes_settled()
        assert c.ready()
        atoms = env.observe().abstract_atoms
        assert {("Settled", (n,)) for n in c.scene.cube_names} <= atoms
        # A teleport without stepping is movement the history has not seen settle.
        piece = c.scene.get_object(c.scene.cube_names[0])
        pan = c.data.xpos[c.model.body("tray").id].copy()
        piece.set_pose(pan + [0, 0, 0.009], [1, 0, 0, 0])
        piece.set_velocity([0.0] * 3, [0.0] * 3)
        mujoco.mj_forward(c.model, c.data)
        assert not c.scene.cubes_settled()
        assert ("Settled", (c.scene.cube_names[0],)) not in env.observe().abstract_atoms
        before = c.data.time
        assert env.reset_movables(destination=HUMAN_RESET)
        assert c.data.time - before >= c.scene.payload_rest.window_seconds
        assert c.scene.cubes_settled()
    finally:
        env.close()


def _pick_and_pour(*, requested: int, params: tuple[float, float, float]) -> dict:
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
@pytest.mark.parametrize(
    ("requested", "params"),
    [
        # Before: delivered 2, settled False (one piece rocking on one contact point).
        (2, (88.37, 0.06, 41.4)),
        (1, (84.64, 0.0841, 31.2)),
        (3, (90.33, 0.0092, 16.6)),
    ],
)
def test_canonical_pours_are_settled_successes(*, requested, params):
    info = _pick_and_pour(requested=requested, params=params)
    assert info["delivered"] == requested
    assert info["settled"]
    assert info["success"]
