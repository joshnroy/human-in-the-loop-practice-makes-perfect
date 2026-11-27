"""After a Pour, the kitchen's joint-5 posture attractor is back where the Pour found it.

EES seed 2 of pouring-long-20261105-de96f85 overshot Pour2, which emptied the jug.
PlacePitcherAtHome then released the jug exactly at home (1.4 mm, -0.02 deg). The
attractor was still at the pour's `pour_posture`, so the IK of the scripted lift off
the handle turned the wrist. The open fingers spun the light, empty jug 17.8 deg, so
PitcherHome was lost and practice stranded.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("kitchen_posture_after_pour"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]

FIXTURE = Path(__file__).parent / "fixtures" / "kitchen_before_withdraw_from_released_jug.json"


def _jug(*, c) -> tuple[np.ndarray, float]:
    position, rotation = c.body_pose(name="pitcher")
    matrix = rotation.as_matrix()
    return position, float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0])))


def test_a_pour_restores_the_posture_attractor_and_the_place_keeps_the_jug_home():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

    env = KitchenEnvironment(requested_count=1)
    try:
        provider = KitchenSkillProvider(env=env)
        c = env.controller()

        def run(*, name: str, params: tuple[float, ...] = ()) -> dict:
            ground = provider.ground(name=name, request=1, shrimp=0)
            env.take_action(action=provider.encode(ground_skill=ground, params=np.array(params)))
            return dict(env.last_info)

        assert run(name="PickPitcher")["success"]
        before = (c.posture, c.posture_goal)
        pour = run(name="Pour1", params=(1.0,))
        assert pour.get("error") is None, pour.get("error")
        assert (c.posture, c.posture_goal) == before
        place = run(name="PlacePitcherAtHome")
        assert place.get("error") is None, place.get("error")
        assert place["success"]
        assert c.vessel_home(name="pitcher")
    finally:
        env.close()


@pytest.mark.parametrize(("posture", "spins"), [(0.0, False), (1.0, True)])
def test_the_lift_off_a_released_jug_spins_it_only_under_the_pours_attractor(
    *, posture: float, spins: bool
):
    import mujoco

    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    state = json.loads(FIXTURE.read_text())
    assert state["posture_rad"] == 1.0  # the pour's, which the run carried into the place
    env = KitchenEnvironment(requested_count=1)
    try:
        c = env.controller()
        # Preserve the historical unanchored positive control for wrist-induced spin.
        c.scene.release_pitcher_home()
        c.data.qpos[:] = state["qpos"]
        c.data.qvel[:] = state["qvel"]
        c.data.time = state["time"]
        mujoco.mj_forward(c.model, c.data)
        c.held, c.grip = None, 0
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        c.posture = c.posture_goal = posture
        c.choose_posture_turn()
        assert c.vessel_home(name="pitcher")
        start, yaw = _jug(c=c)
        worst = [0.0, 0.0]

        def watch(**_: object) -> None:
            position, now = _jug(c=c)
            worst[0] = max(worst[0], float(np.linalg.norm(position - start)))
            worst[1] = max(worst[1], abs(now - yaw))

        c.callback = watch
        # PlacePitcherAtHome's own lift off the handle after the release.
        hand, orientation = c.hand_pose()
        c.move(
            phase="withdraw gripper",
            position=hand + [0, 0, 0.25],
            rotation=orientation,
            seconds=3,
            grip=0,
        )
        if spins:
            assert worst[1] > 5.0
            assert not c.vessel_home(name="pitcher")
        else:
            assert worst[0] <= 0.002
            assert worst[1] <= 1.0
            assert c.vessel_home(name="pitcher")
    finally:
        env.close()
