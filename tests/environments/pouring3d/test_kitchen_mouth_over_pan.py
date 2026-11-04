"""The closed-loop Pour holds the jug's mouth, not the hand, over the pan centre.

The hand grips the handle, ~0.1 m from the jug's axis, so a hand centred over the pan
puts the tilted jug's mouth ~45 mm past the pan centre at the ~91 deg stop tilt.
"""

import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

# Imported inside each test: collecting this file (as tests/test_main_only_marker.py does
# on every CI job) must not import KinDER's OpenGL stack.
pytestmark = [
    pytest.mark.xdist_group("kitchen_mouth_over_pan"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]

# How far the mouth may sit from the pan centre, beyond the hand's wiggle, at the stop.
MOUTH_TOLERANCE_M = 0.010


def test_the_mouth_offset_is_the_grasped_jug_carried_to_the_commanded_hand_rotation():
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

    hand = np.array([0.3, -0.2, 1.1])
    hand_rotation = Rotation.from_euler("xyz", [10, -20, 35], degrees=True)
    grasp = np.array([0.02, -0.095, -0.06])
    jug_in_hand = Rotation.from_euler("zy", [80, 5], degrees=True)
    jug = hand + hand_rotation.apply(grasp)
    jug_rotation = hand_rotation * jug_in_hand
    height = 0.14
    # Commanded = measured: the offset is the measured mouth, less the hand.
    same = KitchenController.mouth_offset(
        hand=hand,
        hand_rotation=hand_rotation,
        jug=jug,
        jug_rotation=jug_rotation,
        commanded_rotation=hand_rotation,
        height=height,
    )
    mouth = jug + jug_rotation.apply([0, 0, height])
    np.testing.assert_allclose(same, (mouth - hand)[:2], atol=1e-12)
    # A commanded rotation carries the measured grasp rigidly with it.
    commanded = Rotation.from_euler("y", 60, degrees=True) * hand_rotation
    turned = KitchenController.mouth_offset(
        hand=hand,
        hand_rotation=hand_rotation,
        jug=jug,
        jug_rotation=jug_rotation,
        commanded_rotation=commanded,
        height=height,
    )
    expected = commanded.apply(grasp + jug_in_hand.apply([0, 0, height]))
    np.testing.assert_allclose(turned, expected[:2], atol=1e-12)


def test_the_correction_is_absent_upright_and_complete_from_the_ramp_on():
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController

    approach = KitchenController.POUR_APPROACH_DEGREES
    assert KitchenController.mouth_correction_weight(angle=0.0) == 0.0
    assert KitchenController.mouth_correction_weight(angle=approach) == 1.0
    assert KitchenController.mouth_correction_weight(angle=95.0) == 1.0
    weights = [KitchenController.mouth_correction_weight(angle=a) for a in range(81)]
    assert all(b >= a for a, b in zip(weights, weights[1:], strict=False))


def _pour_recording_the_mouth(*, requested: int, rate: float) -> tuple[dict, dict]:
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
    from hitl_pmp.environments.pouring3d.kitchen.native import KitchenController
    from hitl_pmp.environments.pouring3d.kitchen.skill_provider import KitchenSkillProvider

    env = KitchenEnvironment(requested_count=requested)
    stop: dict = {}
    original = KitchenController.tracked_step

    def recording(self, *, center, angle, offset=0):  # noqa: PLR0917 -- patched method
        original(self, center=center, angle=angle, offset=offset)
        if not self.phase.startswith("closed-loop"):
            return
        jug, rotation = self.body_pose(name="pitcher")
        mouth = jug + rotation.apply([0, 0, self.scene.get_object("pitcher").height])
        pan = self.body_pose(name="tray")[0]
        hand, _ = self.hand_pose()
        # Discount the sideways wiggle the hand actually made (it lags the commanded
        # one): what is left is the offset the aim, not the wiggle, is responsible for.
        sideways = self.wiggle_direction()
        rest = mouth - pan - float(np.dot(hand - pan, sideways)) * sideways
        stop.update(angle=angle, miss=float(np.linalg.norm(rest[:2])))

    try:
        KitchenController.tracked_step = recording  # type: ignore[method-assign]
        provider = KitchenSkillProvider(env=env)
        for name, p in (("PickPitcher", ()), (f"Pour{requested}", (rate,))):
            g = provider.ground(name=name, request=requested, shrimp=0)
            env.take_action(action=provider.encode(ground_skill=g, params=np.array(p)))
            info = dict(env.last_info)
            assert info.get("error") is None, info.get("error")
        return info, stop
    finally:
        KitchenController.tracked_step = original  # type: ignore[method-assign]
        env.close()


@pytest.mark.main_only
@pytest.mark.parametrize("requested", [1, 2, 3])
def test_the_mouth_is_over_the_pan_centre_at_the_stop_and_the_pour_delivers(*, requested):
    info, stop = _pour_recording_the_mouth(requested=requested, rate=1.0)
    assert stop["angle"] > 85.0, stop
    assert stop["miss"] <= MOUTH_TOLERANCE_M, stop
    assert info["delivered"] == requested
    assert info["success"]
