"""A dumpling the kitchen gripper has taken stays in the hand until it is released.

Grasp quality is not what the Pouring experiment measures (Josh, 2026-11-05): a weld
from the hand to the piece holds it from the grasp to the release, so a dumpling pick
fails only by refusing to plan or not reaching the piece, never by slipping out.
"""

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.xdist_group("kitchen_grasp_weld"),
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
]

PICK_PITCHER = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
POUR1 = (1.0, 0.0, 1.0, 1.0, 0.0, 0.0)
PLACE_PITCHER_AT_HOME = (2.0, 0.0, 1.0, 0.0, 0.0, 0.0)
PICK_SHRIMP_FROM_TRAY = 3.0
DROP_SHRIMP_INTO_PITCHER = 5.0


class Scene:
    """A static-method container, never instantiated."""

    @staticmethod
    def kitchen():
        from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

        env = KitchenEnvironment(requested_count=1)
        env.reset_to_seed(seed=0, requested=1)
        return env

    @staticmethod
    def piece_in_hand(*, controller, name: str) -> np.ndarray:
        import mujoco

        mujoco.mj_forward(controller.model, controller.data)
        hand = controller.model.body(f"{controller.scene.robot_name}_bracelet_link").id
        piece = controller.model.body(name).id
        rotation = controller.data.xmat[hand].reshape(3, 3)
        return rotation.T @ (controller.data.xpos[piece] - controller.data.xpos[hand])


def test_an_attached_dumpling_follows_the_open_hand_and_falls_once_released():
    env = Scene.kitchen()
    c = env.controller()
    name = c.scene.cube_names[0]
    assert c.scene.attached_payload() is None
    c.scene.attach_payload(name=name)
    before = Scene.piece_in_hand(controller=c, name=name)
    c.held = name
    hand, rotation = c.hand_pose()
    c.move(position=hand + [0, 0, 0.10], rotation=rotation, seconds=2, grip=0, phase="test lift")
    assert np.linalg.norm(Scene.piece_in_hand(controller=c, name=name) - before) < 0.005
    height = c.body_pose(name=name)[0][2]
    c.held = None
    c.pause(phase="test release", seconds=1, grip=0)
    assert c.scene.attached_payload() is None
    assert c.body_pose(name=name)[0][2] < height - 0.02
    env.close()


def test_the_human_reset_and_a_new_episode_release_every_dumpling():
    env = Scene.kitchen()
    c = env.controller()
    c.scene.attach_payload(name=c.scene.cube_names[1])
    c.scene.respawn_payload_in_pitcher()
    assert c.scene.attached_payload() is None
    c.scene.attach_payload(name=c.scene.cube_names[1])
    env.reset_to_seed(seed=0, requested=1)
    assert env.controller().scene.attached_payload() is None
    env.close()


def test_a_poured_dumpling_is_picked_carried_and_dropped_without_slipping():
    env = Scene.kitchen()
    for action in (PICK_PITCHER, POUR1, PLACE_PITCHER_AT_HOME):
        env.take_action(action=np.array(action))
        assert env.last_info["success"], env.last_info
    c = env.controller()
    index = c.scene.cube_locations().index("tray")
    env.take_action(action=np.array([PICK_SHRIMP_FROM_TRAY, index, 1.0, 0, 0, 0]))
    assert env.last_info["success"], env.last_info
    assert c.scene.attached_payload() == c.scene.cube_names[index]
    env.take_action(action=np.array([DROP_SHRIMP_INTO_PITCHER, index, 1.0, 0, 0, 0]))
    assert env.last_info["success"], env.last_info
    assert c.scene.attached_payload() is None
    assert c.scene.cube_locations()[index] == "pitcher"
    env.close()
