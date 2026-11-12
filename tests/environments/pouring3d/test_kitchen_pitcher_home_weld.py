"""A parked jug stays home; picking releases it and placement anchors it again."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KinDER"),
    pytest.mark.xdist_group("kitchen_pitcher_home_weld"),
]


def _environment():
    from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment

    env = KitchenEnvironment(requested_count=1)
    env.reset_to_seed(seed=0, requested=1)
    return env


def _anchored(*, controller):
    return bool(controller.data.eq_active[controller.model.equality("pitcher_home_weld").id])


def test_a_parked_jug_resists_a_push_without_changing_its_pose_by_assignment():
    env = _environment()
    try:
        c = env.controller()
        before = c.data.qpos.copy()
        assert _anchored(controller=c)
        c.scene.weld_pitcher_home()
        np.testing.assert_array_equal(c.data.qpos, before)
        jug = c.model.body("pitcher").id
        c.data.xfrc_applied[jug] = [50, 50, 0, 0, 0, 10]
        c.pause(phase="push parked jug", seconds=1, grip=0)
        c.data.xfrc_applied[jug] = 0
        assert c.vessel_home(name="pitcher")
    finally:
        env.close()


def test_pick_pour_place_releases_and_reengages_the_home_weld():
    env = _environment()
    try:
        c = env.controller()
        env.take_action(action=np.array([0, 0, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"], env.last_info
        assert not _anchored(controller=c)
        assert c.body_pose(name="pitcher")[0][2] > c.home[2] + 0.15
        env.take_action(action=np.array([1, 0, 1, 1, 0, 0], dtype=float))
        assert not _anchored(controller=c)
        assert c.held == "pitcher"
        env.take_action(action=np.array([2, 0, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"], env.last_info
        assert _anchored(controller=c)
        assert c.vessel_home(name="pitcher")
        assert abs(c.body_pose(name="pitcher")[0][2] - c.home[2]) < 0.003
    finally:
        env.close()


def test_the_recorded_dumpling_drop_keeps_the_jug_home():
    import mujoco

    path = Path(__file__).parent / "fixtures/kitchen_before_drop_rotates_jug.json"
    state = json.loads(path.read_text())
    env = _environment()
    try:
        c = env.controller()
        c.data.qpos[:] = state["qpos"]
        c.data.qvel[:] = state["qvel"]
        c.data.time = state["time"][0]
        mujoco.mj_forward(c.model, c.data)
        c.last_q = c.data.qpos[c.arm_q].copy()
        c.base_target = c.data.qpos[c.base_q].copy()
        c.held = state["held"]
        c.grip = 1.0
        c.scene.attach_payload(name=c.held)
        # The old recording has no home weld; initialize it at this valid parked pose.
        if hasattr(c.scene, "weld_pitcher_home"):
            c.scene.weld_pitcher_home()
        assert c.vessel_home(name="pitcher")
        env.take_action(action=np.array([5, 1, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"], env.last_info
        assert c.vessel_home(name="pitcher")
        assert c.scene.cube_locations() == ["pitcher", "pitcher", "tray"]
    finally:
        env.close()


def test_a_successful_human_reset_reanchors_a_held_jug():
    env = _environment()
    try:
        c = env.controller()
        env.take_action(action=np.array([0, 0, 1, 0, 0, 0], dtype=float))
        assert env.last_info["success"], env.last_info
        env.reset_movables()
        assert c.held is None
        assert c.vessel_home(name="pitcher")
        assert _anchored(controller=c)
    finally:
        env.close()


def test_anchoring_cannot_hide_a_hovering_or_misaligned_placement():
    import mujoco
    from scipy.spatial.transform import Rotation

    env = _environment()
    try:
        c = env.controller()
        c.scene.release_pitcher_home()
        home = c.data.qpos[c.free_qpos(name="pitcher")].copy()
        for height, yaw in [(0.01, 0), (0, 10)]:
            pose = home.copy()
            pose[2] += height
            pose[3:] = Rotation.from_euler("z", yaw, degrees=True).as_quat()[[3, 0, 1, 2]]
            c.data.qpos[c.free_qpos(name="pitcher")] = pose
            mujoco.mj_forward(c.model, c.data)
            with pytest.raises(ValueError, match="has not landed at home"):
                c.anchor_parked_pitcher()
            assert not _anchored(controller=c)
            np.testing.assert_array_equal(c.data.qpos[c.free_qpos(name="pitcher")], pose)
    finally:
        env.close()
