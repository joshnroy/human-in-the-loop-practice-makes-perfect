"""The dumpling pick closes on the dumpling's long sides, never on its two narrow ends.

Recorded 2026-11-04 (cycle 2 of a four-cycle kitchen run): a dumpling lying near the
skillet wall, where the gripper's whole open span does not fit. The pick turned its
grasp 90 degrees to the span that fitted best, so the pads closed on the dumpling's
tapered ends, 55 mm apart; the piece slipped out 1.6 s into the carry to the jug.
"""

import mujoco
import numpy as np
import pytest

pytest.importorskip("kinder")

from hitl_pmp.environments.pouring3d.kitchen.environment import KitchenEnvironment
from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES

pytestmark = pytest.mark.xdist_group("kitchen_wide_grasp")

DUMPLING = 1
PICK = np.array([SKILL_NAMES.index("PickShrimpFromTray"), DUMPLING, 1, 0, 0, 0], dtype=float)
DROP = np.array([SKILL_NAMES.index("DropShrimpIntoPitcher"), DUMPLING, 1, 0, 0, 0], dtype=float)
# The recorded pan and dumpling at rest before that pick (MuJoCo qpos: xyz, wxyz).
RECORDED_PAN = np.array([0.8542, 0.8422, 0.74])
RECORDED_DUMPLING = np.array([0.9097, 0.7973, 0.7618, -0.5525, -0.8282, -0.0788, -0.0506])
# Both pads on the long sides: the closing axis within 20 degrees of the dumpling's
# local y-z plane, i.e. at most sin(20 deg) along its 55 mm long axis (local x).
LONG_AXIS_LIMIT = float(np.sin(np.deg2rad(20.0)))


class Scene:
    @staticmethod
    def recorded_dumpling_near_the_wall() -> KitchenEnvironment:
        env = KitchenEnvironment(requested_count=1)
        c = env.controller()
        m, d = c.model, c.data
        joint = int(m.jnt_qposadr[m.body(c.scene.cube_names[DUMPLING]).jntadr[0]])
        pan = d.xpos[m.body("tray").id]
        d.qpos[joint : joint + 3] = RECORDED_DUMPLING[:3] - RECORDED_PAN + pan
        d.qpos[joint + 3 : joint + 7] = RECORDED_DUMPLING[3:]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        c.pause(phase="settle test dumpling", seconds=3, grip=0)
        env.current_state = env.observe()
        assert c.scene.cube_locations()[DUMPLING] == "tray"
        return env

    @staticmethod
    def long_axis_component(*, env: KitchenEnvironment) -> float:
        """How much of the pads' closing axis lies along the dumpling's long axis."""
        c = env.controller()
        m, d = c.model, c.data
        closing = d.xpos[m.body("robot_left_pad").id] - d.xpos[m.body("robot_right_pad").id]
        _, rotation = c.body_pose(name=c.scene.cube_names[DUMPLING])
        local = rotation.inv().apply(closing / np.linalg.norm(closing))
        return float(abs(local[0]))


def test_the_chosen_grasp_closes_across_the_long_sides():
    env = Scene.recorded_dumpling_near_the_wall()
    c = env.controller()
    name = c.scene.cube_names[DUMPLING]
    point, across = c.payload_grasp_geometry(name=name)
    yaw = c.payload_grasp_yaw(point=point, across=across, location="tray")
    _, rotation = c.body_pose(name=name)
    long_axis = rotation.apply([1.0, 0.0, 0.0])
    # The gripper closes along the grasp frame's yaw direction.
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    horizontal = long_axis[:2] / np.linalg.norm(long_axis[:2])
    assert abs(float(closing[:2] @ horizontal)) < LONG_AXIS_LIMIT
    env.close()


def test_the_recorded_pick_closes_on_the_long_sides_and_the_drop_lands_in_the_jug():
    env = Scene.recorded_dumpling_near_the_wall()
    c = env.controller()
    closing: list[float] = []
    sink = c.callback

    def first_descent(**kwargs):  # noqa: PLR0917 -- controller callback signature
        if sink is not None:
            sink(**kwargs)
        # The pads' axis as the hand starts down, before the fingers can touch the
        # dumpling. A descent along the long axis can knock it a quarter turn into a
        # long-side grasp by luck (it did in this replay), or not (it did not on record).
        if c.phase == "descend to shrimp body" and not closing:
            closing.append(Scene.long_axis_component(env=env))

    c.callback = first_descent
    env.take_action(action=PICK)
    assert env.last_info["success"] is True, env.last_info
    assert closing and closing[0] < LONG_AXIS_LIMIT, closing
    env.take_action(action=DROP)
    assert env.last_info["success"] is True, env.last_info
    assert c.scene.cube_locations()[DUMPLING] == "pitcher"
    env.close()
