"""Compact carry selection must yield an endpoint eligible for base transport."""

import json
from pathlib import Path

import mujoco
import numpy as np
from pybullet_helpers.geometry import Pose, multiply_poses
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.physical.types import KitchenScene
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_recorded_overhead_grasp_skips_tilted_home_for_checked_upright_goal() -> None:
    fixture = json.loads(Path(__file__).with_name("overhead_carry_fixture.json").read_text())
    session = SweepSimpleSession(seed=0)
    primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0)
    try:
        state = session.state.copy()
        for name, values in fixture["state"].items():
            obj = state.get_object_from_name(name)
            for key, value in zip(state.type_features[obj.type], values, strict=True):
                state.set(obj, key, value)
        session.env.unwrapped._object_centric_env.set_state(state)
        session._state = state
        session.mj_data.qpos[:] = fixture["qpos"]
        session.mj_data.qvel[:] = 0
        mujoco.mj_forward(session.mj_model, session.mj_data)
        scene = primitive.scene
        scene.sync()
        held = multiply_poses(
            scene.ee_now().invert(),
            Pose(tuple(session.position(name="wiper_0")), session.quaternion(name="wiper_0")),
        )
        home_tool = multiply_poses(scene.fk(arm=KitchenScene.HOME), held)
        assert np.arccos(Rotation.from_quat(home_tool.orientation).as_matrix()[2, 2]) > 1.1
        before = session.mj_data.qpos.copy()
        goal = primitive.wiper_stow_goal()
        assert not np.allclose(goal, KitchenScene.HOME)
        tool = multiply_poses(scene.fk(arm=goal), held)
        cosine = Rotation.from_quat(tool.orientation).as_matrix()[2, 2]
        assert np.arccos(np.clip(cosine, -1, 1)) <= 1.1
        assert (
            scene.plan_arm(goal=goal, bodies=scene.bodies(), held=scene.wiper_body, held_tf=held)
            is not None
        )
        np.testing.assert_array_equal(before, session.mj_data.qpos)
        assert scene.max_tool_tilt == 1.1
    finally:
        primitive.scene._sim.close()
        session.close()
