"""A high corner away from the cube must not trigger a downward correction."""

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_local_contact_edge_excludes_unrelated_high_blade_corner() -> None:
    session = SweepSimpleSession(seed=0)
    primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
    try:
        model, data = session.mj_model, session.mj_data
        tool = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_0")
        tool_q = model.jnt_qposadr[model.body_jntadr[tool]]
        cube_q = model.jnt_qposadr[model.body_jntadr[cube]]
        pitch = -0.1
        data.qpos[tool_q : tool_q + 7] = [
            1.5,
            0.5,
            0.02,
            np.cos(pitch / 2),
            0.0,
            np.sin(pitch / 2),
            0.0,
        ]
        data.qpos[cube_q : cube_q + 7] = [1.37, 0.52, 0.01, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        global_high = primitive.blade_bottom_height()
        low_edge = primitive.blade_bottom_height(cube="cube_0")
        # The analytic bottom plane rises about 0.1 m per meter along blade x.
        assert 0.006 < low_edge < 0.01 < global_high
        data.qpos[cube_q] = 1.63
        mujoco.mj_forward(model, data)
        high_edge = primitive.blade_bottom_height(cube="cube_0")
        assert high_edge > 0.03
        assert high_edge > low_edge + 0.02
        target = primitive.floor_tool_pose(xy=np.array([0.5, 1.2]), yaw=1.1)
        data.qpos[tool_q : tool_q + 3] = target.position
        data.qpos[tool_q + 3 : tool_q + 7] = np.asarray(target.orientation)[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        assert abs(primitive.blade_minimum_height() - 0.001) < 1e-8
        matrix = data.xmat[tool].reshape(3, 3)
        assert abs(np.arccos(matrix[2, 2]) - abs(pitch)) < 1e-8
        assert abs(np.arctan2(matrix[1, 0], matrix[0, 0]) - 1.1) < 1e-8
    finally:
        primitive.scene._sim.close()
        session.close()


def test_contact_guard_stops_arm_motion_after_first_failed_physical_tick() -> None:
    session = SweepSimpleSession(seed=0)
    primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0.0)
    try:
        start = session.ticks
        target = session.arm().copy()
        target[0] += 0.1
        calls = []

        def guard() -> None:
            calls.append(session.ticks)
            raise ExecutionError("measured grasp loss")

        with pytest.raises(ExecutionError, match="measured grasp loss"):
            primitive.motion.follow(path=[target], grip=0.0, tick_guard=guard)
        assert calls == [start + 1]
        assert session.ticks == start + 1
    finally:
        primitive.scene._sim.close()
        session.close()
