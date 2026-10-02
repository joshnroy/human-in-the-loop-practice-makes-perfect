"""Tipped recovery appends native-face overhead frames without changing nominal search."""

import json
from pathlib import Path

import mujoco
import numpy as np

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.physical.primitives import Primitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_nominal_candidates_remain_exactly_unchanged() -> None:
    session = SweepSimpleSession(seed=0)
    primitive = FloorPrimitives.create(session=session, distance=0.7, heading_offset=0)
    try:
        handle, index = primitive.wiper_handle_geometry()
        axis = session.mj_data.geom_xmat[handle].reshape(3, 3)[:, index]
        expected = Primitives.wiper_grasp_orientations(primitive, axis=axis)
        actual = primitive.wiper_grasp_orientations(axis=axis)
        assert len(actual) == len(expected) == 24
        np.testing.assert_array_equal(actual, expected)
    finally:
        primitive.scene._sim.close()
        session.close()


def test_recorded_tipped_overhead_frame_and_native_descent_clear() -> None:
    fixture = json.loads(Path(__file__).with_name("tipped_pickup_fixture.json").read_text())
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
        session.mj_data.qpos[:] = fixture["native_qpos"]
        mujoco.mj_forward(session.mj_model, session.mj_data)
        handle, index = primitive.wiper_handle_geometry()
        axes = session.mj_data.geom_xmat[handle].reshape(3, 3)
        original = Primitives.wiper_grasp_orientations(primitive, axis=axes[:, index])
        actual = primitive.wiper_grasp_orientations(axis=axes[:, index])
        assert len(actual) == len(original) + 6
        np.testing.assert_array_equal(actual[: len(original)], original)
        closing, approach = actual[len(original)]
        np.testing.assert_allclose(approach, [0, 0, -1])
        np.testing.assert_allclose(closing[:2], axes[:2, 0] / np.linalg.norm(axes[:2, 0]))
        primitive.scene.sync(base=tuple(fixture["stance"]))
        before = session.mj_data.qpos.copy()
        assert primitive.wiper_pickup_descent_clear(
            start=np.array(fixture["hover_arm"]),
            path=[np.array(q) for q in fixture["down_arms"]],
        )
        np.testing.assert_array_equal(before, session.mj_data.qpos)
    finally:
        primitive.scene._sim.close()
        session.close()
