"""Reset contact policy preserves forward behavior and exact native blade support."""

from itertools import product
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_reset_height_uses_native_overlap_without_leaking_into_forward() -> None:
    primitive = SimpleNamespace(floor_clearance=.005,
                                cube_contact_ceiling=lambda **_: .01789)
    assert FloorPrimitives.contact_control_ceiling(
        primitive, cube="cube_0", region="blocks_init_region"
    ) == .01789
    assert FloorPrimitives.contact_control_ceiling(
        primitive, cube="cube_0", region="sweep_region"
    ) == .008
    assert primitive.floor_clearance == .005


def test_longaxis_candidate_is_local_to_broad_reset_and_keeps_fallback_order() -> None:
    original = [(0., False, None), (0., True, .4)]
    primitive = SimpleNamespace(floor_orientation_candidates=lambda **_: list(original))
    broad = FloorPrimitives.contact_orientation_candidates(
        primitive, tool_yaws=(0.,), region="blocks_init_region", narrow_contact=False
    )
    assert broad == [(0., True, .2, True), *[(*x, False) for x in original]]
    for region, narrow in (("sweep_region", False), ("sweep_region", True),
                           ("blocks_init_region", True)):
        assert FloorPrimitives.contact_orientation_candidates(
            primitive, tool_yaws=(0.,), region=region, narrow_contact=narrow
        ) == [(*x, False) for x in original]


def test_native_longaxis_pose_preserves_blade_support_and_live_state() -> None:
    session = SweepSimpleSession(seed=1)
    try:
        model, data = session.mj_model, session.mj_data
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wiper_0")
        blade = max((g for g in range(model.ngeom) if model.geom_bodyid[g] == body),
                    key=lambda g: model.geom_size[g, 0])
        before = data.qpos.copy()
        primitive = SimpleNamespace(session=session, scene=SimpleNamespace(max_tool_tilt=1.1),
                                    floor_clearance=.005)
        xy = np.array([1.49056, 1.04004])
        pose = FloorPrimitives.floor_tool_pose(
            primitive, xy=xy, yaw=0., tilt=.2, blade_axis=True
        )
        observed = data.xmat[body].reshape(3, 3)
        local_rotation = observed.T @ data.geom_xmat[blade].reshape(3, 3)
        target_rotation = Rotation.from_quat(pose.orientation).as_matrix()
        blade_rotation = target_rotation @ local_rotation
        np.testing.assert_allclose(blade_rotation, Rotation.from_euler("x", .2).as_matrix(),
                                   atol=1e-12)
        local_position = observed.T @ (data.geom_xpos[blade] - data.xpos[body])
        corners = np.array([local_position + local_rotation @ (model.geom_size[blade] * signs)
                            for signs in product((-1., 1.), repeat=3)])
        assert ((corners @ target_rotation.T) + pose.position)[:, 2].min() == pytest.approx(.005)
        np.testing.assert_array_equal(data.qpos, before)
        np.testing.assert_array_equal(pose.position[:2], xy)
    finally:
        session.close()
