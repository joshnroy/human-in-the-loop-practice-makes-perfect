"""Recorded native palm separation and deliberate penetration regressions."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_simple3d.native_palm import NativePalmClearance
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

FIXTURES = json.loads(Path(__file__).with_name("native_palm_fixtures.json").read_text())


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda row: row["source"])
def test_native_palm_candidate_separation_and_penetration(fixture: dict) -> None:
    session = SweepSimpleSession(seed=0)
    try:
        model, live = session.mj_model, session.mj_data
        live.qpos[:] = fixture["qpos"]
        mujoco.mj_forward(model, live)
        query = NativePalmClearance(model, live)
        before = {name: getattr(live, name).copy() for name in
                  ("qpos", "qvel", "ctrl", "xpos", "geom_xpos")}
        flags = (int(model.opt.enableflags), int(model.opt.disableflags))
        masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
        adr = query.tool_address
        pose = Pose(tuple(live.qpos[adr:adr + 3]),
                    tuple(live.qpos[adr + 3:adr + 7][[1, 2, 3, 0]]))
        assert query.distance(joints=fixture["joints"], tool_pose=pose) == pytest.approx(1e-6)
        # Center the native handle inside the native palm, without changing live state.
        palm_center = live.geom_xpos[query.palm_geoms[0]].copy()
        overlap = Pose(tuple(palm_center), pose.orientation)
        assert query.distance(joints=fixture["joints"], tool_pose=overlap) < -0.02
        # Reuse must discard the previous hypothetical penetration.
        assert query.distance(joints=fixture["joints"], tool_pose=pose) == pytest.approx(1e-6)
        # A shared world translation preserves relative native geometry.
        shift = np.array([0.3, -0.2, 0.0])
        base = live.qpos[query.base_addresses].copy()
        base[:2] += shift[:2]
        moved = Pose(tuple(np.array(pose.position) + shift), pose.orientation)
        assert query.distance(joints=fixture["joints"], tool_pose=moved, base=base) == pytest.approx(1e-6)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(live, name), value)
        assert (int(model.opt.enableflags), int(model.opt.disableflags)) == flags
        np.testing.assert_array_equal(model.geom_contype, masks[0])
        np.testing.assert_array_equal(model.geom_conaffinity, masks[1])
        assert set(query.palm_geoms) == {161}  # Native palm only, not finger pads.
    finally:
        session.close()
