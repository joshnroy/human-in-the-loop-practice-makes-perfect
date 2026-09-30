"""Native side-counter geometry omitted by the upstream 2D planner still blocks routes."""
import json
from pathlib import Path

import mujoco
import numpy as np

from hitl_pmp.environments.sweep_simple3d.native_chassis import NativeChassisClearance
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


def test_recorded_side_handles_rejected_without_live_or_model_mutation() -> None:
    fixture = json.loads(Path(__file__).with_name("chassis_route_fixture.json").read_text())
    session = SweepSimpleSession(seed=0)
    try:
        session.mj_data.qpos[:] = fixture["qpos"]
        mujoco.mj_forward(session.mj_model, session.mj_data)
        helper = NativeChassisClearance(model=session.mj_model, live_data=session.mj_data)
        qpos, qvel = session.mj_data.qpos.copy(), session.mj_data.qvel.copy()
        ctrl = session.mj_data.ctrl.copy()
        masks = session.mj_model.geom_contype.copy(), session.mj_model.geom_conaffinity.copy()
        flags = session.mj_model.opt.enableflags, session.mj_model.opt.disableflags
        assert not helper.contacts(base=fixture["safe_base"], held_wiper=True)
        blocked = helper.contacts(base=fixture["blocked_base"], held_wiper=True)
        assert {c["body"] for c in blocked} >= {
            "kitchen_left_side_drawer_s0c1_handle", "kitchen_left_side_drawer_s1c1_handle"}
        assert min(c["distance"] for c in blocked) < -.02
        # Safe endpoints cannot bypass a blocked intermediate portion.
        rejected = helper.first_route_rejection(
            path=[fixture["safe_base"], fixture["blocked_base"], fixture["safe_base"]],
            held_wiper=True,
        )
        assert rejected is not None and rejected["segment"] == 1
        assert 0. < rejected["fraction"] <= 1.
        assert helper.first_route_rejection(path=[fixture["safe_base"]], held_wiper=True) is None
        np.testing.assert_array_equal(session.mj_data.qpos, qpos)
        np.testing.assert_array_equal(session.mj_data.qvel, qvel)
        np.testing.assert_array_equal(session.mj_data.ctrl, ctrl)
        np.testing.assert_array_equal(session.mj_model.geom_contype, masks[0])
        np.testing.assert_array_equal(session.mj_model.geom_conaffinity, masks[1])
        assert (session.mj_model.opt.enableflags, session.mj_model.opt.disableflags) == flags
        # Current cube geometry is copied on every query, not frozen at construction.
        joint = mujoco.mj_name2id(session.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "cube_0_joint")
        assert joint >= 0
        address = session.mj_model.jnt_qposadr[joint]
        # A cube elevated into the chassis is a genuine native collision;
        # a ground cube beneath the chassis need not touch its raised shell.
        session.mj_data.qpos[address:address + 3] = [*fixture["safe_base"][:2], .233]
        assert "cube_0" in {c["body"] for c in helper.contacts(
            base=fixture["safe_base"], held_wiper=True)}
    finally:
        session.close()
