"""Native non-pad pickup guard rejects recorded obstruction, preserving valid paths."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from pybullet_helpers.geometry import Pose

from hitl_pmp.environments.sweep_drawer3d.primitives import Primitives
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.native_palm import NativePalmClearance
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

FIXTURES = json.loads(Path(__file__).with_name("native_pickup_nonpad_fixtures.json").read_text())


@pytest.mark.parametrize("name", ["blocked", "pad_contact"])
def test_native_obstruction_vs_allowed_pad_contact_without_mutation(*, name: str) -> None:
    session = SweepSimpleSession(seed=0)
    try:
        model, live = session.mj_model, session.mj_data
        live.qpos[:] = FIXTURES[name]["qpos"]
        mujoco.mj_forward(model, live)
        query = NativePalmClearance(model=model, live_data=live)
        before = {key: getattr(live, key).copy() for key in
                  ("qpos", "qvel", "ctrl", "xpos", "geom_xpos")}
        flags = (int(model.opt.enableflags), int(model.opt.disableflags))
        masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
        adr = query.tool_address
        pose = Pose(tuple(live.qpos[adr:adr + 3]),
                    tuple(live.qpos[adr + 3:adr + 7][[1, 2, 3, 0]]))
        rejected = query.nonpad_tool_contacts(
            joints=live.qpos[query.joint_addresses], tool_pose=pose
        )
        if name == "blocked":
            assert {c["robot_body"] for c in rejected} == {
                "robot_right_spring_link", "robot_left_spring_link"
            }
            assert all(c["tool_geom"] == 134 and c["distance"] < -.0006 for c in rejected)
        else:
            assert rejected == []
            # Physical pad contact exists but is deliberately permitted by this guard.
            assert any(c.geom1 == 133 and c.geom2 in (170, 171, 182, 183)
                       for c in live.contact[:live.ncon])
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(live, key), value)
        assert flags == (int(model.opt.enableflags), int(model.opt.disableflags))
        np.testing.assert_array_equal(model.geom_contype, masks[0])
        np.testing.assert_array_equal(model.geom_conaffinity, masks[1])
    finally:
        session.close()


@pytest.mark.parametrize("fixture", FIXTURES["nominal"], ids=lambda row: str(row["seed"]))
def test_nominal_lower_face_descent_stays_valid(*, fixture: dict) -> None:
    session = SweepSimpleSession(seed=fixture["seed"])
    primitive = FloorPrimitives.create(session=session, distance=.7, heading_offset=0)
    try:
        primitive.scene.sync(base=tuple(fixture["base"]))
        before = session.mj_data.qpos.copy()
        assert primitive.wiper_pickup_descent_clear(
            start=np.array(fixture["start"]), path=[np.array(q) for q in fixture["path"]]
        )
        np.testing.assert_array_equal(session.mj_data.qpos, before)
    finally:
        primitive.scene._sim.close()
        session.close()


def test_shared_default_retains_existing_pickup_behavior() -> None:
    assert Primitives.wiper_pickup_descent_clear(SimpleNamespace(), start=np.zeros(7), path=[])
