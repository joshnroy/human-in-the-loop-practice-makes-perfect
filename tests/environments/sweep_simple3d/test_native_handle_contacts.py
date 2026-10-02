"""Recorded contact classification, without stepping or physical-success claims."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession

FIXTURES = json.loads(Path(__file__).with_name("native_handle_contacts.json").read_text())


@pytest.mark.parametrize("record", FIXTURES)
def test_recorded_pad_contacts_distinguish_handle_wedging(*, record: dict) -> None:
    session = SweepSimpleSession(seed=0)
    try:
        model, data = session.mj_model, session.mj_data
        data.qpos[:] = record["qpos"]
        mujoco.mj_forward(model, data)
        primitive = FloorPrimitives.model_construct(session=session, scene=None, motion=None)
        handle, _ = primitive.wiper_handle_geometry()
        # Both recordings genuinely have bilateral pad contact. Empty output
        # must mean pad-only contact, rather than a dropped/untouched handle.
        touching = set()
        for contact in data.contact[: data.ncon]:
            if handle not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == handle else contact.geom1
            touching.add(mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])
            ))
        assert {"robot_left_pad", "robot_right_pad"} <= touching
        before = {name: getattr(data, name).copy() for name in
                  ("qpos", "qvel", "ctrl", "geom_xpos")}
        contacts_before = [(int(c.geom1), int(c.geom2), float(c.dist))
                           for c in data.contact[: data.ncon]]
        time_before = data.time
        flags = (int(model.opt.enableflags), int(model.opt.disableflags))
        assert primitive.handle_nonpad_gripper_contacts() == record["expected"]
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(data, name), value)
        assert [(int(c.geom1), int(c.geom2), float(c.dist))
                for c in data.contact[: data.ncon]] == contacts_before
        assert data.time == time_before
        assert (int(model.opt.enableflags), int(model.opt.disableflags)) == flags
    finally:
        session.close()
