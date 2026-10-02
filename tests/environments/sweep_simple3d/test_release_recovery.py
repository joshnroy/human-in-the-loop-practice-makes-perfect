"""Native recorded release contacts require monotonic separation and live checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives
from hitl_pmp.environments.sweep_simple3d.physical.motion import ExecutionError
from hitl_pmp.environments.sweep_simple3d.release_recovery import (
    ReleaseContacts,
    separate_released_tool,
)
from hitl_pmp.environments.sweep_simple3d.session import SweepSimpleSession


@pytest.mark.parametrize(
    "current,allowed",
    [
        ({(134, 149): -0.0001}, True),
        ({}, True),
        ({(134, 149): -0.0004}, False),
        ({(133, 170): -0.00001}, False),
    ],
)
def test_no_new_or_deeper_native_contact(*, current: dict, allowed: bool) -> None:
    assert ReleaseContacts.separating(previous={(134, 149): -0.0003}, current=current) is allowed


@pytest.mark.parametrize("inject_contact", [False, True])
def test_recorded_native_release_preflight_and_live_guard(
    *, inject_contact: bool, monkeypatch
) -> None:
    fixture = json.loads(Path(__file__).with_name("release_contact_fixture.json").read_text())
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
        mujoco.mj_forward(session.mj_model, session.mj_data)
        before = session.mj_data.qpos.copy()
        called = []

        def follow(*, path, grip, max_ticks, tick_guard):
            np.testing.assert_array_equal(before, session.mj_data.qpos)
            called.append(path)
            assert grip == 0 and max_ticks == 180
            if inject_contact:
                monkeypatch.setattr(
                    ReleaseContacts, "contacts", lambda *_, **__: {(133, 171): -0.001}
                )
                tick_guard()
            # No simulation: inability to clear the live starting contact is not success.
            return True

        object.__setattr__(primitive, "motion", SimpleNamespace(follow=follow))
        if inject_contact:
            with pytest.raises(ExecutionError, match="gained or deepened"):
                separate_released_tool(primitive=primitive)
        else:
            assert not separate_released_tool(primitive=primitive)
        assert len(called) == 1
        assert len(called[0]) == 4
        np.testing.assert_array_equal(before, session.mj_data.qpos)
    finally:
        primitive.scene._sim.close()
        session.close()
