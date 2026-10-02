"""Observed loaded tilt unloads before the unchanged planning limit."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_simple3d.controllers import ContactTiltLimit, FloorPrimitives


def guard_fixture(*, tilt: float, loaded: bool):
    records = []
    session = SimpleNamespace(
        ticks=19,
        quaternion=lambda **_: Rotation.from_euler("x", tilt).as_quat(),
        _write=lambda **kwargs: records.append(kwargs["record"]),
    )
    primitive = SimpleNamespace(
        session=session, scene=SimpleNamespace(max_tool_tilt=1.1),
        wiper_loaded_by_cube=lambda: loaded,
    )
    return primitive, records


def test_loaded_tilt_stops_before_hard_limit_and_records_actual_state() -> None:
    primitive, records = guard_fixture(tilt=1.01, loaded=True)
    with pytest.raises(ContactTiltLimit):
        FloorPrimitives.guard_loaded_tool_tilt(primitive, phase="contact drive")
    assert primitive.scene.max_tool_tilt == 1.1
    assert records[0]["tilt"] == pytest.approx(1.01)
    assert records[0]["unload_tilt"] == pytest.approx(1.0)
    assert not records[0]["already_over_planning_limit"]


def test_unloaded_pose_does_not_trigger_loaded_guard() -> None:
    primitive, records = guard_fixture(tilt=1.12, loaded=False)
    FloorPrimitives.guard_loaded_tool_tilt(primitive, phase="contact drive")
    assert records == []


def test_loaded_safe_pose_continues_and_over_limit_start_is_disclosed() -> None:
    primitive, records = guard_fixture(tilt=.99, loaded=True)
    FloorPrimitives.guard_loaded_tool_tilt(primitive, phase="contact correction")
    assert records == []
    primitive.session.quaternion = lambda **_: Rotation.from_euler("x", 1.128).as_quat()
    with pytest.raises(ContactTiltLimit):
        FloorPrimitives.guard_loaded_tool_tilt(primitive, phase="contact correction")
    assert records[-1]["already_over_planning_limit"]
    assert records[-1]["phase"] == "contact correction"


def test_yaw_does_not_count_as_tilt() -> None:
    primitive, records = guard_fixture(tilt=0, loaded=True)
    primitive.session.quaternion = lambda **_: Rotation.from_euler("z", np.pi).as_quat()
    FloorPrimitives.guard_loaded_tool_tilt(primitive, phase="contact drive")
    assert records == []
