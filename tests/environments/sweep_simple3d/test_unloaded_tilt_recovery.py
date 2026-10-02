"""Tilt-only recovery remains bounded and requires real measured improvement."""

from types import SimpleNamespace

import pytest
from scipy.spatial.transform import Rotation

from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError
from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives


def recovery_fixture(*, initial: float, changes: list[float], loaded: bool = False):
    state = {"tilt": initial, "calls": 0}
    records = []

    def level(*, bodies):
        assert bodies == {42}
        state["tilt"] += changes[min(state["calls"], len(changes) - 1)]
        state["calls"] += 1
        return True

    primitive = SimpleNamespace(
        session=SimpleNamespace(
            ticks=99,
            quaternion=lambda **_: Rotation.from_euler("x", state["tilt"]).as_quat(),
            _write=lambda **kwargs: records.append(kwargs["record"]),
        ),
        scene=SimpleNamespace(max_tool_tilt=1.1),
        wiper_loaded_by_cube=lambda: loaded,
        level_blade=level,
    )
    return primitive, state, records


def test_checked_leveling_restores_margin_with_actual_progress() -> None:
    primitive, state, records = recovery_fixture(initial=1.027, changes=[-0.08])
    FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == 1
    assert state["tilt"] < 0.95
    assert len(records) == 1
    assert all(r["measured_progress"] == pytest.approx(0.08) for r in records)
    assert primitive.scene.max_tool_tilt == 1.1


def test_joint_success_without_tool_response_remains_failure() -> None:
    primitive, state, records = recovery_fixture(initial=1.027, changes=[0])
    with pytest.raises(ExecutionError, match="no measured tilt progress"):
        FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == 1
    assert records[0]["measured_progress"] == 0


def test_four_turn_budget_does_not_mask_incomplete_recovery() -> None:
    primitive, state, records = recovery_fixture(initial=1.089, changes=[-0.02])
    with pytest.raises(ExecutionError, match="four-turn"):
        FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == len(records) == 4
    assert state["tilt"] > 0.95


def test_loaded_blade_never_attempts_leveling() -> None:
    primitive, state, _ = recovery_fixture(initial=1.027, changes=[-0.08], loaded=True)
    with pytest.raises(ExecutionError, match="unloaded blade"):
        FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == 0


def test_existing_safe_margin_needs_no_leveling() -> None:
    primitive, state, records = recovery_fixture(initial=0.88, changes=[-0.08])
    FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == 0
    assert records == []


def test_recorded_reachable_margin_avoids_unreachable_extra_turn() -> None:
    primitive, state, _ = recovery_fixture(initial=1.011547, changes=[-0.068057])
    FloorPrimitives.restore_unloaded_tilt_margin(primitive, bodies={42})
    assert state["calls"] == 1
    assert state["tilt"] == pytest.approx(0.94349)
    assert state["tilt"] < primitive.scene.max_tool_tilt - 0.15
