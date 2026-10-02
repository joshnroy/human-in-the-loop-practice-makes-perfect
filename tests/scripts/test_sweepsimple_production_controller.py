"""Ordinary controller validation cannot silently inherit probe overrides."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[2] / "scripts/probe_sweepsimple_pick.py"
    spec = importlib.util.spec_from_file_location("simple_production_probe_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PickupProbe


def test_production_skips_every_controller_override(*, probe) -> None:
    from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives

    primitive = FloorPrimitives.model_construct(scene=SimpleNamespace(max_tool_tilt=0.83))
    fields = {
        name: getattr(primitive, name)
        for name in (
            "stand_ahead",
            "native_contact_guard",
            "retain_pickup_carry_pose",
            "floor_clearance",
            "diagnostic_grasp_standoff",
            "narrow_contact",
            "contact_step",
            "contact_stroke_length",
        )
    }
    methods = {
        name: getattr(FloorPrimitives, name)
        for name in (
            "broad_blade_center",
            "wiper_grasp_yaw",
            "wiper_grasp_offsets",
            "wiper_approach_angles",
            "wiper_handle_geometry",
            "wiper_grasp_point",
        )
    }
    # No diagnostic attributes exist: reading even one is an error.
    probe.configure_controller(
        primitive=primitive, args=SimpleNamespace(production_controller=True)
    )
    assert {name: getattr(primitive, name) for name in fields} == fields
    assert primitive.scene.max_tool_tilt == 0.83
    assert all(getattr(FloorPrimitives, name) is method for name, method in methods.items())


@pytest.mark.parametrize(
    "flag",
    [
        "--grasp-offset=-.09",
        "--grasp-insertion-offset=.028",
        "--grasp-approach-angle=1.2",
        "--grasp-height=0",
        "--grasp-mode=handle",
        "--grasp-yaw-offset=0",
        "--tilt-limit=1.1",
        "--stroke-length=.1",
        "--contact-step=.003",
        "--floor-clearance=.001",
        "--narrow-contact",
        "--center-selected-cube",
        "--stand-ahead",
        "--native-contact-guard",
        "--retain-pickup-carry-pose",
        "--grasp-face-seeded",
    ],
)
def test_explicit_diagnostic_options_are_rejected_even_at_default(*, probe, flag: str) -> None:
    with pytest.raises(SystemExit) as error:
        probe.validate_controller_mode(
            args=SimpleNamespace(production_controller=True),
            argv=[flag],
            parser=argparse.ArgumentParser(),
        )
    assert error.value.code == 2


def test_nominal_parameters_orchestration_and_readonly_logging_remain_allowed(*, probe) -> None:
    probe.validate_controller_mode(
        args=SimpleNamespace(production_controller=True),
        argv=[
            "--pick-distance=.7",
            "--sweep-distance",
            ".7",
            "--sweep-angle=0",
            "--finish-selected-first",
            "--forward-budget=10",
            "--log-live-grasp",
        ],
        parser=argparse.ArgumentParser(),
    )


def test_deeper_insertion_diagnostic_parses_without_starting_physics(
    *, probe, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--tag",
            "unused",
            "--grasp-insertion-offset",
            ".010",
            "--cube-order",
            "0",
            "0",
            "0",
            "0",
            "0",
        ],
    )
    with pytest.raises(SystemExit):
        probe.run()
    assert "--cube-order must contain every native cube" in capsys.readouterr().err


def test_face_seed_reads_native_cross_section_not_vertical_handle_axis(
    *, probe, monkeypatch
) -> None:
    import numpy as np

    from hitl_pmp.environments.sweep_simple3d.controllers import FloorPrimitives

    original = FloorPrimitives.wiper_grasp_yaw
    monkeypatch.setattr(FloorPrimitives, "wiper_grasp_yaw", original)
    args = SimpleNamespace(
        production_controller=False,
        tilt_limit=1.1,
        stroke_length=0.1,
        contact_step=0.003,
        floor_clearance=0.005,
        narrow_contact=False,
        center_selected_cube=False,
        stand_ahead=True,
        native_contact_guard=True,
        retain_pickup_carry_pose=True,
        grasp_insertion_offset=0.020,
        grasp_mode="handle",
        grasp_yaw_offset=0,
        grasp_face_seeded=True,
        grasp_height=0,
        grasp_approach_angle=None,
        grasp_offset=None,
    )
    fake = SimpleNamespace(scene=SimpleNamespace())
    probe.configure_controller(primitive=fake, args=args)
    angle = 0.6
    matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    fake.session = SimpleNamespace(mj_data=SimpleNamespace(geom_xmat=matrix.reshape(1, 9)))
    fake.wiper_handle_geometry = lambda: (0, 2)
    assert FloorPrimitives.wiper_grasp_yaw(fake, axis=np.array([0, 0, 1])) == pytest.approx(angle)


@pytest.mark.parametrize(
    "success,note", [(True, ""), (False, "Observed action preconditions do not hold")]
)
def test_initial_dispatch_uses_ordinary_action_result(*, probe, success, note) -> None:
    import numpy as np

    from hitl_pmp.environments.sweep_drawer3d.motion import ExecutionError

    calls = []
    session = SimpleNamespace(_steps=[])

    def take_action(*, action):
        calls.append(action)
        session._steps.append(SimpleNamespace(success=success, note=note))

    env = SimpleNamespace(
        ACTION_NAMES=["PickFloorWiper"], take_action=take_action, session=lambda: session
    )
    if success:
        assert (
            probe.dispatch_ordinary(env=env, name="PickFloorWiper", cube=-1, params=(0.7, 0.0))
            == ""
        )
    else:
        with pytest.raises(ExecutionError, match="preconditions"):
            probe.dispatch_ordinary(env=env, name="PickFloorWiper", cube=-1, params=(0.7, 0.0))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], [0, -1, 0.7, 0.0])
