"""The pouring kitchen opts in to simulation checkpoints, with evaluations spawned
beside practice, and a resume is checked against the geometry file's contents."""

import json
from types import SimpleNamespace

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli
from hitl_pmp.environments.pouring3d.kitchen.config import KitchenGeometry
from hitl_pmp.environments.pouring3d.kitchen.step_protocol import KitchenStepProtocol
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, StepProtocolEntry
from tests.practice_cost_fixtures import CostFixtures


def _argv(*, output, geometry=None, extra=()):
    return [
        "--env",
        "pouring3d_kitchen",
        "--method",
        "ees",
        "--practice-step-budget",
        "100",
        "--practice-cost-config",
        str(CostFixtures.BY_ENV["pouring3d_kitchen"]),
        *([] if geometry is None else ["--kitchen-geometry", str(geometry)]),
        "--output-dir",
        str(output),
        "--evaluation-workers",
        "9",
        "--evaluation-task-rng",
        "per-task",
        "--defer-rendering",
        *extra,
    ]


def _forbid_building(*, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("resume validation must not build a physical environment")

    monkeypatch.setattr(KitchenCli, "build_problem", staticmethod(forbidden))


def _entry(*, received):
    return {
        "method_cli": SimpleNamespace(run=lambda *, args, env_cli: received.append(args)),
        "env_cli": KitchenCli,
        "environment": KitchenStepProtocol,
    }


def test_checkpointing_is_accepted_with_evaluation_workers(*, tmp_path, monkeypatch):
    _forbid_building(monkeypatch=monkeypatch)
    received = []
    args = Cli.parse_args(argv=_argv(output=tmp_path / "run", extra=["--checkpoint-cycles"]))
    StepProtocolEntry.run(args=args, **_entry(received=received))
    assert received == [args]
    assert KitchenStepProtocol.evaluation_pool() is None  # spawned beside practice


def test_a_run_that_does_not_checkpoint_records_no_resolved_geometry(*, tmp_path, monkeypatch):
    _forbid_building(monkeypatch=monkeypatch)
    received = []
    args = Cli.parse_args(argv=_argv(output=tmp_path / "run"))
    before = dict(vars(args))
    StepProtocolEntry.run(args=args, **_entry(received=received))
    assert vars(received[0]) == before


def test_resume_accepts_the_same_geometry_and_refuses_a_changed_file(*, tmp_path, monkeypatch):
    _forbid_building(monkeypatch=monkeypatch)
    output = tmp_path / "run"
    output.mkdir()
    geometry = tmp_path / "geometry.json"
    geometry.write_text(KitchenGeometry().model_dump_json())
    argv = _argv(output=output, geometry=geometry, extra=["--checkpoint-cycles"])
    first = Cli.parse_args(argv=argv)
    KitchenStepProtocol.resolve_configuration(args=first)
    (output / "config_snapshot.json").write_text(
        json.dumps({"args": {k: str(v) for k, v in vars(first).items()}})
    )
    (output / CHECKPOINT_FILENAME).write_bytes(b"not opened until validation passes")
    received = []
    resume = [*argv, "--resume", str(output), "--stop-after-cycles", "3"]
    StepProtocolEntry.run(args=Cli.parse_args(argv=resume), **_entry(received=received))
    assert len(received) == 1
    geometry.write_text(KitchenGeometry(retreat_distance_m=0.5).model_dump_json())
    with pytest.raises(ValueError, match="kitchen_geometry_resolved"):
        StepProtocolEntry.run(args=Cli.parse_args(argv=resume), **_entry(received=received))
    assert len(received) == 1
