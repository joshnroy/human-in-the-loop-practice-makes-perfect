"""`--evaluate-only`: the untrained policy's one measurement -- what a step-0
measurement records before practice cycle 0 -- and nothing else: no practice, no
learning update, no checkpoint, no restore prompt."""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, PREVIOUS_CHECKPOINT_FILENAME

pytestmark = pytest.mark.xdist_group("tossing3d_real_evaluate_only")

BASE = (
    *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
    *("--robot", "fake", "--seed", "0", "--measure-every-cycles", "1"),
)


@pytest.fixture(autouse=True)
def room(*, monkeypatch: pytest.MonkeyPatch) -> FakeTossingRobot:
    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    return robot


def _events(*, output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize(
    "budget", [(), ("--practice-step-budget", "85000")], ids=["no-budget", "budget-ignored"]
)
def test_evaluate_only_records_one_step_zero_measurement_and_nothing_else(
    *, budget: tuple[str, ...], tmp_path: Path
) -> None:
    output = tmp_path / "baseline"
    Cli.main(
        argv=[
            *BASE,
            *budget,
            *("--num-test-tasks", "5", "--evaluate-only", "--output-dir", str(output)),
        ]
    )
    measurements = json.loads((output / "measurements.json").read_text())
    assert len(measurements) == 1
    (record,) = measurements
    assert record["index"] == 0 and record["practice_steps"] == 0
    assert record["learning_updates"] == 0 and record["robot_invocations"] == 0
    assert "after_cycle" not in record
    (evaluated,) = json.loads((output / "step_evaluations.json").read_text())
    assert evaluated["complete"] and evaluated["num_total"] == 5
    assert evaluated["policy_sha256"] == record["policy_sha256"]
    assert (output / "evaluations" / "0000" / "results.json").exists()
    assert not (output / "evaluations" / "0001").exists()
    kinds = [event["event"] for event in _events(output=output)]
    for absent in ("session_start", "robot", "human", "stop", "learning_update", "checkpoint"):
        assert absent not in kinds
    assert "practice_scene_restored" not in kinds
    assert kinds.count("measurement") == 1
    assert not (output / CHECKPOINT_FILENAME).exists()
    assert not (output / PREVIOUS_CHECKPOINT_FILENAME).exists()
    for name in ("robot_session.log", "config_snapshot.json"):
        assert (output / name).exists()
    marker = json.loads((output / "evaluate_only.json").read_text())
    assert marker["evaluate_only"] is True
    assert json.loads((output / "step_protocol.json").read_text())["evaluate_only"] is True
    assert json.loads((output / "status.json").read_text())["phase"] == "complete"


def test_evaluate_only_measures_the_policy_a_fresh_run_measures_at_step_zero(
    *, tmp_path: Path
) -> None:
    baseline, run = tmp_path / "baseline", tmp_path / "run"
    common = (*BASE, "--num-test-tasks", "1")
    Cli.main(argv=[*common, "--evaluate-only", "--output-dir", str(baseline)])
    Cli.main(
        argv=[
            *common,
            *("--initial-measurement", "--practice-step-budget", "1500"),
            *("--max-steps-per-interaction", "4", "--output-dir", str(run)),
        ]
    )
    (only,) = json.loads((baseline / "measurements.json").read_text())
    first = json.loads((run / "measurements.json").read_text())[0]
    assert first["practice_steps"] == 0 and first["learning_updates"] == 0
    assert only["policy_sha256"] == first["policy_sha256"]
    assert (baseline / "snapshots" / "0000.pickle").read_bytes() == (
        run / "snapshots" / "0000.pickle"
    ).read_bytes()


def test_evaluate_only_refuses_resume(*, tmp_path: Path) -> None:
    output = tmp_path / "baseline"
    with pytest.raises((ValueError, SystemExit)):
        Cli.main(
            argv=[
                *BASE,
                *("--num-test-tasks", "1", "--evaluate-only", "--output-dir", str(output)),
                *("--resume", str(output)),
            ]
        )
    assert not output.exists()
