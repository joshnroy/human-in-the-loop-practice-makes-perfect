"""With `--measure-every-cycles K`, every learning update leaves a deployment snapshot,
not only the updates an evaluation measures, so any cycle's policy can be evaluated
later. The measurement snapshots keep their names and contents."""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.environments.tossing3d.real.step_protocol import Tossing3DRealStepProtocol
from hitl_pmp.step_protocol import DeploymentSnapshot, PracticeCheckpoint, StepFiles

pytestmark = pytest.mark.xdist_group("tossing3d_real_snapshot_every_update")


def _argv(*, output: Path, resume: bool = False) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "1", "--seed", "0"),
        *("--practice-step-budget", "4500", "--max-steps-per-interaction", "4"),
        *("--measure-every-cycles", "2", "--checkpoint-cycles", "--output-dir", str(output)),
        *(["--resume", str(output)] if resume else []),
    ]


def _events(*, output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def _updates(*, output: Path) -> int:
    return sum(e["event"] == "learning_update" for e in _events(output=output))


def test_every_learning_update_leaves_a_snapshot_beside_the_measurement_ones(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
    output = tmp_path / "run"
    Cli.main(argv=_argv(output=output))
    updates = _updates(output=output)
    assert updates >= 3
    after = sorted((output / "snapshots").glob("after-update-*.pickle"))
    assert [p.name for p in after] == [
        f"after-update-{n:04d}.pickle" for n in range(1, updates + 1)
    ]
    args = argparse.Namespace(**json.loads((output / "config_snapshot.json").read_text())["args"])
    problem = Tossing3DRealStepProtocol.build_evaluation_problem(args=args)
    provider = Tossing3DRealStepProtocol.evaluation_skill_provider(problem=problem)
    for path in after:
        DeploymentSnapshot.restore(raw=path.read_bytes(), env=problem.env, provider=provider)

    # The measurement snapshots are untouched: same names, same indices, and each is
    # byte-for-byte the after-update snapshot of the update it measured.
    records = json.loads((output / "measurements.json").read_text())
    measured = sorted(p.name for p in (output / "snapshots").glob("[0-9]*.pickle"))
    assert measured == [f"{r['index']:04d}.pickle" for r in records]
    for record in records:
        if "after_cycle" in record:
            assert record["learning_updates"] % 2 == 0
            raw = (output / "snapshots" / f"{record['index']:04d}.pickle").read_bytes()
            n = record["learning_updates"]
            assert raw == (output / "snapshots" / f"after-update-{n:04d}.pickle").read_bytes()


def test_a_resume_leaves_one_snapshot_per_update_of_the_surviving_history(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = tmp_path / "alone"
    monkeypatch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
    Cli.main(argv=_argv(output=alone))
    expected = sorted(p.name for p in (alone / "snapshots").glob("after-update-*.pickle"))

    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    output = tmp_path / "resumed"
    original = StepFiles.event
    seen = {"updates": 0}

    # Ctrl-C right after the third update, before the checkpoint that would record it:
    # the resume redoes that cycle and must rewrite its snapshot, never keep the stale one.
    def interrupting(*, path: Path, **record: Any) -> None:
        original(path=path, **record)
        if record.get("event") == "learning_update":
            seen["updates"] += 1
            if seen["updates"] == 3:
                raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(StepFiles, "event", staticmethod(interrupting))
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))
    assert (output / "snapshots" / "after-update-0003.pickle").exists()
    header = PracticeCheckpoint.header(path=output / "checkpoint.pickle")
    assert header["learning_updates"] == 2
    Cli.main(argv=_argv(output=output, resume=True))
    names = sorted(p.name for p in (output / "snapshots").glob("after-update-*.pickle"))
    assert names == expected
