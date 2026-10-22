"""`--record-motion all` is a pure observer: a short step-protocol run through the real
CLI takes the same actions and writes the same results with it on as with it off."""

import importlib.util
import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import (
    RECORDING_FILENAME,
    ActionEntry,
    SegmentEnd,
    TickEntry,
)
from scripts.sweepoffshelf3d_launch import Budget, ShelfLaunch

pytestmark = [
    pytest.mark.skipif(importlib.util.find_spec("kinder") is None, reason="needs KINDER"),
    pytest.mark.xdist_group("sweepoffshelf3d_motion_observer"),
]

SHORT = Budget(practice_steps=120, evaluation_interval=120, evaluation_control_steps=60, tasks=1)


def _run(*, output: Path, record_motion: str) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[6])["ees"]
    Cli.main(
        argv=[
            "--env", "sweepoffshelf3d",
            "--method", arm.method,
            "--seed", "0",
            "--output-dir", str(output),
            *ShelfLaunch.method_args(arm=arm, size=SHORT),
            "--no-scene-bg",
            "--record-motion", record_motion,
            "--skip-skill-benchmark-gate",
        ]
    )  # fmt: skip


# Differ between any two processes, recording or not: the wall clock, and the snapshot's
# pickle bytes, whose memo order follows string-set iteration and so PYTHONHASHSEED.
PER_PROCESS = {"elapsed_seconds", "policy_file_sha256"}


def _without_per_process(*, path: Path) -> list[dict]:
    return [
        {key: value for key, value in record.items() if key not in PER_PROCESS}
        for record in json.loads(path.read_text())
    ]


def _motion_ticks(*, path: Path) -> int:
    return sum(
        isinstance(entry, TickEntry) and "robot_joints" in entry.state
        for entry in RunRecordReader.entries(path=path)
    )


def _actions(*, path: Path) -> list[dict]:
    """Every action and segment outcome, less the tick count motion adds."""
    return [
        entry.model_dump(exclude={"num_ticks"})
        for entry in RunRecordReader.entries(path=path)
        if isinstance(entry, ActionEntry | SegmentEnd)
    ]


def test_recording_motion_changes_no_result(*, tmp_path: Path) -> None:
    _run(output=tmp_path / "off", record_motion="off")
    _run(output=tmp_path / "all", record_motion="all")
    off, on = tmp_path / "off", tmp_path / "all"
    assert (on / "stats.json").read_bytes() == (off / "stats.json").read_bytes()
    assert _without_per_process(path=on / "step_evaluations.json") == _without_per_process(
        path=off / "step_evaluations.json"
    )
    evaluations = sorted((on / "evaluations").glob(f"*/{RECORDING_FILENAME}"))
    assert evaluations
    assert _motion_ticks(path=on / RECORDING_FILENAME) > 0
    assert all(_motion_ticks(path=path) > 0 for path in evaluations)
    assert _motion_ticks(path=off / RECORDING_FILENAME) == 0
    # The recordings themselves say the same thing happened, ticks aside.
    for path in [on / RECORDING_FILENAME, *evaluations]:
        relative = path.relative_to(on)
        assert _actions(path=path) == _actions(path=off / relative), relative
