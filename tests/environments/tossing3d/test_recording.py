"""Tossing3D as the first implementation of the run recording.

Two claims: the recording carries the same information the Tossing3D state log did,
and a recorded run played back from its states -- never re-simulated -- ends in
exactly the state the live run ended in.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.core.renderer.renderer import VideoStream
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.layout import Tossing3DLayout
from hitl_pmp.environments.tossing3d.recording import (
    LegacyTossing3DStateLog,
    Tossing3DRecording,
    Tossing3DReplayRenderer,
)
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader, StateLogWriter
from hitl_pmp.recording.replay.replayer import RecordingReplayer
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import (
    ActionEntry,
    Phase,
    RecordingHeader,
    TickEntry,
)
from hitl_pmp.recording.run_record.writer import RunRecordWriter

HAS_KINDER = (
    importlib.util.find_spec("kinder") is not None
    and importlib.util.find_spec("kinder_models") is not None
)
SCENE = {
    "layout": Tossing3DLayout.BARRIER,
    "variant": "o1",
    "scene_bg": True,
    "canonical_seed": 125,
    "seed": 0,
    "test_env_seed_offset": 10000,
}


def test_the_recording_carries_everything_the_state_log_did(*, tmp_path: Path) -> None:
    """Feed one sequence of environment reports to both writers; the legacy log,
    loaded through its adapter, must equal the new recording field for field."""
    header = StateLogHeader(**SCENE)
    legacy = StateLogWriter(output_path=tmp_path / "legacy.jsonl", header=header)
    recording = RunRecordWriter(
        output_path=tmp_path / "recording.jsonl",
        header=RecordingHeader(environment="tossing3d", scene=header.model_dump(mode="json")),
    )
    reports = [
        ("PickCube", ("robot", "cube_0"), (0.0, 0.0, 0.0, 0.0), {}, 2),
        (
            "MoveToTossLocationAndToss",
            ("robot", "bin_0", "cube_0"),
            (1.35, 140.0, 792.0),
            {"toss_direction_deg": 90.0},
            3,
        ),
    ]
    for name, objects, params, controller, ticks in reports:
        for sink in (legacy, recording):
            sink.record_skill(name=name, objects=objects, params=params, controller=controller)
            for i in range(ticks):
                sink.record_tick(
                    state={"cube_0": [float(i), 0.5, 0.25]},
                    diagnostics={"holding": {"contact_holding": bool(i)}},
                )
    legacy.close()
    recording.close()

    legacy_header, legacy_entries = LegacyTossing3DStateLog.load(path=tmp_path / "legacy.jsonl")
    assert legacy_header.scene == RunRecordReader.header(path=tmp_path / "recording.jsonl").scene

    def content(*, entries: list) -> list:
        return [
            (e.skill, e.objects, e.params, e.controller)
            if isinstance(e, ActionEntry)
            else (e.seq, e.state, e.diagnostics)
            for e in entries
            if isinstance(e, (ActionEntry, TickEntry))
        ]

    new_entries = list(RunRecordReader.entries(path=tmp_path / "recording.jsonl"))
    assert content(entries=list(legacy_entries)) == content(entries=new_entries)
    assert len(content(entries=new_entries)) == 2 + 2 + 3


@pytest.mark.skipif(not HAS_KINDER, reason="needs the KINDER simulator (tossing3d extra)")
def test_a_recorded_run_replays_to_exactly_its_recorded_final_state(*, tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    args = argparse.Namespace(**SCENE)
    problem = Tossing3DCli.build_problem(args=args)
    env = problem.env
    writer = Tossing3DRecording.writer(env=env, args=args, method="test", output_path=path)
    env.attach_state_log_writer(writer=writer)
    try:
        problem.hard_reset()
        task = problem.sample_train_task()
        problem.reset_to_task(task=task)
        writer.begin_segment(
            phase=Phase.PRACTICE, task_seed=Tossing3DRecording.task_seed(env=env, task=task)
        )
        pick = np.array([env.pick_cube_id, 0.0, 0.0, 0.0, 0.0])
        problem.take_action(action=pick)
        writer.robot_action(
            label="PickCube(robot, cube_0, barrier_0)",
            start_step=0,
            end_step=1,
            success=env.last_skill_error() is None,
            outcome=env.last_skill_error(),
        )
        problem.execute_movables_reset(destination="opposite_side")
        writer.human_action(
            skill="reset_cube_far",
            objects=("opposite_side",),
            start_step=1,
            end_step=2,
            success=True,
        )
        live_final = Tossing3DRecording.state(env=env)
    finally:
        writer.close()
        env.close()

    header = RunRecordReader.header(path=path)
    entries = list(RunRecordReader.entries(path=path))
    ticks = [e for e in entries if isinstance(e, TickEntry)]
    robot = next(e for e in entries if isinstance(e, ActionEntry) and e.actor == "robot")
    human = next(e for e in entries if isinstance(e, ActionEntry) and e.actor == "human")
    assert robot.num_ticks > 50, "the pick must actually have been recorded tick by tick"
    assert human.moved_objects, "a far reset moves the cube and bin"
    assert ticks[-1].state == live_final

    renderer = Tossing3DReplayRenderer(scene=header.scene)
    video = VideoStream(output_path=tmp_path / "replay.mp4", fps=renderer.fps())
    try:
        summary = RecordingReplayer(renderer=renderer).play(
            header=header, entries=iter(entries), video=video
        )
    finally:
        video.close()
        renderer.close()
    assert video.frames_written > 0
    # Read back from MuJoCo after the restore, at the recording's float32 precision.
    assert summary.final_state == live_final
    assert summary.final_state != ticks[0].state
    # No spurious settle tick: the live state a skill ends on is its last drained tick,
    # so no extra tick is appended that differs only in fixed-collider representation.
    robot_ticks = [t for t in ticks if t.seq == robot.seq]
    changed = {
        name
        for name, values in robot_ticks[-1].state.items()
        if robot_ticks[-2].state[name] != values
    }
    assert not changed or not all(name.startswith("collider:") for name in changed)
