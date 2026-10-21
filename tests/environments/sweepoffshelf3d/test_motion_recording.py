"""`--record-motion`: per-control-step ticks, with the arm's joints, in the shelf's run
recording -- written for the scopes asked for, and drawn back as full motion."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend
from hitl_pmp.environments.sweepoffshelf3d.cli import SweepOffShelf3DCli
from hitl_pmp.environments.sweepoffshelf3d.environment import SKILL_IDS
from hitl_pmp.environments.sweepoffshelf3d.recording import SweepOffShelf3DReplayRenderer
from hitl_pmp.environments.sweepoffshelf3d.step_protocol import SweepOffShelf3DStepProtocol
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import ActionEntry, Phase, TickEntry

_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="needs KINDER"
)
pytestmark = pytest.mark.xdist_group("sweepoffshelf3d_test_motion_recording")


def _args(*, tmp_path: Path, record_motion: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    SweepOffShelf3DCli.add_arguments(parser=parser)
    args = parser.parse_args(["--no-scene-bg", "--record-motion", record_motion])
    args.seed = 0
    args.method = "ees"
    args.output_dir = str(tmp_path)
    return args


def test_motion_recording_is_off_by_default(*, tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    SweepOffShelf3DCli.add_arguments(parser=parser)
    assert parser.parse_args([]).record_motion == "off"
    with pytest.raises(SystemExit):
        parser.parse_args(["--record-motion", "sometimes"])


@pytest.mark.parametrize(
    ("record_motion", "practice", "evaluation"),
    [("off", False, False), ("evaluation", False, True), ("all", True, True)],
)
def test_each_scope_records_motion_only_where_asked(
    *, tmp_path: Path, record_motion: str, practice: bool, evaluation: bool
) -> None:
    args = _args(tmp_path=tmp_path, record_motion=record_motion)
    built = {
        "practice": SweepOffShelf3DStepProtocol.build_practice_problem(args=args),
        "evaluation": SweepOffShelf3DStepProtocol.build_evaluation_problem(args=args),
    }
    assert built["practice"].env.motion_recording is practice
    assert built["evaluation"].env.motion_recording is evaluation


def test_the_stride_follows_the_replay_renderers_fps() -> None:
    fps = SweepOffShelf3DReplayRenderer.MOTION_FPS
    stride = SweepOffShelf3DReplayRenderer.motion_stride()
    assert stride == max(1, round(SweepOffShelf3DBackend.CONTROL_FREQUENCY / fps))
    # Played one tick per frame, a recording at that stride runs at real time.
    assert SweepOffShelf3DBackend.CONTROL_FREQUENCY / stride == pytest.approx(fps)


def test_the_cycle_runner_refuses_motion_recording(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path, record_motion="all")
    with pytest.raises(ValueError, match="--record-motion"):
        SweepOffShelf3DCli.run_method(
            args=args,
            method_factory=lambda context: None,
            num_cycles=1,
            max_steps_per_interaction=1,
        )


def _record_one_skill(*, tmp_path: Path, record_motion: str) -> tuple[Path, int]:
    args = _args(tmp_path=tmp_path, record_motion=record_motion)
    problem = SweepOffShelf3DStepProtocol.build_evaluation_problem(args=args)
    path = tmp_path / "recording.jsonl"
    try:
        writer = SweepOffShelf3DStepProtocol.attach_recording(
            args=args, problem=problem, output_path=path
        )
        problem.hard_reset()
        live = problem.env.backend().core()._robot_env.control_frequency  # noqa: SLF001
        assert live == SweepOffShelf3DBackend.CONTROL_FREQUENCY
        writer.begin_segment(phase=Phase.EVALUATION, checkpoint=0, task_index=0)
        problem.take_action(action=np.array([float(SKILL_IDS["PickWiper"]), 0.0, 0.0, 0.0]))
        steps = problem.env.last_controller_steps()
        writer.robot_action(label="PickWiper", start_step=0, end_step=steps, success=True)
        writer.end_segment(success=False, outcome="step_budget", steps=steps)
        writer.close()
    finally:
        problem.env.close()
    return path, steps


@_NEEDS_KINDER
def test_motion_ticks_carry_the_arm_at_the_stride(*, tmp_path: Path) -> None:
    path, steps = _record_one_skill(tmp_path=tmp_path, record_motion="evaluation")
    header = RunRecordReader.header(path=path)
    motion = header.scene["motion"]
    stride = SweepOffShelf3DReplayRenderer.motion_stride()
    assert motion["stride"] == stride
    assert motion["fps"] == SweepOffShelf3DReplayRenderer.MOTION_FPS
    assert motion["robot_joints"] == list(SweepOffShelf3DBackend.ROBOT_JOINTS)
    entries = list(RunRecordReader.entries(path=path))
    (action,) = [e for e in entries if isinstance(e, ActionEntry)]
    ticks = [e for e in entries if isinstance(e, TickEntry) and e.seq == action.seq]
    assert steps > 10
    # Every stride-th control step, plus the settled end state if it differs.
    assert steps // stride <= len(ticks) <= steps // stride + 1
    assert action.num_ticks == len(ticks)
    # The action line is the same one a recording without motion writes.
    assert (action.skill, action.objects, action.params) == ("PickWiper", [], [])
    joints = [np.asarray(t.state["robot_joints"]) for t in ticks]
    assert all(j.shape == (len(SweepOffShelf3DBackend.ROBOT_JOINTS),) for j in joints)
    arm = np.array([j[3:10] for j in joints])
    assert np.ptp(arm, axis=0).max() > 0.05, "the arm never moved"
    # Each movable also carries its full pose after its usual features.
    assert len(ticks[0].state["cube_0"]) == 3 + 7
    assert len(ticks[0].state["wiper_0"]) == 4 + 7
    assert len(ticks[0].state["box_0"]) == 5 + 7


@_NEEDS_KINDER
def test_without_motion_the_recording_keeps_one_state_per_action(*, tmp_path: Path) -> None:
    path, _ = _record_one_skill(tmp_path=tmp_path, record_motion="off")
    assert "motion" not in RunRecordReader.header(path=path).scene
    entries = list(RunRecordReader.entries(path=path))
    (action,) = [e for e in entries if isinstance(e, ActionEntry)]
    assert action.num_ticks <= 1
    assert all("robot_joints" not in e.state for e in entries if isinstance(e, TickEntry))


@_NEEDS_KINDER
def test_a_recorded_tick_restores_and_reads_back(*, tmp_path: Path) -> None:
    """Round trip: restore recorded mid-skill ticks into the drawing scene, read them back."""
    path, _ = _record_one_skill(tmp_path=tmp_path, record_motion="evaluation")
    header = RunRecordReader.header(path=path)
    ticks = [e for e in RunRecordReader.entries(path=path) if isinstance(e, TickEntry)]
    renderer = SweepOffShelf3DReplayRenderer(scene=header.scene, scene_bg=False)
    try:
        assert not renderer.is_storyboard()
        assert renderer.fps() == SweepOffShelf3DReplayRenderer.MOTION_FPS
        for tick in (ticks[len(ticks) // 2], ticks[-1]):
            frame = renderer.show(state=tick.state)
            assert frame.ndim == 3 and frame.std() > 0
            back = renderer.current_state()
            assert back.keys() == tick.state.keys()
            for name, values in tick.state.items():
                assert np.allclose(back[name], values, atol=1e-6), name
    finally:
        renderer.close()


def test_a_recording_without_motion_still_plays_as_a_storyboard() -> None:
    renderer = SweepOffShelf3DReplayRenderer(scene={"canonical_seed": 0})
    assert renderer.is_storyboard()
    assert renderer.fps() == renderer.frames_per_second
    assert renderer.scene_bg is True  # the lab2 room, never the plain scene
