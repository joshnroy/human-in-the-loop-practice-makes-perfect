"""`--record-motion` on the rack: per-control-step ticks of every body that moves, and
`RackServingReplayRenderer` drawing them back in the lab2 room as full motion."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.racktoserving3d.cli import RackServingCli
from hitl_pmp.environments.racktoserving3d.recording import RackServingReplayRenderer
from hitl_pmp.environments.racktoserving3d.skills import SKILL_IDS
from hitl_pmp.environments.racktoserving3d.step_protocol import RackServingStepProtocol
from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend
from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import ActionEntry, Phase, RecordingHeader, TickEntry
from tests.practice_cost_fixtures import CostFixtures

_NEEDS_KINDER = pytest.mark.skipif(
    importlib.util.find_spec("kinder") is None, reason="needs KINDER"
)
pytestmark = pytest.mark.xdist_group("racktoserving3d_test_recording")

# Every body with a free joint in the rack scene: five dumplings, the scraper, the
# plate and the rack itself.
MOVABLES = ("cube_0", "cube_1", "cube_2", "cube_3", "cube_4", "wiper_0", "box_0", "rack_0")


def _args(*, tmp_path: Path, record_motion: str):
    return Cli.parse_args(
        argv=[
            "--env", "racktoserving3d", "--method", "ees",
            "--practice-step-budget", "5", "--practice-reset-policy", "never",
            *CostFixtures.argv(env="racktoserving3d"),
            "--defer-rendering", "--output-dir", str(tmp_path), "--seed", "0",
            "--record-motion", record_motion,
        ]
    )  # fmt: skip


def test_motion_recording_is_off_by_default(*, tmp_path: Path) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    RackServingCli.add_arguments(parser=parser)
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
    assert RackServingStepProtocol.build_practice_problem(args=args).env.motion_recording is (
        practice
    )
    assert RackServingStepProtocol.build_evaluation_problem(args=args).env.motion_recording is (
        evaluation
    )


def test_the_cycle_runner_refuses_motion_recording(*, tmp_path: Path) -> None:
    args = _args(tmp_path=tmp_path, record_motion="all")
    with pytest.raises(ValueError, match="--record-motion"):
        RackServingCli.run_method(
            args=args,
            method_factory=lambda context: None,
            num_cycles=1,
            max_steps_per_interaction=1,
        )


def test_the_renderer_draws_the_lab2_room_and_is_registered() -> None:
    from analysis.render_recording import RecordingVideo

    header = RecordingHeader(environment="racktoserving3d", scene={"canonical_seed": 0})
    renderer = RecordingVideo.default_renderer(header=header, realistic_background=True)
    assert isinstance(renderer, RackServingReplayRenderer)
    assert renderer.scene_bg is True
    # No motion in the header: a storyboard.
    assert renderer.is_storyboard()
    motion = {"motion": {"fps": 10, "stride": 1}}
    assert not RackServingReplayRenderer(scene=motion).is_storyboard()
    assert RackServingReplayRenderer(scene=motion).fps() == 10


def _record_one_skill(*, tmp_path: Path, record_motion: str) -> tuple[Path, int]:
    args = _args(tmp_path=tmp_path, record_motion=record_motion)
    problem = RackServingStepProtocol.build_evaluation_problem(args=args)
    path = tmp_path / "recording.jsonl"
    try:
        writer = RackServingStepProtocol.attach_recording(
            args=args, problem=problem, output_path=path
        )
        problem.hard_reset()
        writer.begin_segment(phase=Phase.EVALUATION, checkpoint=0, task_index=0)
        problem.take_action(action=np.array([float(SKILL_IDS["PickScraper"]), 0.0, 0.0, 0.0, 0.0]))
        steps = problem.env.last_controller_steps()
        writer.robot_action(
            label="PickScraper(robot, wiper_0)", start_step=0, end_step=steps, success=True
        )
        writer.end_segment(success=False, outcome="step_budget", steps=steps)
        writer.close()
    finally:
        problem.env.close()
    return path, steps


@pytest.fixture(name="motion_recording", scope="module")
def _motion_recording(*, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    return _record_one_skill(
        tmp_path=tmp_path_factory.mktemp("rack_motion"), record_motion="evaluation"
    )


@_NEEDS_KINDER
def test_motion_ticks_carry_the_arm_and_every_free_body(*, motion_recording) -> None:
    path, steps = motion_recording
    header = RunRecordReader.header(path=path)
    motion = header.scene["motion"]
    assert motion["stride"] == RackServingReplayRenderer.motion_stride()
    assert motion["fps"] == RackServingReplayRenderer.MOTION_FPS
    assert motion["robot_joints"] == list(SweepOffShelf3DBackend.ROBOT_JOINTS)
    entries = list(RunRecordReader.entries(path=path))
    (action,) = [e for e in entries if isinstance(e, ActionEntry)]
    ticks = [e for e in entries if isinstance(e, TickEntry) and e.seq == action.seq]
    assert steps > 10
    assert action.num_ticks == len(ticks) >= steps // motion["stride"]
    assert (action.skill, action.objects) == ("PickScraper", ["robot", "wiper_0"])
    arm = np.array([t.state["robot_joints"][3:10] for t in ticks])
    assert np.ptp(arm, axis=0).max() > 0.05, "the arm never moved"
    for name in MOVABLES:
        pose = np.asarray(ticks[0].state[name][-7:])
        assert np.isclose(np.linalg.norm(pose[3:]), 1.0), name
    # The rack is a free body too: its pose follows its five support bounds.
    assert len(ticks[0].state["rack_0"]) == 5 + 7


@_NEEDS_KINDER
def test_without_motion_the_recording_keeps_one_state_per_action(*, tmp_path: Path) -> None:
    path, _ = _record_one_skill(tmp_path=tmp_path, record_motion="off")
    assert "motion" not in RunRecordReader.header(path=path).scene
    entries = list(RunRecordReader.entries(path=path))
    (action,) = [e for e in entries if isinstance(e, ActionEntry)]
    assert action.num_ticks <= 1
    assert all("robot_joints" not in e.state for e in entries if isinstance(e, TickEntry))


@_NEEDS_KINDER
def test_a_recorded_tick_restores_and_reads_back(*, motion_recording) -> None:
    path, _ = motion_recording
    header = RunRecordReader.header(path=path)
    assert json.dumps(header.scene)  # the full variant config travels in the header
    ticks = [e for e in RunRecordReader.entries(path=path) if isinstance(e, TickEntry)]
    renderer = RackServingReplayRenderer(scene=header.scene, width=320, height=240)
    try:
        frames = []
        for tick in (ticks[len(ticks) // 2], ticks[-1]):
            frames.append(renderer.show(state=tick.state))
            back = renderer.current_state()
            for name in ("robot_joints", *MOVABLES):
                assert np.allclose(back[name], tick.state[name], atol=1e-5), name
        assert frames[0].shape == (240, 320, 3)
        assert np.abs(frames[0].astype(int) - frames[1].astype(int)).mean() > 0.1
    finally:
        renderer.close()
