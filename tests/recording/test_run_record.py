"""The domain-agnostic run recording: what the writer emits, in what order, and that
the reader gives back exactly what was written."""

import json
from pathlib import Path

import pytest

from hitl_pmp.recording.run_record.reader import RunRecordReader
from hitl_pmp.recording.run_record.types import (
    SCHEMA_VERSION,
    ActionEntry,
    CycleStart,
    Phase,
    RecordingHeader,
    SegmentEnd,
    SegmentStart,
    TickEntry,
)
from hitl_pmp.recording.run_record.writer import RunRecordWriter


class _World:
    """A stand-in environment whose state the test moves by hand."""

    def __init__(self) -> None:
        self.state = {"cube": [0.0, 0.0, 0.0], "bin": [2.0, 0.0, 0.0], "robot": [0.0]}

    def plain(self) -> dict[str, list[float]]:
        return {name: list(values) for name, values in self.state.items()}


def _writer(*, path: Path, world: _World) -> RunRecordWriter:
    return RunRecordWriter(
        output_path=path,
        header=RecordingHeader(
            environment="toy",
            method="ees",
            run_seed=3,
            scene={"seed": 3},
            feature_names={"cube": ["x", "y", "z"], "bin": ["x", "y", "z"], "robot": ["q"]},
        ),
        state_provider=world.plain,
    )


def test_a_practice_segment_records_actions_ticks_and_human_skills_in_order(
    *, tmp_path: Path
) -> None:
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.PRACTICE, scene_seed=3)
    writer.begin_cycle(cycle=0, task_seed=11, start_step=0)
    # The environment reports the skill and its ticks while the skill executes;
    # the step runner reports the outcome once it knows it.
    writer.record_skill(
        name="Pick", objects=("robot", "cube"), params=(0.5,), controller={"yaw_deg": 90.0}
    )
    world.state["robot"] = [0.5]
    writer.record_tick(state=world.plain(), diagnostics={"holding": True})
    world.state["robot"] = [1.0]
    world.state["cube"] = [0.0, 0.0, 0.1]
    writer.record_tick(state=world.plain())
    writer.robot_action(
        label="Pick(robot, cube)", start_step=0, end_step=2, success=True, cycle=0, cost=2.0
    )
    world.state["cube"] = [5.0, 5.0, 0.0]
    writer.human_action(
        skill="reset_cube_far",
        objects=("opposite_side",),
        start_step=2,
        end_step=3,
        success=True,
        cost=7.0,
        cycle=0,
    )
    writer.end_segment(outcome="practice_step_budget", steps=3)
    writer.close()

    header = RunRecordReader.header(path=path)
    assert header.schema_version == SCHEMA_VERSION
    assert header.environment == "toy"
    entries = list(RunRecordReader.entries(path=path))
    kinds = [type(entry) for entry in entries]
    assert kinds == [
        SegmentStart,
        TickEntry,  # the segment's initial state
        CycleStart,
        ActionEntry,
        TickEntry,
        TickEntry,
        ActionEntry,
        TickEntry,  # the state the human left behind
        SegmentEnd,
    ]
    initial = entries[1]
    assert isinstance(initial, TickEntry)
    assert initial.seq is None
    assert initial.state["cube"] == [0.0, 0.0, 0.0]

    robot = entries[3]
    assert isinstance(robot, ActionEntry)
    assert robot.seq == 0
    assert robot.actor == "robot"
    assert (robot.skill, robot.objects, robot.params) == ("Pick", ["robot", "cube"], [0.5])
    assert robot.controller == {"yaw_deg": 90.0}
    assert (robot.start_step, robot.end_step, robot.success, robot.cost) == (0, 2, True, 2.0)
    assert (robot.phase, robot.cycle, robot.num_ticks) == (Phase.PRACTICE, 0, 2)
    assert all(isinstance(t, TickEntry) and t.seq == 0 for t in entries[4:6])
    assert entries[4].diagnostics == {"holding": True}  # type: ignore[union-attr]

    human = entries[6]
    assert isinstance(human, ActionEntry)
    assert (human.seq, human.actor, human.skill) == (1, "human", "reset_cube_far")
    assert human.objects == ["opposite_side"]
    # Only the object the human actually moved, with its resulting state.
    assert human.moved_objects == {"cube": [5.0, 5.0, 0.0]}
    after = entries[7]
    assert isinstance(after, TickEntry)
    assert after.seq == 1
    assert after.state == world.plain()

    end = entries[8]
    assert isinstance(end, SegmentEnd)
    assert (end.outcome, end.steps) == ("practice_step_budget", 3)


def test_a_settled_state_that_differs_from_the_last_tick_is_recorded(*, tmp_path: Path) -> None:
    """The recording's state at every action boundary is the live state, even if the
    environment moved after its last reported tick."""
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.EVALUATION, checkpoint=2, task_index=4, task_seed=99)
    writer.record_skill(name="Toss", objects=("robot",), params=(), controller={})
    world.state["cube"] = [1.0, 0.0, 0.0]
    writer.record_tick(state=world.plain())
    world.state["cube"] = [1.5, 0.0, 0.0]  # moved after the last drained tick
    writer.robot_action(label="Toss(robot)", start_step=0, end_step=1, success=True)
    writer.end_segment(success=True, outcome="goal", steps=1)
    writer.close()

    entries = list(RunRecordReader.entries(path=path))
    ticks = [e for e in entries if isinstance(e, TickEntry)]
    assert ticks[-1].state == world.plain()
    action = next(e for e in entries if isinstance(e, ActionEntry))
    assert (action.phase, action.checkpoint, action.task_index) == (Phase.EVALUATION, 2, 4)
    assert action.num_ticks == 2
    start = entries[0]
    assert isinstance(start, SegmentStart)
    assert start.task_seed == 99


def test_an_action_the_environment_never_reported_is_still_recorded(*, tmp_path: Path) -> None:
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.PRACTICE)
    writer.robot_action(
        label="Pick(robot, cube)", start_step=0, end_step=0, success=False, outcome="no plan"
    )
    writer.close()
    action = next(e for e in RunRecordReader.entries(path=path) if isinstance(e, ActionEntry))
    assert (action.skill, action.objects, action.num_ticks) == ("Pick", ["robot", "cube"], 0)
    assert action.outcome == "no plan"


def test_a_skill_reported_by_name_only_is_named_by_its_label(*, tmp_path: Path) -> None:
    """An environment that knows only which skill runs (SweepOffShelf3D) still writes
    the same action line, from the label, as one whose actions were never reported."""
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.PRACTICE)
    writer.record_skill(name="Pick", objects=(), params=(), controller={})
    world.state["cube"] = [0.0, 0.0, 1.0]
    writer.record_tick(state=world.plain())
    writer.robot_action(
        label="random: Pick(robot, cube), params=[0.5]", start_step=0, end_step=1, success=True
    )
    writer.close()
    action = next(e for e in RunRecordReader.entries(path=path) if isinstance(e, ActionEntry))
    assert (action.skill, action.objects, action.num_ticks) == (
        "random: Pick",
        ["robot", "cube"],
        1,
    )


def test_a_skill_left_unfinished_is_flushed_on_close(*, tmp_path: Path) -> None:
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.PRACTICE)
    writer.record_skill(name="Pick", objects=(), params=(), controller={})
    writer.record_tick(state=world.plain())
    writer.close()
    entries = list(RunRecordReader.entries(path=path))
    action = next(e for e in entries if isinstance(e, ActionEntry))
    assert action.success is None
    assert isinstance(entries[-1], TickEntry)


def test_a_disabled_writer_writes_nothing(*, tmp_path: Path) -> None:
    writer = RunRecordWriter.disabled()
    writer.begin_segment(phase=Phase.PRACTICE)
    writer.record_skill(name="Pick", objects=(), params=(), controller={})
    writer.robot_action(label="Pick()", start_step=0, end_step=0, success=True)
    writer.human_action(skill="reset", start_step=0, end_step=1, success=True)
    writer.end_segment()
    writer.close()
    assert list(tmp_path.iterdir()) == []


def test_an_unknown_schema_version_is_refused(*, tmp_path: Path) -> None:
    path = tmp_path / "recording.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "schema_version": SCHEMA_VERSION + 1, "environment": "x"})
        + "\n"
    )
    with pytest.raises(ValueError, match="schema_version"):
        RunRecordReader.header(path=path)


def test_fixed_geometry_noise_is_not_motion(*, tmp_path: Path) -> None:
    """A static object reported slightly differently is neither a settled state worth a
    tick nor something a human moved."""
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = RunRecordWriter(
        output_path=path,
        header=RecordingHeader(environment="toy"),
        state_provider=world.plain,
        scene_description_provider=lambda: {"static_objects": ["bin"]},
    )
    writer.begin_segment(phase=Phase.PRACTICE)
    writer.record_skill(name="Pick", objects=(), params=(), controller={})
    world.state["robot"] = [1.0]
    writer.record_tick(state=world.plain())
    world.state["bin"] = [2.0000001, 0.0, 0.0]
    writer.robot_action(label="Pick()", start_step=0, end_step=1, success=True)
    world.state["bin"] = [2.0, 0.0, 0.0]
    world.state["cube"] = [3.0, 0.0, 0.0]
    writer.human_action(skill="reset", start_step=1, end_step=2, success=True)
    writer.close()

    assert RunRecordReader.header(path=path).static_objects == ["bin"]
    actions = [e for e in RunRecordReader.entries(path=path) if isinstance(e, ActionEntry)]
    assert actions[0].num_ticks == 1
    assert actions[1].moved_objects == {"cube": [3.0, 0.0, 0.0]}


def test_a_skill_the_goal_latch_cut_short_records_goal_reached(*, tmp_path: Path) -> None:
    """A skill stopped mid-execution because the task was solved did not complete, so
    `success` stays False; `goal_reached` says why it stopped."""
    world = _World()
    path = tmp_path / "recording.jsonl"
    writer = _writer(path=path, world=world)
    writer.begin_segment(phase=Phase.EVALUATION, checkpoint=1, task_index=0)
    writer.robot_action(
        label="Sweep(robot, cube)",
        start_step=0,
        end_step=1796,
        success=False,
        outcome="stopped: goal reached",
        goal_reached=True,
    )
    writer.robot_action(label="Pick(robot, cube)", start_step=0, end_step=5, success=True)
    writer.close()
    actions = [e for e in RunRecordReader.entries(path=path) if isinstance(e, ActionEntry)]
    assert [(a.success, a.goal_reached) for a in actions] == [(False, True), (True, False)]


def test_an_action_line_written_before_goal_reached_existed_reads_as_false() -> None:
    line = {"kind": "action", "seq": 0, "phase": "evaluation", "actor": "robot", "skill": "P"}
    assert ActionEntry.model_validate(line).goal_reached is False
