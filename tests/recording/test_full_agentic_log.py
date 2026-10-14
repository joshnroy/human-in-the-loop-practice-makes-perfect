"""Full-agentic runs predate the run recording; their per-step logs load into it.

The fixture is a trimmed excerpt of a real run
(`artifacts/shelf-wiper-prompt-full-agentic-20261013/learning-k0/run`): practice lines
1 and 3116-3121, which include the run's first human request, and the first two
control steps of task 0 at checkpoint 5. Bulky constant fields (`action_spec`,
`state_spec`, `fixture_geometry`, `human_skill_quotes`) are stripped; the adapter does
not read them.
"""

import json
from pathlib import Path

from hitl_pmp.recording.run_record.full_agentic_log import FullAgenticRunLog
from hitl_pmp.recording.run_record.types import (
    ActionEntry,
    Phase,
    SegmentEnd,
    SegmentStart,
    TickEntry,
)

RUN = Path(__file__).parent / "fixtures" / "full_agentic_run"


def _observation_state(*, observation: dict) -> dict[str, list[float]]:
    return {o["name"]: list(o["features"].values()) for o in observation["objects"]}


def test_the_header_names_the_environment_and_every_objects_features() -> None:
    header = FullAgenticRunLog.header(run=RUN)
    assert header.environment == "sweepoffshelf3d"
    assert header.method == "full-agent"
    assert header.run_seed == 0
    assert header.feature_names["cube_0"][:7] == ["x", "y", "z", "qw", "qx", "qy", "qz"]
    assert "pos_arm_joint1" in header.feature_names["robot"]


def test_practice_steps_and_human_requests_become_actions_with_recorded_states() -> None:
    entries = list(FullAgenticRunLog.practice_entries(run=RUN))
    lines = [json.loads(line) for line in (RUN / "practice.jsonl").read_text().splitlines()]

    start = entries[0]
    assert isinstance(start, SegmentStart)
    assert start.phase == Phase.PRACTICE
    initial = entries[1]
    assert isinstance(initial, TickEntry)
    assert initial.state == _observation_state(
        observation=lines[0]["response"]["result"]["observation"]
    )

    actions = [e for e in entries if isinstance(e, ActionEntry)]
    assert [(a.actor, a.skill) for a in actions] == [
        ("robot", "control_step"),
        ("robot", "control_step"),
        ("human", "reset_wiper"),
        ("robot", "control_step"),
    ]
    first = actions[0]
    assert first.params == lines[1]["request"]["action"]
    assert (first.start_step, first.end_step) == (3097, 3098)
    human = actions[2]
    assert (human.start_step, human.end_step) == (3099, 3100)
    assert human.success is True
    help_after = _observation_state(observation=lines[4]["response"]["result"]["observation"])
    assert human.moved_objects is not None
    assert "wiper_0" in human.moved_objects
    assert human.moved_objects["wiper_0"] == help_after["wiper_0"]
    assert all(help_after[name] == values for name, values in human.moved_objects.items())

    # Every action is followed by exactly the state its response reported.
    ticks = [e for e in entries if isinstance(e, TickEntry) and e.seq is not None]
    assert [t.seq for t in ticks] == [a.seq for a in actions]
    assert ticks[-1].state == _observation_state(
        observation=lines[-1]["response"]["result"]["observation"]
    )


def test_an_evaluation_task_log_becomes_a_segment_with_its_outcome() -> None:
    entries = list(FullAgenticRunLog.evaluation_entries(run=RUN, checkpoint=5))
    start = entries[0]
    assert isinstance(start, SegmentStart)
    assert (start.phase, start.checkpoint, start.task_index) == (Phase.EVALUATION, 5, 0)
    assert start.task_seed == 357381689
    actions = [e for e in entries if isinstance(e, ActionEntry)]
    assert [(a.start_step, a.end_step) for a in actions] == [(0, 1), (1, 2)]
    end = entries[-1]
    assert isinstance(end, SegmentEnd)
    assert (end.success, end.steps) == (True, 2)


def test_checkpoints_are_discovered_from_the_run_directory() -> None:
    assert FullAgenticRunLog.checkpoints(run=RUN) == [5]
