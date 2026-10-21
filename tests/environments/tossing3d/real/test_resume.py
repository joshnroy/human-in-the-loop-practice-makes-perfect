"""A real practice run checkpoints at every cycle end and resumes after an interruption.

The fake robot stands in for the room: one `FakeTossingRobot` per run, kept across the
interruption the way the real scene stays where it is while the robot charges. Nothing
about the world is restored on resume; the run re-observes it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.step_protocol import (
    CHECKPOINT_FILENAME,
    PREVIOUS_CHECKPOINT_FILENAME,
    PracticeCheckpoint,
    StepFiles,
)

pytestmark = pytest.mark.xdist_group("tossing3d_real_resume")

BUDGET = "1500"
ROBOT_ACTIONS = ("pick", "toss", "open_gripper")


def _argv(*, output: Path, seed: int = 0, resume: bool = False) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "1", "--seed", str(seed)),
        *("--practice-step-budget", BUDGET, "--max-steps-per-interaction", "3"),
        *("--output-dir", str(output)),
        *(["--resume", str(output)] if resume else []),
    ]


@pytest.fixture
def room(*, monkeypatch: pytest.MonkeyPatch) -> FakeTossingRobot:
    """One fake robot for every problem the run builds, practice and evaluation, and
    for every process of the run: the room persists across an interruption."""
    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    return robot


def _events(*, output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def _robot_actions(*, robot: FakeTossingRobot) -> list[tuple[str, tuple[Any, ...]]]:
    return [call for call in robot.calls if call[0] in ROBOT_ACTIONS]


def _discarded_lines(*, events: list[dict[str, Any]]) -> set[int]:
    lines: set[int] = set()
    for event in events:
        if event["event"] == "cycle_discarded":
            lines.update(range(event["first_line"], event["last_line"] + 1))
    return lines


def _kept(*, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The events of the run that counted: none of a discarded cycle's."""
    discarded = _discarded_lines(events=events)
    return [event for line, event in enumerate(events, start=1) if line not in discarded]


def _summary(*, output: Path, robot: FakeTossingRobot) -> dict[str, Any]:
    events = _kept(events=_events(output=output))
    status = json.loads((output / "status.json").read_text())
    return {
        "robot_actions": _robot_actions(robot=robot),
        "robot_events": [(e["label"], e["action"]) for e in events if e["event"] == "robot"],
        "decisions": [
            (e["event"], e["cycle"], e["start_step"], e["end_step"])
            for e in events
            if e["event"] in {"robot", "stop", "human"}
        ],
        "learning_updates": sum(e["event"] == "learning_update" for e in events),
        "human_resets": sum(e["event"] == "human" for e in events),
        "final_steps": [e for e in events if e["event"] == "practice_end"][-1]["practice_steps"],
        "status_learning_updates": status["learning_updates"],
        "status_human_invocations": status["human_invocations"],
        "status_steps": status["practice_steps"],
        "phase": status["phase"],
        "evaluations": [
            (r["practice_steps"], r["num_solved"], r["num_total"])
            for r in json.loads((output / "step_evaluations.json").read_text())
        ],
    }


def _interrupt_on_event(*, monkeypatch: pytest.MonkeyPatch, event: str, cycle: int) -> None:
    """Ctrl-C right after `event` for `cycle` is written, as an operator would press it.
    For `checkpoint`, right after the checkpoint file that follows the event is saved."""
    if event == "checkpoint":
        save = PracticeCheckpoint.save

        def saving(**kwargs: Any) -> None:
            save(**kwargs)
            if kwargs["header"]["next_cycle"] == cycle + 1:
                raise KeyboardInterrupt

        monkeypatch.setattr(PracticeCheckpoint, "save", staticmethod(saving))
        return
    original = StepFiles.event

    def interrupting(*, path: Path, **record: Any) -> None:
        original(path=path, **record)
        if record.get("event") == event and record.get("cycle") == cycle:
            raise KeyboardInterrupt

    monkeypatch.setattr(StepFiles, "event", staticmethod(interrupting))


def _interrupted_run(
    *, output: Path, monkeypatch: pytest.MonkeyPatch, event: str, cycle: int
) -> None:
    with monkeypatch.context() as patch:
        _interrupt_on_event(monkeypatch=patch, event=event, cycle=cycle)
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))


@pytest.mark.parametrize(
    ("event", "cycle"),
    [("checkpoint", 1), ("session_start", 2)],
    ids=["right-after-the-checkpoint", "before-the-next-cycle-acts"],
)
def test_a_run_interrupted_after_a_cycle_and_resumed_is_the_run_left_alone(
    *, event: str, cycle: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone_robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: alone_robot)
    alone = tmp_path / "alone"
    Cli.main(argv=_argv(output=alone))
    expected = _summary(output=alone, robot=alone_robot)
    # The comparison needs cycles after the interruption and robot skills in them.
    assert expected["learning_updates"] > cycle + 1
    assert expected["robot_actions"]

    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    resumed = tmp_path / "resumed"
    _interrupted_run(output=resumed, monkeypatch=monkeypatch, event=event, cycle=cycle)
    assert PracticeCheckpoint.header(path=resumed / CHECKPOINT_FILENAME)["next_cycle"] == 2
    Cli.main(argv=_argv(output=resumed, resume=True))

    assert _summary(output=resumed, robot=robot) == expected
    # Every robot call is the same, in the same order, but the resume's re-observation:
    # one `observe`, and the `can_pick` dry run it asks for where the sim's gate would.
    reobservations = (
        (("observe", ()),),
        (("observe", ()), ("can_pick", ())),
    )
    assert any(
        tuple(robot.calls[i : i + len(extra)]) == extra
        and robot.calls[:i] + robot.calls[i + len(extra) :] == alone_robot.calls
        for extra in reobservations
        for i in range(len(robot.calls))
    )
    resumes = [e for e in _events(output=resumed) if e["event"] == "resume"]
    assert len(resumes) == 1 and resumes[0]["cycle"] == 2


def test_a_resume_with_a_different_config_refuses(
    *, room: FakeTossingRobot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del room
    output = tmp_path / "run"
    _interrupted_run(output=output, monkeypatch=monkeypatch, event="checkpoint", cycle=0)
    before = (output / "step_events.jsonl").read_text()
    with pytest.raises(ValueError, match="seed"):
        Cli.main(argv=_argv(output=output, seed=1, resume=True))
    # Refused before anything was built, observed or written.
    assert (output / "step_events.jsonl").read_text() == before


def test_a_resume_needs_a_checkpoint(*, room: FakeTossingRobot, tmp_path: Path) -> None:
    del room
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        Cli.main(argv=_argv(output=tmp_path / "never-ran", resume=True))


def test_a_resume_into_a_different_output_dir_refuses(
    *, room: FakeTossingRobot, tmp_path: Path
) -> None:
    del room
    argv = _argv(output=tmp_path / "a")
    with pytest.raises(ValueError, match="--output-dir"):
        Cli.main(argv=[*argv, "--resume", str(tmp_path / "b")])


def test_a_cycle_interrupted_part_way_is_discarded(
    *, room: FakeTossingRobot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    # Ctrl-C in the middle of cycle 2, after its first robot skill.
    _interrupted_run(output=output, monkeypatch=monkeypatch, event="robot", cycle=2)
    checkpoint = PracticeCheckpoint.header(path=output / CHECKPOINT_FILENAME)
    # The Ctrl-C left the last checkpoint, cycle 1's, intact, and the one before it.
    assert checkpoint["next_cycle"] == 2
    previous = PracticeCheckpoint.header(path=output / PREVIOUS_CHECKPOINT_FILENAME)
    assert previous["next_cycle"] == 1
    interrupted = _events(output=output)
    partial = [e for e in interrupted if e.get("cycle") == 2]
    assert partial[0]["event"] == "session_start" and partial[-1]["event"] == "robot"
    lost_steps = partial[-1]["end_step"] - checkpoint["practice_steps"]
    assert lost_steps > 0

    Cli.main(argv=_argv(output=output, resume=True))

    events = _events(output=output)
    # The interrupted run's events stay where they were.
    assert events[: len(interrupted)] == interrupted
    discards = [e for e in events if e["event"] == "cycle_discarded"]
    assert len(discards) == 1
    discard = discards[0]
    first = interrupted.index(partial[0]) + 1
    assert (discard["cycle"], discard["first_line"], discard["last_line"]) == (
        2,
        first,
        len(interrupted),
    )
    assert discard["practice_steps_discarded"] == lost_steps
    assert discard["robot_actions_discarded"] == sum(e["event"] == "robot" for e in partial)
    assert discard["practice_steps"] == checkpoint["practice_steps"]

    kept = _kept(events=events)
    # The discarded skill's steps are not charged: every counted step belongs to a
    # kept robot skill, STOP or human reset, and the resumed cycle 2 starts at the
    # checkpoint's step count.
    charged = sum(
        e["end_step"] - e["start_step"] for e in kept if e["event"] in {"robot", "stop", "human"}
    )
    status = json.loads((output / "status.json").read_text())
    assert charged == status["practice_steps"]
    restarted = [e for e in kept if e["event"] == "session_start" and e["cycle"] == 2]
    assert [e["practice_steps"] for e in restarted] == [checkpoint["practice_steps"]]
    # Its outcome never reached the learner: the learning updates are the kept ones.
    assert status["learning_updates"] == sum(e["event"] == "learning_update" for e in kept)
    assert status["phase"] == "complete"


def test_a_ctrl_c_mid_skill_leaves_the_last_checkpoint_readable(
    *, room: FakeTossingRobot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declined y/N confirm raises KeyboardInterrupt from inside a robot skill."""
    output = tmp_path / "run"
    original = FakeTossingRobot.toss
    tosses = 0

    def declined(self: FakeTossingRobot, **kwargs: Any) -> dict[str, Any]:  # noqa: PLR0917
        nonlocal tosses
        tosses += 1
        if tosses == 3:
            raise KeyboardInterrupt
        return original(self, **kwargs)

    monkeypatch.setattr(FakeTossingRobot, "toss", declined)
    with pytest.raises(KeyboardInterrupt):
        Cli.main(argv=_argv(output=output))
    header = PracticeCheckpoint.header(path=output / CHECKPOINT_FILENAME)
    learned = sum(e["event"] == "learning_update" for e in _events(output=output))
    assert header["next_cycle"] == learned >= 1
    assert not list(output.glob("*.tmp"))
    del room
