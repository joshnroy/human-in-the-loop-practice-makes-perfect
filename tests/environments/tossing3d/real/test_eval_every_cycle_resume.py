"""Evaluation after every cycle, combined with cycle checkpoints and `--resume`.

Each cycle ends: learn, checkpoint (its evaluation pending), evaluate, checkpoint again
(evaluation recorded, with practice's remembered scene), restore, practise. A resume
re-runs an evaluation that never finished and never one that did, then asks the person
to put practice's scene back; no curve point is dropped or counted twice.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, PracticeCheckpoint, StepFiles

pytestmark = pytest.mark.xdist_group("tossing3d_real_eval_every_cycle_resume")


def _argv(*, output: Path, resume: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "1", "--seed", "0"),
        *("--practice-step-budget", "1500", "--max-steps-per-interaction", "4"),
        *("--measure-every-cycles", "1", "--output-dir", str(output)),
        *(["--resume", str(output)] if resume else []),
        *extra,
    ]


@pytest.fixture
def room(*, monkeypatch: pytest.MonkeyPatch) -> FakeTossingRobot:
    """One fake robot for every problem and process of the run: the room persists."""
    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    return robot


def _events(*, output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def _curve(*, output: Path) -> list[tuple[Any, ...]]:
    return [
        (r.get("after_cycle"), r["practice_steps"], r["learning_updates"], r["num_total"])
        for r in json.loads((output / "step_evaluations.json").read_text())
    ]


def test_each_cycle_checkpoints_before_its_evaluation_and_again_once_it_is_recorded(
    *, room: FakeTossingRobot, tmp_path: Path
) -> None:
    del room
    output = tmp_path / "run"
    Cli.main(argv=_argv(output=output))
    events = _events(output=output)
    evaluated = json.loads((output / "step_evaluations.json").read_text())
    updates = [
        i
        for i, event in enumerate(events)
        if event["event"] == "learning_update" and event["reason"] != "practice_step_budget"
    ]
    assert len(updates) >= 2
    for position in updates:
        cycle = events[position]["cycle"]
        before, measurement, after, restored = events[position + 1 : position + 5]
        assert before["event"] == "checkpoint" and before["cycle"] == cycle
        assert before["evaluation"].endswith(f"pending (after cycle {cycle})")
        assert measurement["event"] == "measurement" and measurement["after_cycle"] == cycle
        assert after["event"] == "checkpoint" and after["cycle"] == cycle
        index = measurement["index"]
        solved = evaluated[index]["num_solved"]
        assert after["evaluation"] == f"evaluation {index} done: {solved}/1"
        assert restored["event"] == "practice_scene_restored"
    header = PracticeCheckpoint.header(path=output / CHECKPOINT_FILENAME)
    assert "done:" in header["evaluation"]


def _interrupt_after(
    *, monkeypatch: pytest.MonkeyPatch, event: str, after_cycle: int, evaluation: str = ""
) -> None:
    """Ctrl-C right after a `measurement` event (the evaluation then never finishes) or
    a `checkpoint` save whose evaluation text contains `evaluation`, for `after_cycle`."""
    if event == "checkpoint":
        save = PracticeCheckpoint.save

        def saving(**kwargs: Any) -> None:
            save(**kwargs)
            header = kwargs["header"]
            if header["next_cycle"] == after_cycle + 1 and evaluation in (
                header["evaluation"] or ""
            ):
                raise KeyboardInterrupt

        monkeypatch.setattr(PracticeCheckpoint, "save", staticmethod(saving))
        return
    original = StepFiles.event

    def interrupting(*, path: Path, **record: Any) -> None:
        original(path=path, **record)
        if record.get("event") == event and record.get("after_cycle") == after_cycle:
            raise KeyboardInterrupt

    monkeypatch.setattr(StepFiles, "event", staticmethod(interrupting))


@pytest.mark.parametrize(
    ("event", "evaluation", "reruns"),
    [("measurement", "", True), ("checkpoint", "done:", False)],
    ids=["during-the-evaluation", "after-the-evaluation-was-recorded"],
)
def test_a_resume_re_runs_only_an_unfinished_evaluation_then_restores_practice(
    *, event: str, evaluation: str, reruns: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = tmp_path / "alone"
    monkeypatch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
    Cli.main(argv=_argv(output=alone))
    expected = _curve(output=alone)
    assert len(expected) >= 3

    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    output = tmp_path / "resumed"
    with monkeypatch.context() as patch:
        _interrupt_after(monkeypatch=patch, event=event, after_cycle=1, evaluation=evaluation)
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))
    interrupted = len(_events(output=output))
    Cli.main(argv=_argv(output=output, resume=True))

    # Every curve point once: none dropped, none counted twice.
    assert _curve(output=output) == expected
    resumed = _events(output=output)[interrupted:]
    measured = [e for e in resumed if e["event"] == "measurement" and e.get("after_cycle") == 1]
    assert len(measured) == (1 if reruns else 0)
    # Before practice goes on, the person puts practice's scene back.
    kinds = [e["event"] for e in resumed]
    restore = kinds.index("practice_scene_restored")
    assert restore < kinds.index("session_start")
    assert resumed[restore]["cycle"] == 1
    session = (output / "robot_session.log").read_text()
    assert session.count("restore practice:") == sum(
        e["event"] == "practice_scene_restored" for e in _events(output=output)
    )
