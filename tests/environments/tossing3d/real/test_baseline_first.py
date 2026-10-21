"""`--env tossing3d_real` measures the untrained policy (step 0) before any practice.

The step-0 evaluation runs in this process and uses the room, so practice's scene --
set up and observed before it -- is put back by the person before cycle 0's first
action, exactly as after a cycle-end evaluation. A resume never re-runs it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real import cli as real_cli
from hitl_pmp.environments.tossing3d.real.fake_robot import FakeTossingRobot
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, PracticeCheckpoint, StepFiles

pytestmark = pytest.mark.xdist_group("tossing3d_real_baseline_first")


def _argv(*, output: Path, resume: bool = False, extra: tuple[str, ...] = ()) -> list[str]:
    return [
        *("--env", "tossing3d_real", "--method", "pomdp", "--preset", "tossing3d-canonical"),
        *("--robot", "fake", "--num-test-tasks", "1", "--seed", "0"),
        *("--practice-step-budget", "1500", "--max-steps-per-interaction", "4"),
        *("--output-dir", str(output)),
        *(["--resume", str(output)] if resume else []),
        *extra,
    ]


@pytest.fixture
def room(*, monkeypatch: pytest.MonkeyPatch) -> FakeTossingRobot:
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


def test_by_default_the_baseline_is_measured_then_practices_scene_is_restored(
    *, room: FakeTossingRobot, tmp_path: Path
) -> None:
    del room
    output = tmp_path / "run"
    Cli.main(argv=_argv(output=output))
    # The checkpoint between them is `test_a_run_that_dies_in_cycle_zero_...`'s.
    events = [e for e in _events(output=output) if e["event"] != "checkpoint"]
    kinds = [event["event"] for event in events]
    first = kinds.index("measurement")
    measurement, restored, start = events[first : first + 3]
    assert measurement["index"] == 0 and measurement["practice_steps"] == 0
    assert measurement["learning_updates"] == 0 and "after_cycle" not in measurement
    assert restored["event"] == "practice_scene_restored"
    assert restored["practice_steps"] == 0
    assert start["event"] == "session_start" and start["cycle"] == 0
    assert "session_start" not in kinds[:first]
    # Practice carries on from the scene the person put back.
    assert restored["observed"]["cube"] == pytest.approx(restored["scene"]["cube"], abs=1e-6)
    assert restored["observed"]["bin_side"] == restored["scene"]["bin_side"]
    session = (output / "robot_session.log").read_text().splitlines()
    baseline = next(i for i, line in enumerate(session) if "eval 0 (before practice)" in line)
    restore = next(i for i, line in enumerate(session) if "restore practice:" in line)
    practice = next(i for i, line in enumerate(session) if "cycle 0 [" in line)
    assert baseline < restore < practice
    assert not any("eval 0 (final)" in line for line in session)
    evaluated = json.loads((output / "step_evaluations.json").read_text())
    assert evaluated[0]["practice_steps"] == 0 and evaluated[0]["complete"]


def test_no_initial_measurement_still_opts_out(*, room: FakeTossingRobot, tmp_path: Path) -> None:
    del room
    output = tmp_path / "run"
    Cli.main(argv=_argv(output=output, extra=("--no-initial-measurement",)))
    events = _events(output=output)
    kinds = [event["event"] for event in events]
    assert "practice_scene_restored" not in kinds
    measurements = [event for event in events if event["event"] == "measurement"]
    assert all(m["practice_steps"] > 0 for m in measurements)
    assert kinds.index("session_start") < kinds.index("measurement")


def test_a_resume_never_re_runs_the_baseline_or_its_restore(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = tmp_path / "alone"
    monkeypatch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
    Cli.main(argv=_argv(output=alone))
    expected = _curve(output=alone)
    assert expected[0][1] == 0

    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    output = tmp_path / "resumed"
    save = PracticeCheckpoint.save

    def saving(**kwargs: Any) -> None:
        save(**kwargs)
        if kwargs["header"]["next_cycle"] == 1:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(PracticeCheckpoint, "save", staticmethod(saving))
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))
    interrupted = len(_events(output=output))
    Cli.main(argv=_argv(output=output, resume=True))

    assert _curve(output=output) == expected
    resumed = _events(output=output)[interrupted:]
    assert not [e for e in resumed if e["event"] == "measurement" and e["practice_steps"] == 0]
    # Without --measure-every-cycles a checkpoint remembers no scene: no restore at all.
    assert "practice_scene_restored" not in [e["event"] for e in resumed]
    assert sum(e["event"] == "practice_scene_restored" for e in _events(output=output)) == 1


def test_a_run_that_dies_in_cycle_zero_resumes_without_redoing_the_baseline(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = tmp_path / "alone"
    monkeypatch.setattr(real_cli, "FakeTossingRobot", FakeTossingRobot)
    Cli.main(argv=_argv(output=alone))
    expected = _curve(output=alone)

    robot = FakeTossingRobot()
    monkeypatch.setattr(real_cli, "FakeTossingRobot", lambda: robot)
    output = tmp_path / "resumed"
    original = StepFiles.event

    def interrupting(*, path: Path, **record: Any) -> None:
        original(path=path, **record)
        if record.get("event") == "robot":  # cycle 0's first skill, after the baseline
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(StepFiles, "event", staticmethod(interrupting))
        with pytest.raises(KeyboardInterrupt):
            Cli.main(argv=_argv(output=output))
    before = _events(output=output)
    kinds = [e["event"] for e in before]
    baseline = kinds.index("measurement")
    # Measure, checkpoint (remembering practice's scene), restore, practise.
    assert kinds[baseline : baseline + 4] == [
        "measurement",
        "checkpoint",
        "practice_scene_restored",
        "session_start",
    ]
    header = PracticeCheckpoint.header(path=output / CHECKPOINT_FILENAME)
    assert header["next_cycle"] == 0 and header["practice_steps"] == 0
    assert header["learning_updates"] == 0
    session_lines = len((output / "robot_session.log").read_text().splitlines())

    Cli.main(argv=_argv(output=output, resume=True))
    assert _curve(output=output) == expected
    resumed = _events(output=output)[len(before) :]
    resumed_kinds = [e["event"] for e in resumed]
    assert not [e for e in resumed if e["event"] == "measurement" and e["practice_steps"] == 0]
    # The person is asked for practice's scene again, once, before practice goes on.
    assert resumed_kinds.count("practice_scene_restored") == 1
    assert resumed_kinds.index("practice_scene_restored") < resumed_kinds.index("session_start")
    assert "cycle_discarded" in resumed_kinds
    later = (output / "robot_session.log").read_text().splitlines()[session_lines:]
    assert not any("(before practice)" in line for line in later)
    assert any("restore practice:" in line for line in later)
