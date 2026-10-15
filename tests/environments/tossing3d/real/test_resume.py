"""A real run can pause at a session boundary (a battery swap) and resume intact.

The check is equivalence: on the deterministic fake plant, a run paused part-way and
resumed takes exactly the actions, and reaches exactly the result, of the same run
left alone. The fake world persists across the pause the way the room does
(`RealWorldRegistry` is not cleared in between), so only the run's own state -- the
method's learning, the clock, the metrics -- has to survive through the checkpoint.
"""

import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.environments.tossing3d.real.cli import RealWorldRegistry
from hitl_pmp.step_protocol import CHECKPOINT_FILENAME, PAUSE_FILENAME, StepFiles

pytestmark = pytest.mark.xdist_group("tossing3d_real_resume")

BUDGET = "14"


@pytest.fixture(autouse=True)
def _fresh_registry():
    RealWorldRegistry.clear()
    yield
    RealWorldRegistry.clear()


def _argv(*, method: str, output: Path, resume: bool = False) -> list[str]:
    return [
        "--env",
        "tossing3d_real",
        "--method",
        method,
        "--robot-plant",
        "fake",
        "--practice-step-budget",
        BUDGET,
        "--practice-reset-policy",
        "never",
        "--max-steps-per-interaction",
        "3",
        "--num-test-tasks",
        "2",
        "--evaluation-control-steps",
        "4",
        "--output-dir",
        str(output),
        "--defer-rendering",
        *(["--resume"] if resume else []),
    ]


def _pause_after_session(*, output: Path, cycle: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Touch the PAUSE file once session `cycle` has learned, as an operator would."""
    original = StepFiles.event

    def event(*, path: Path, **record) -> None:
        original(path=path, **record)
        if record.get("event") == "learning_update" and record.get("cycle") == cycle:
            (output / PAUSE_FILENAME).touch()

    monkeypatch.setattr(StepFiles, "event", staticmethod(event))


def _events(*, output: Path) -> list[dict]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def _decisions(*, output: Path) -> list[tuple[str, int, int]]:
    """Every robot skill, STOP and human reset, with its steps -- the run's trajectory."""
    return [
        (e["event"], e.get("start_step"), e.get("end_step"))
        for e in _events(output=output)
        if e["event"] in {"robot", "stop", "human"}
    ]


def _robot_actions(*, output: Path) -> list[tuple[str, list[float]]]:
    events = [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]
    return [(e["label"], e["action"]) for e in events if e["event"] == "robot"]


@pytest.mark.parametrize("method", ["ees", "pomdp"])
def test_a_paused_and_resumed_run_is_the_run_left_alone(
    *, method: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alone = tmp_path / "alone"
    Cli.main(argv=_argv(method=method, output=alone))
    if method == "ees":
        # Model B STOPs every session at its priors on this plant; EES acts, so its
        # case is the one that carries robot skills across the pause.
        assert _robot_actions(output=alone)
    RealWorldRegistry.clear()

    paused = tmp_path / "paused"
    paused.mkdir()
    with monkeypatch.context() as patch:
        _pause_after_session(output=paused, cycle=1, monkeypatch=patch)
        Cli.main(argv=_argv(method=method, output=paused))
    status = json.loads((paused / "status.json").read_text())
    assert status["phase"] == "paused"
    assert (paused / CHECKPOINT_FILENAME).exists()
    assert not (paused / "step_evaluations.json").exists() or not json.loads(
        (paused / "step_evaluations.json").read_text()
    )
    assert len(_events(output=paused)) < len(_events(output=alone))

    (paused / PAUSE_FILENAME).unlink()
    Cli.main(argv=_argv(method=method, output=paused, resume=True))

    assert _robot_actions(output=paused) == _robot_actions(output=alone)
    assert _decisions(output=paused) == _decisions(output=alone)
    left_alone = json.loads((alone / "step_evaluations.json").read_text())
    resumed = json.loads((paused / "step_evaluations.json").read_text())
    assert [(r["practice_steps"], r["num_solved"], r["num_total"]) for r in resumed] == [
        (r["practice_steps"], r["num_solved"], r["num_total"]) for r in left_alone
    ]
    assert json.loads((paused / "status.json").read_text())["phase"] == "complete"


def test_resume_needs_a_checkpoint(*, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        Cli.main(argv=_argv(method="ees", output=tmp_path, resume=True))


def test_the_sim_does_not_offer_resume(*, tmp_path: Path) -> None:
    from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol

    with pytest.raises(NotImplementedError):
        Tossing3DStepProtocol.resume_practice_environment(problem=None, state=None)
    del tmp_path
