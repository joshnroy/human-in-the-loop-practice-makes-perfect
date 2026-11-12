"""A domain can end a held-out task after a named action, whatever that action reported.

RackToServing3D's bulk-transfer protocol gives a task one Sweep or GatherAndSweep: a
partial transfer reports no error, so `--evaluation-max-pour-attempts` (which counts
failures) cannot express it. The fake below ends a task at its first Light Switch
toggle."""

import json
from pathlib import Path
from typing import Any

from hitl_pmp.step_protocol import StepEvaluation
from tests.step_evaluation_fakes import LightSwitchEvaluationProtocol


class FirstToggleEndsTheTask(LightSwitchEvaluationProtocol):
    @staticmethod
    def evaluation_terminal_reason(*, problem: Any, action: Any) -> str | None:
        del problem
        return "toggle_attempt_finished" if float(action[1]) != 0 else None


def _evaluate(*, tmp_path: Path, environment: type) -> Path:
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    output = tmp_path / environment.__name__
    StepEvaluation.run(
        snapshot=str(snapshot),
        configuration={"seed": 0, "num_test_tasks": 4, "evaluation_control_steps": 12},
        output=str(output),
        environment=environment,
    )
    return output


def _toggles(*, output: Path, task: int) -> int:
    actions = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    return sum(1 for a in actions if a["task"] == task and "TurnOnLight" in a["label"])


def test_a_task_ends_at_the_first_action_its_domain_names_terminal(*, tmp_path):
    default = _evaluate(tmp_path=tmp_path, environment=LightSwitchEvaluationProtocol)
    assert _toggles(output=default, task=0) > 1
    output = _evaluate(tmp_path=tmp_path, environment=FirstToggleEndsTheTask)
    tasks = json.loads((output / "results.json").read_text())["tasks"]
    ended = [t for t in tasks if t["reason"] == "toggle_attempt_finished"]
    assert ended, tasks
    for task in ended:
        assert task["solved"] is False
        assert _toggles(output=output, task=task["task"]) == 1
    assert all(t["reason"] in {"toggle_attempt_finished", "goal"} for t in tasks)
    assert all(t["solved"] for t in tasks if t["reason"] == "goal")


class VerdictWhenTheToggleReturns(FirstToggleEndsTheTask):
    @staticmethod
    def evaluation_solved_after_action(*, problem: Any, action: Any) -> bool | None:
        del problem
        return True if float(action[1]) != 0 else None


def test_a_domain_can_score_a_task_when_a_named_action_returns(*, tmp_path):
    ended = _evaluate(tmp_path=tmp_path, environment=FirstToggleEndsTheTask)
    unsolved = {
        t["task"]
        for t in json.loads((ended / "results.json").read_text())["tasks"]
        if t["reason"] == "toggle_attempt_finished"
    }
    assert unsolved
    output = _evaluate(tmp_path=tmp_path, environment=VerdictWhenTheToggleReturns)
    tasks = json.loads((output / "results.json").read_text())["tasks"]
    for task in tasks:
        if task["task"] in unsolved:
            assert task["solved"] is True and task["reason"] == "goal"
            assert _toggles(output=output, task=task["task"]) == 1
