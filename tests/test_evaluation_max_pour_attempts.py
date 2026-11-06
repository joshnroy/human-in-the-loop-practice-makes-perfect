"""`--evaluation-max-pour-attempts`: a held-out task ends at its Nth failed one-shot skill.

In the Pouring kitchen every failure is recoverable during evaluation, so a test task can
retry a bad Pour until the step budget runs out. At deployment the pan is hot and a bad
pour is committed: with the flag at 1, a task ends unsolved (`pour_failed`) the moment its
first Pour finishes unsuccessfully. The domain names its one-shot skills through
`one_shot_failure_reason`; the fake below treats Light Switch's toggles as one-shot."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.environments.pouring3d.practice.skills import SKILL_NAMES
from hitl_pmp.environments.pouring3d.practice.step_protocol import FrozenPourStepProtocol
from hitl_pmp.step_protocol import StepEvaluation, StepProtocolResume
from tests.practice_cost_fixtures import CostFixtures
from tests.step_evaluation_fakes import LightSwitchEvaluationProtocol

PINNED_DEFAULT = [
    (False, 12, "step_budget"),
    (True, 6, "goal"),
    (False, 12, "step_budget"),
    (True, 3, "goal"),
]


class OneShotToggleProtocol(LightSwitchEvaluationProtocol):
    """Light Switch whose toggles are one-shot and every action reports a failure."""

    @staticmethod
    def last_skill_error(*, problem: Any) -> str | None:
        del problem
        return "observed success conditions not achieved"

    @staticmethod
    def one_shot_failure_reason(*, action: Any) -> str | None:
        return "toggle_failed" if float(action[1]) != 0 else None


def _evaluate(*, tmp_path: Path, name: str, **extra: Any) -> Path:
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    output = tmp_path / name
    StepEvaluation.run(
        snapshot=str(snapshot),
        configuration={
            "seed": 0,
            "num_test_tasks": 4,
            "evaluation_control_steps": 12,
            **extra,
        },
        output=str(output),
        environment=OneShotToggleProtocol,
    )
    return output


def _tasks(*, output: Path) -> list[dict[str, Any]]:
    return json.loads((output / "results.json").read_text())["tasks"]


def _toggles(*, output: Path, task: int) -> int:
    lines = (output / "actions.jsonl").read_text().splitlines()
    actions = [json.loads(line) for line in lines]
    return sum(1 for a in actions if a["task"] == task and "TurnOnLight" in a["label"])


def test_without_the_flag_a_failed_one_shot_skill_is_retried_as_before(*, tmp_path):
    output = _evaluate(tmp_path=tmp_path, name="default")
    assert [(t["solved"], t["steps"], t["reason"]) for t in _tasks(output=output)] == (
        PINNED_DEFAULT
    )
    assert _toggles(output=output, task=0) > 1


def test_with_one_attempt_a_task_ends_unsolved_at_its_first_failed_one_shot_skill(*, tmp_path):
    output = _evaluate(tmp_path=tmp_path, name="one_shot", evaluation_max_pour_attempts=1)
    tasks = _tasks(output=output)
    failed = [t for t in tasks if t["reason"] == "toggle_failed"]
    assert failed, tasks
    for task in failed:
        assert task["solved"] is False
        assert _toggles(output=output, task=task["task"]) == 1
    # A task whose single toggle reaches the goal is still solved.
    assert all(t["reason"] in {"toggle_failed", "goal"} for t in tasks)
    assert all(t["solved"] for t in tasks if t["reason"] == "goal")


def test_two_attempts_allow_exactly_one_retry(*, tmp_path):
    output = _evaluate(tmp_path=tmp_path, name="two", evaluation_max_pour_attempts=2)
    for task in _tasks(output=output):
        if task["reason"] == "toggle_failed":
            assert _toggles(output=output, task=task["task"]) == 2


def _pouring_action(*, skill: str) -> np.ndarray:
    return np.array([SKILL_NAMES.index(skill), 0, 2, 1.0, 0, 0], dtype=float)


def test_the_pouring_domains_name_every_pour_one_shot_and_nothing_else():
    assert FrozenPourStepProtocol.one_shot_failure_reason(action=_pouring_action(skill="Pour")) == (
        "pour_failed"
    )
    for skill in SKILL_NAMES:
        if skill != "Pour":
            assert (
                FrozenPourStepProtocol.one_shot_failure_reason(action=_pouring_action(skill=skill))
                is None
            )


def test_the_flag_defaults_to_unlimited_and_parses_a_positive_count():
    argv = ["--env", "pouring3d_kitchen", "--method", "ees", "--output-dir", "/x"]
    assert CostFixtures.parse(argv=argv).evaluation_max_pour_attempts is None
    parsed = CostFixtures.parse(argv=[*argv, "--evaluation-max-pour-attempts", "1"])
    assert parsed.evaluation_max_pour_attempts == 1


def test_a_run_recorded_before_the_flag_existed_still_resumes():
    before = {"seed": "0"}
    assert StepProtocolResume.comparable(arguments=before) == StepProtocolResume.comparable(
        arguments={**before, "evaluation_max_pour_attempts": "None"}
    )
