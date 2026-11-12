"""A simulator that diverged ends practice with that reason, and fails only its own
evaluation task, rather than crashing the run on a downstream symptom.

`SimulatorDiverged` is a `BaseException`, like `ControlStepLimitReached`: no skill's
`except ValueError` (or a future `except Exception`) can score it as a skill failure.
The domain here is Light Switch, so nothing leans on MuJoCo."""

import argparse
import json
from pathlib import Path
from typing import Any

from hitl_pmp.core.control_steps import SimulatorDiverged
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.humans.oracle import UnconditionalHumanOracle
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.step_protocol import StepEvaluation, StepPracticeRunner
from tests.step_evaluation_fakes import (
    LightSwitchEvaluationProtocol,
    ObservedLightSwitchEnvironment,
)
from tests.test_step_protocol_generic import CountingLightSwitch, FakeStepProtocol, _args

DETAILS: dict[str, Any] = {
    "time": 7176.5,
    "time_after_reset": 0.075,
    "warning": "mjWARN_BADQACC",
    "index": 21,
    "body": "potsticker_1",
    "scene_step": 71765,
}


class DivergesOnThirdAction(CountingLightSwitch):
    actions_taken: int = 0

    def take_action(self, *, action: Action) -> State:
        self.actions_taken += 1
        if self.actions_taken == 3:
            if self._before is not None:
                self._before()
            raise SimulatorDiverged(**DETAILS)
        return super().take_action(action=action)


class DivergingPracticeProtocol(FakeStepProtocol):
    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> LightSwitchProblem:
        env = DivergesOnThirdAction(grid_size=4)
        tasks = LightSwitchTasks(env=env, seed=args.seed)
        return LightSwitchProblem(env=env, tasks=tasks, human=UnconditionalHumanOracle)


def _events(*, output: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()]


def test_a_divergence_in_practice_ends_it_with_that_reason_and_keeps_the_evaluations(
    *, tmp_path: Path
) -> None:
    output = tmp_path / "run"
    args = _args(output=output)
    problem = DivergingPracticeProtocol.build_practice_problem(args=args)
    method = EesMethod(env=problem.env, skill_provider=LightSwitchSkillProvider(env=problem.env))
    StepPracticeRunner.run(
        args=args, method=method, problem=problem, environment=DivergingPracticeProtocol
    )

    events = _events(output=output)
    (diverged,) = [e for e in events if e["event"] == "simulator_diverged"]
    assert {k: diverged[k] for k in DETAILS} == DETAILS
    assert "practice_steps" in diverged and "cycle" in diverged
    (end,) = [e for e in events if e["event"] == "practice_end"]
    assert end["reason"] == "simulator_diverged"
    # Two robot actions ran before the third diverged; none is recorded for it.
    assert len([e for e in events if e["event"] == "robot"]) == 2
    assert json.loads((output / "status.json").read_text())["phase"] == "simulator_diverged"
    assert "SimulatorDiverged" in (output / "simulator_diverged.txt").read_text()
    assert not (output / "failure.txt").exists()
    # The measurement before practice, then the final one of the policy at divergence.
    evaluations = json.loads((output / "step_evaluations.json").read_text())
    assert len(evaluations) == 2
    assert all(e["complete"] for e in evaluations)


class DivergesInSecondTask(ObservedLightSwitchEnvironment):
    def take_action(self, *, action: Action) -> State:
        if self.task_number == 1:
            if self._before is not None:
                self._before()
            raise SimulatorDiverged(**DETAILS)
        return super().take_action(action=action)


class DivergingEvaluationProtocol(LightSwitchEvaluationProtocol):
    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> LightSwitchProblem:
        env = DivergesInSecondTask(grid_size=3)
        return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env, seed=args.seed))


def test_a_divergence_in_an_evaluation_task_fails_only_that_task(*, tmp_path: Path) -> None:
    snapshot = LightSwitchEvaluationProtocol.write_snapshot(path=tmp_path / "snapshot.pickle")
    output = tmp_path / "evaluation"
    result = StepEvaluation.run(
        snapshot=str(snapshot),
        configuration={"seed": 0, "num_test_tasks": 3, "evaluation_control_steps": 12},
        output=str(output),
        environment=DivergingEvaluationProtocol,
    )

    assert result["complete"]
    tasks = result["tasks"]
    assert [t["task"] for t in tasks] == [0, 1, 2]
    assert tasks[1]["reason"] == "simulator_diverged"
    assert not tasks[1]["solved"]
    assert tasks[1]["simulator_diverged"] == DETAILS
    assert all(t["reason"] != "simulator_diverged" for t in (tasks[0], tasks[2]))
    actions = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    (diverged,) = [a for a in actions if a.get("simulator_diverged")]
    assert diverged["task"] == 1
    assert diverged["simulator_diverged"] == DETAILS
    assert not (output / "failure.txt").exists()


def test_a_divergence_survives_pickling_with_its_details() -> None:
    import pickle

    error = pickle.loads(pickle.dumps(SimulatorDiverged(**DETAILS)))
    assert isinstance(error, SimulatorDiverged)
    assert error.details == DETAILS
