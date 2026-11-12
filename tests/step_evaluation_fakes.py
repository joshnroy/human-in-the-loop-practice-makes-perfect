"""A cheap step-protocol domain for evaluation tests that must run in spawned processes.

`StepEvaluation.run` is spawned by module path, so its domain must be importable from a
fresh interpreter: these classes live here, not inside a test function. Light Switch
supplies a real `EesMethod` deployment whose toggle parameters are drawn from the
method's RNG; every action also draws from the process-global numpy, `random` and torch
RNGs and advances a counter the environment keeps across tasks, and each step's trace
reports all of them, so any cross-task leak of RNG or environment state shows up in
`actions.jsonl`."""

import argparse
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import PrivateAttr

from hitl_pmp.core.method.types import GroundSkill
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.skill_provider import LightSwitchSkillProvider
from hitl_pmp.environments.lightswitch.skills import LightSwitchSkills
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter
from hitl_pmp.step_protocol import DeploymentSnapshot


class ObservedLightSwitchEnvironment(LightSwitchEnvironment):
    """Light Switch whose one control step per action is observable, plus a counter that
    survives between tasks unless the protocol's isolation hook clears it."""

    carried: int = 0
    task_number: int = 0
    fail_task: int | None = None
    sleep_task: int | None = None
    _before: Callable[[], None] | None = PrivateAttr(default=None)
    _after: Callable[[], None] | None = PrivateAttr(default=None)
    _draws: dict[str, float] = PrivateAttr(default_factory=dict)

    def set_control_step_observers(
        self, *, before: Callable[[], None] | None, after: Callable[[], None] | None
    ) -> None:
        self._before, self._after = before, after

    def take_action(self, *, action: Action) -> State:
        if self.task_number == self.fail_task:
            raise RuntimeError(f"scripted failure in task {self.task_number}")
        if self.task_number == self.sleep_task:
            time.sleep(600)
        if self._before is not None:
            self._before()
        state = super().take_action(action=action)
        self.carried += 1
        self._draws = {
            "numpy": float(np.random.random()),  # noqa: NPY002 -- the global RNG is the point
            "random": random.random(),
            "torch": float(torch.rand(1)[0]),
        }
        if self._after is not None:
            self._after()
        return state

    def trace(self) -> dict[str, Any]:
        return {"carried": self.carried, **self._draws}


class LightSwitchEvaluationProtocol:
    """The `StepEvaluation` hooks over `ObservedLightSwitchEnvironment`."""

    @staticmethod
    def env(*, problem: Problem) -> ObservedLightSwitchEnvironment:
        assert isinstance(problem.env, ObservedLightSwitchEnvironment)
        return problem.env

    @staticmethod
    def evaluation_solved_after_action(*, problem: Problem, action: Action) -> bool | None:
        del problem, action
        return None

    @staticmethod
    def evaluation_terminal_reason(*, problem: Problem, action: Action) -> str | None:
        del problem, action
        return None

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> LightSwitchProblem:
        env = ObservedLightSwitchEnvironment(
            grid_size=3,
            fail_task=getattr(args, "fake_fail_task", None),
            sleep_task=getattr(args, "fake_sleep_task", None),
        )
        return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env, seed=args.seed))

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = LightSwitchEvaluationProtocol.env(problem=problem)
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(environment="lightswitch_fake", run_seed=args.seed),
            state_provider=lambda: {
                "robot": [float(env.get_current_state().get(obj=env.robot, feature_name="x"))],
                "carried": [float(env.carried)],
            },
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> LightSwitchSkillProvider:
        return LightSwitchSkillProvider(env=LightSwitchEvaluationProtocol.env(problem=problem))

    @staticmethod
    def evaluation_progress(
        *, problem: Problem, index: int, label: str, task: int, num_tasks: int
    ) -> None:
        del index, label, num_tasks
        LightSwitchEvaluationProtocol.env(problem=problem).task_number = task - 1

    @staticmethod
    def task_seed(*, task: Task) -> int:
        target = task.initial_state.get(obj=LightSwitchEnvironment.light, feature_name="target")
        return int(round(float(target) * 1e6))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        env = LightSwitchEvaluationProtocol.env(problem=problem)
        return lambda: task.goal.is_satisfied(state=env.get_current_state())

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        del problem, action

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        del problem
        return None

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        return LightSwitchEvaluationProtocol.env(problem=problem).trace()

    @staticmethod
    def isolate_evaluation_task(*, problem: Problem) -> None:
        LightSwitchEvaluationProtocol.env(problem=problem).carried = 0

    @staticmethod
    def evaluation_pool() -> None:
        return None

    @staticmethod
    def write_snapshot(*, path: Path, seed: int = 3) -> Path:
        """A deployment that has learned the jump never works, so it walks to the light
        and toggles it with parameters drawn from its RNG until the level reaches the
        target: how many draws a task takes depends on the draws themselves."""
        env = LightSwitchEnvironment(grid_size=3)
        method = EesMethod(env=env, skill_provider=LightSwitchSkillProvider(env=env), seed=seed)
        cells = env.get_cells()
        jump = GroundSkill(
            skill=LightSwitchSkills.JUMP_TO_LIGHT,
            objects=(env.robot, cells[0], cells[1], cells[2], env.light),
        )
        for _ in range(10):
            method.observe_outcome(ground_skill=jump, success=False)
        method.advance_competence_cycle()
        path.write_bytes(DeploymentSnapshot.encode(method=method))
        return path
