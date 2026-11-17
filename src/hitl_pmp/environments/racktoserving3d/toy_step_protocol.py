"""The step protocol's hooks for `--env racktoserving3d_toy`: the Serving adapter's
(`RackServingStepProtocol`, delegated to), with the toy's problem, provider, human
reset and one-attempt evaluation."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .problem import RackServingProblem
from .step_protocol import RackServingStepProtocol
from .toy_cli import RackServingToyCli
from .toy_environment import TOY_SKILL_NAMES, RackServingToyEnvironment
from .toy_interface import GATHER, HUMAN_RESET, SWEEP3, SWEEP5, ServingToyTransitions
from .toy_skill_provider import RackServingToySkillProvider


class RackServingToyStepProtocol(SimulatedStepProtocolHooks):
    @staticmethod
    def evaluation_pool() -> None:
        return RackServingStepProtocol.evaluation_pool()

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        return RackServingStepProtocol.attach_recording(
            args=args, problem=problem, output_path=output_path
        )

    @staticmethod
    def isolate_evaluation_task(*, problem: Problem) -> None:
        RackServingStepProtocol.isolate_evaluation_task(problem=problem)

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return RackServingStepProtocol.task_seed(task=task)

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        RackServingStepProtocol.interrupted(problem=problem, action=action)

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return RackServingStepProtocol.adopt_interrupted_skill(
            problem=problem, action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return RackServingStepProtocol.last_skill_error(problem=problem)

    @staticmethod
    def toy_env(*, problem: Problem) -> RackServingToyEnvironment:
        assert isinstance(problem.env, RackServingToyEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingToyCli.build_problem(args=args, mode="training")

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingToyCli.build_problem(args=args, mode="deployment")

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        return RackServingToyCli.practice_context(
            args=args, env=RackServingToyStepProtocol.toy_env(problem=problem)
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> RackServingToySkillProvider:
        return RackServingToySkillProvider(
            env=RackServingToyStepProtocol.toy_env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {HUMAN_RESET: HUMAN_RESET}

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """`Served`, the toy's goal: already held, or measured as the running sweep3 or
        sweep5 lands its count. The step protocol then stops the skill, and its
        transition is taken from the scene as it stands (`take_action`)."""
        del task
        env = RackServingToyStepProtocol.toy_env(problem=problem)

        def served() -> bool:
            if "Served" in env.toy_flags():
                return True
            skill = env.running_toy_skill()
            if skill not in (SWEEP3, SWEEP5):
                return False
            return ServingToyTransitions.succeeded(skill=skill, measured=env.measure())

        return served

    @staticmethod
    def evaluation_solved_after_action(*, problem: Problem, action: Action) -> bool | None:
        del action
        return "Served" in RackServingToyStepProtocol.toy_env(problem=problem).toy_flags()

    @staticmethod
    def evaluation_terminal_reason(*, problem: Problem, action: Action) -> str | None:
        del problem
        return RackServingToyStepProtocol.one_shot_failure_reason(action=action)

    @staticmethod
    def one_shot_failure_reason(*, action: Action) -> str | None:
        """A route skill's attempt is the task's one attempt, as in the toy."""
        values = np.asarray(action, dtype=float)
        if values.shape != (5,) or not np.isfinite(values[0]):
            return None
        skill = TOY_SKILL_NAMES.get(int(values[0]))
        return "route_failed" if skill in (SWEEP3, GATHER, SWEEP5) else None

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        """The toy state after the skill and the measured scene its outcome was read
        from, so every robot event says what the physics decided."""
        env = RackServingToyStepProtocol.toy_env(problem=problem)
        measured = env.toy_measurement()
        return {
            "toy_state": sorted(env.toy_flags()),
            "measured": None if measured is None else measured.model_dump(),
        }
