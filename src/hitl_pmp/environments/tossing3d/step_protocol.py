"""Tossing3D's `StepProtocolEnvironment`: what `step_protocol.py` hard-wired for this
domain before it served every domain, moved here unchanged."""

import argparse
from collections.abc import Callable
from pathlib import Path

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task

from .cli import STATE_LOG_FILENAME, Tossing3DCli
from .environment import Tossing3DEnvironment
from .problem import Tossing3DProblem
from .skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from .state_log import StateLogHeader, StateLogWriter

EVALUATION_STATE_LOG_FILENAME = "states.jsonl"


class Tossing3DStepProtocol:
    """A static-method container, never instantiated. Takes any `Problem`, as the
    protocol does, and narrows it to this domain's environment."""

    @staticmethod
    def env(*, problem: Problem) -> Tossing3DEnvironment:
        assert isinstance(problem.env, Tossing3DEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DCli.build_practice_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DCli.build_evaluation_problem(args=args)

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        env = Tossing3DStepProtocol.env(problem=problem)
        return DomainContext(
            env=env,
            oracle=Tossing3DOracle(env=env, throw_standoff=args.oracle_throw_standoff),
            skill_provider=Tossing3DSkillProvider(
                env=env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            ),
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> Tossing3DSkillProvider:
        env = Tossing3DStepProtocol.env(problem=problem)
        return Tossing3DSkillProvider(env=env, offer_human_reset=False)

    @staticmethod
    def attach_state_log(
        *, args: argparse.Namespace, problem: Problem, directory: Path, evaluation: bool
    ) -> Callable[[], None]:
        env = Tossing3DStepProtocol.env(problem=problem)
        name = EVALUATION_STATE_LOG_FILENAME if evaluation else STATE_LOG_FILENAME
        writer = StateLogWriter(
            output_path=directory / name,
            header=StateLogHeader(
                layout=env.layout,
                variant=args.variant,
                scene_bg=args.scene_bg,
                canonical_seed=args.canonical_seed,
                seed=args.seed,
                test_env_seed_offset=args.test_env_seed_offset,
            ),
        )
        env.attach_state_log_writer(writer=writer)
        return writer.close

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {"robot_side": "reset_cube_and_bin_near", "opposite_side": "reset_cube_far"}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(task.initial_state.get(obj=Tossing3DEnvironment.scene, feature_name="seed"))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        del task  # The live scene's own goal region is the task's goal.
        return Tossing3DStepProtocol.env(problem=problem).backend().check_goals

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        Tossing3DStepProtocol.env(problem=problem)._log_skill_ticks(action=action)

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return Tossing3DStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return Tossing3DStepProtocol.env(problem=problem).last_skill_error()
