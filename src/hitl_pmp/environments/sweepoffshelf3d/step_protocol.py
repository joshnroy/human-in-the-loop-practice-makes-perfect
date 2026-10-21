"""SweepOffShelf3D's `StepProtocolEnvironment`: practice measured in KINDER control steps
on the shelf, with its two human skills and KINDER's own goal verdict."""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import SweepOffShelf3DCli
from .environment import SweepOffShelf3DEnvironment
from .objects import SweepOffShelf3DObjects as O
from .problem import SweepOffShelf3DProblem
from .skill_provider import SweepOffShelf3DOracle, SweepOffShelf3DSkillProvider
from .skill_trace import SkillTraces
from .skills import RESET_CUBES_NAME, RESET_WIPER_NAME


class SweepOffShelf3DStepProtocol(SimulatedStepProtocolHooks):
    """A static-method container, never instantiated. Takes any `Problem`, as the
    protocol does, and narrows it to this domain's environment."""

    @staticmethod
    def env(*, problem: Problem) -> SweepOffShelf3DEnvironment:
        assert isinstance(problem.env, SweepOffShelf3DEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> SweepOffShelf3DProblem:
        return SweepOffShelf3DCli.build_problem(args=args, human_seed=args.seed)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> SweepOffShelf3DProblem:
        # As `SweepOffShelf3DCli.run_method`: its own human stream, the same test tasks.
        return SweepOffShelf3DCli.build_problem(args=args, human_seed=args.seed + 1)

    @staticmethod
    def evaluation_pool() -> None:
        """The protocol's default: one fresh spawned process per evaluation."""
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        env = SweepOffShelf3DStepProtocol.env(problem=problem)
        return DomainContext(
            env=env,
            skill_provider=SweepOffShelf3DSkillProvider(
                env=env,
                human_reset_practice_cost=args.human_reset_practice_cost,
                offer_human_reset=args.human_reset,
            ),
            oracle=SweepOffShelf3DOracle(env=env),
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> SweepOffShelf3DSkillProvider:
        env = SweepOffShelf3DStepProtocol.env(problem=problem)
        return SweepOffShelf3DSkillProvider(env=env, offer_human_reset=False)

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        """Every action and its outcome, and the live scene at each action boundary.
        The shelf drains no per-tick states, so the recording carries no ticks."""
        env = SweepOffShelf3DStepProtocol.env(problem=problem)
        return RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="sweepoffshelf3d",
                method=args.method,
                run_seed=args.seed,
                scene={
                    "scene_bg": args.scene_bg,
                    "canonical_seed": args.canonical_seed,
                    "reset_cubes_success_probability": args.reset_cubes_success_probability,
                    "test_env_seed_offset": args.test_env_seed_offset,
                },
            ),
            state_provider=lambda: SweepOffShelf3DStepProtocol.live_state(env=env),
            task_seed_provider=lambda task: SweepOffShelf3DStepProtocol.task_seed(task=task),
        )

    @staticmethod
    def live_state(*, env: SweepOffShelf3DEnvironment) -> dict[str, list[float]]:
        return {
            name: [float(v) for v in values] for name, values in env.backend().features().items()
        }

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        # The destinations are the human skills' own names (`SkillProvider.human_skill_name`).
        return {RESET_CUBES_NAME: RESET_CUBES_NAME, RESET_WIPER_NAME: RESET_WIPER_NAME}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(round(task.initial_state.get(obj=O.scene, feature_name="seed")))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """KINDER's own verdict, as for Tossing3D: the live scene's goal is the task's."""
        del task
        core = SweepOffShelf3DStepProtocol.env(problem=problem).backend().core()
        return lambda: bool(core._check_goals())  # noqa: SLF001

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        # Evaluation: nothing is buffered per action, and the next task resets the scene.
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return SweepOffShelf3DStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return SweepOffShelf3DStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        trace = SkillTraces.last(env=SweepOffShelf3DStepProtocol.env(problem=problem))
        return None if trace is None else trace.model_dump(mode="json")
