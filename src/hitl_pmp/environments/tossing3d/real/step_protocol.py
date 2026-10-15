"""`--env tossing3d_real` under the step protocol (`hitl_pmp/step_protocol.py`).

The sim's adapter (`Tossing3DStepProtocol`) with three differences, each because the
room is real:

- **Evaluations run in this process, synchronously** (`InProcessEvaluations`). The sim
  spawns a fresh process per evaluation and runs it beside practice; a second process
  cannot drive the robot practice is driving, and the robot cannot practise and be
  evaluated at once. Practice therefore waits while an evaluation runs, and both use
  the one `RealWorld` (`RealWorldRegistry`).
- **The recording** describes the real backend's states (its feature names are this
  domain's `Type`s), and has no static colliders.
- **The problems** are `Tossing3DRealCli`'s: real environment, planned tasks, the real
  human.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import Tossing3DRealCli
from .environment import RealTossing3DEnvironment


class InProcessEvaluations:
    """The step protocol's evaluation pool, run here and now rather than in a spawned
    process: `submit` evaluates before it returns. Same interface as
    `step_protocol.FreshProcessEvaluations`, which the sim uses."""

    def submit(self, *, fn: Callable[..., Any], **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(**kwargs))
        except Exception as error:  # noqa: BLE001 - recorded on the future, as a pool does
            future.set_exception(error)
        return future

    def has_running_process(self) -> bool:
        return False

    def finish(self) -> None:
        """Every evaluation finished when it was submitted."""

    def abort(self) -> None:
        """Nothing is ever queued."""


class Tossing3DRealStepProtocol:
    """A static-method container, never instantiated."""

    @staticmethod
    def env(*, problem: Problem) -> RealTossing3DEnvironment:
        assert isinstance(problem.env, RealTossing3DEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_practice_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_evaluation_problem(args=args)

    @staticmethod
    def evaluation_pool() -> InProcessEvaluations:
        return InProcessEvaluations()

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        env = Tossing3DRealStepProtocol.env(problem=problem)
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
        env = Tossing3DRealStepProtocol.env(problem=problem)
        return Tossing3DSkillProvider(env=env, offer_human_reset=False)

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = Tossing3DRealStepProtocol.env(problem=problem)
        scene = StateLogHeader(
            layout=env.layout,
            variant=args.variant,
            scene_bg=False,
            canonical_seed=args.canonical_seed,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
        )
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="tossing3d_real",
                method=args.method,
                run_seed=args.seed,
                scene={
                    **scene.model_dump(mode="json"),
                    "robot_plant": args.robot_plant,
                    "scene_layout": env.scene_layout.model_dump(mode="json"),
                },
            ),
            state_provider=lambda: Tossing3DRealStepProtocol.state(env=env),
            scene_description_provider=lambda: Tossing3DRealStepProtocol.describe(env=env),
            task_seed_provider=lambda task: Tossing3DRealStepProtocol.task_seed(task=task),
        )
        env.attach_state_log_writer(writer=writer)
        return writer

    @staticmethod
    def state(*, env: Tossing3DEnvironment) -> dict[str, list[float]]:
        backend = env.backend()
        return backend.snapshot_to_plain(snapshot=backend.live_state())

    @staticmethod
    def describe(*, env: Tossing3DEnvironment) -> dict[str, Any]:
        return {
            "feature_names": {
                obj.name: list(obj.type.feature_names)
                for obj in (env.robot, env.cube, env.bin, env.barrier)
            },
            "static_objects": [],
        }

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return Tossing3DStepProtocol.human_skill_names()

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return Tossing3DStepProtocol.task_seed(task=task)

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        del task
        return Tossing3DRealStepProtocol.env(problem=problem).backend().check_goals

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        Tossing3DStepProtocol.interrupted(problem=problem, action=action)

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return Tossing3DStepProtocol.adopt_interrupted_skill(
            problem=problem, action=action, reason=reason
        )

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return Tossing3DStepProtocol.last_skill_error(problem=problem)
