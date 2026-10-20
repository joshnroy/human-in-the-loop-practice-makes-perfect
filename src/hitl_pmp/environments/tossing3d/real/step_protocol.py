"""`--env tossing3d_real` under the step protocol: the sim's adapter
(`Tossing3DStepProtocol`) with three differences, each because the room is real.

- **Evaluations run in this process, synchronously** (`InProcessEvaluations`): a
  spawned process beside practice would need a second robot.
- **The problems** are `Tossing3DRealCli`'s: real environment, planned tasks.
- **The recording** describes the real backend's states and has no static colliders.

It also hands practice's and each evaluation's progress to the backend
(`practice_progress`, `evaluation_progress`), for the line the person sees before each
skill; the sim's adapter ignores both. With `--measure-every-cycles` it remembers
practice's scene before each cycle-end evaluation and has the person put it back after
(`practice_scene`, `restore_practice_scene`); the sim's adapter refuses both.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from hitl_pmp.core.problem.environment.types import State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .backend import PracticeScene, RealTossingBackend
from .cli import Tossing3DRealCli
from .environment import RealTossing3DEnvironment


class InProcessEvaluations:
    """The step protocol's evaluation pool, run here and now: `submit` evaluates before
    it returns. The interface of `step_protocol.FreshProcessEvaluations`."""

    def submit(self, *, fn: Callable[..., Any], **kwargs: Any) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(**kwargs))
        except Exception as error:  # noqa: BLE001 -- recorded on the future, as a pool does
            future.set_exception(error)
        return future

    def has_running_process(self) -> bool:
        return False

    def finish(self) -> None:
        """Every evaluation finished when it was submitted."""

    def abort(self) -> None:
        """Nothing is ever queued."""


class Tossing3DRealStepProtocol(Tossing3DStepProtocol):
    """A static-method container, never instantiated."""

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_practice_problem(args=args)

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Tossing3DProblem:
        return Tossing3DRealCli.build_evaluation_problem(args=args)

    @staticmethod
    def resume_practice_environment(*, problem: Problem, state: State) -> None:
        env = Tossing3DStepProtocol.env(problem=problem)
        assert isinstance(env, RealTossing3DEnvironment)
        env.reobserve(state=state)

    @staticmethod
    def evaluation_pool() -> InProcessEvaluations:
        return InProcessEvaluations()

    @staticmethod
    def practice_progress(
        *, problem: Problem, cycle: int, action: int, max_actions: int, steps: int, budget: int
    ) -> None:
        """Handed to the backend, which shows it before the next skill runs."""
        backend = Tossing3DStepProtocol.env(problem=problem).backend()
        assert isinstance(backend, RealTossingBackend)
        backend.practice_progress(
            cycle=cycle, action=action, max_actions=max_actions, steps=steps, budget=budget
        )

    @staticmethod
    def evaluation_progress(
        *, problem: Problem, index: int, after_cycle: int | None, task: int, num_tasks: int
    ) -> None:
        """Handed to the evaluation's backend: `eval 3 after cycle 3: task 2/5`."""
        backend = Tossing3DStepProtocol.env(problem=problem).backend()
        assert isinstance(backend, RealTossingBackend)
        backend.evaluation_progress(
            index=index, after_cycle=after_cycle, task=task, num_tasks=num_tasks
        )

    @staticmethod
    def practice_scene(*, problem: Problem) -> PracticeScene:
        env = Tossing3DStepProtocol.env(problem=problem)
        assert isinstance(env, RealTossing3DEnvironment)
        return env.practice_scene()

    @staticmethod
    def restore_practice_scene(*, problem: Problem, scene: Any) -> dict[str, Any]:
        env = Tossing3DStepProtocol.env(problem=problem)
        assert isinstance(env, RealTossing3DEnvironment) and isinstance(scene, PracticeScene)
        env.restore_practice_scene(scene=scene)
        return {
            "scene": scene.model_dump(mode="json"),
            "observed": env.practice_scene().model_dump(mode="json"),
        }

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        env = Tossing3DStepProtocol.env(problem=problem)
        scene = StateLogHeader(
            layout=env.layout,
            variant=args.variant,
            scene_bg=False,
            canonical_seed=args.canonical_seed,
            seed=args.seed,
            test_env_seed_offset=args.test_env_seed_offset,
        )

        def state() -> dict[str, list[float]]:
            backend = env.backend()
            return backend.snapshot_to_plain(snapshot=backend.live_state())

        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="tossing3d_real",
                method=args.method,
                run_seed=args.seed,
                scene={**scene.model_dump(mode="json"), "robot": args.robot},
            ),
            state_provider=state,
            scene_description_provider=lambda: {
                "feature_names": {
                    obj.name: list(obj.type.feature_names)
                    for obj in (env.robot, env.cube, env.bin, env.barrier)
                },
                "static_objects": [],
            },
            task_seed_provider=lambda task: Tossing3DStepProtocol.task_seed(task=task),
        )
        env.attach_state_log_writer(writer=writer)
        return writer
