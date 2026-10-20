"""`--env tossing3d_real` under the step protocol: the sim's adapter
(`Tossing3DStepProtocol`) with three differences, each because the room is real.

- **Evaluations run in this process, synchronously** (`InProcessEvaluations`): a
  spawned process beside practice would need a second robot.
- **The problems** are `Tossing3DRealCli`'s: real environment, planned tasks.
- **The recording** describes the real backend's states and has no static colliders.
"""

import argparse
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.tossing3d.problem import Tossing3DProblem
from hitl_pmp.environments.tossing3d.state_log import StateLogHeader
from hitl_pmp.environments.tossing3d.step_protocol import Tossing3DStepProtocol
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import Tossing3DRealCli


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
    def evaluation_pool() -> InProcessEvaluations:
        return InProcessEvaluations()

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
