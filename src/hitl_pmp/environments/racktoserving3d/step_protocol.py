"""Rack-serving integration with the existing control-step practice runner."""

import argparse
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hitl_pmp.cli_protocols import SimulatedStepProtocolHooks
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.environments.sweepoffshelf3d.backend import SweepOffShelf3DBackend
from hitl_pmp.recording.run_record.types import RecordingHeader
from hitl_pmp.recording.run_record.writer import RunRecordWriter

from .cli import RackServingCli
from .environment import RackServingEnvironment
from .objects import RackServing3DObjects as Objects
from .problem import RackServingProblem
from .recording import RackServingReplayRenderer
from .semantics import HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY
from .skill_provider import RackServingSkillProvider
from .skills import SKILL_IDS


class RackServingStepProtocol(SimulatedStepProtocolHooks):
    """A spawn-importable adapter; all dynamics remain in the rack environment."""

    simulation_checkpoints = True

    @staticmethod
    def checkpoint_practice_environment(*, problem: Problem) -> Any:
        return RackServingStepProtocol.env(problem=problem).simulation_checkpoint()

    @staticmethod
    def restore_checkpoint_environment(*, problem: Problem, checkpoint: Any) -> None:
        RackServingStepProtocol.env(problem=problem).restore_simulation_checkpoint(
            checkpoint=checkpoint
        )

    @staticmethod
    def env(*, problem: Problem) -> RackServingEnvironment:
        assert isinstance(problem.env, RackServingEnvironment)
        return problem.env

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingCli.build_problem(args=args, mode="training")

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> RackServingProblem:
        return RackServingCli.build_problem(args=args, mode="deployment")

    @staticmethod
    def evaluation_pool() -> None:
        """Use the runner's fresh spawned process for each evaluation."""
        return None

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        return RackServingCli.practice_context(
            args=args, env=RackServingStepProtocol.env(problem=problem)
        )

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> RackServingSkillProvider:
        return RackServingSkillProvider(
            env=RackServingStepProtocol.env(problem=problem), offer_human_reset=False
        )

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        """Live action-boundary states; with `env.motion_recording` (`--record-motion`)
        also every stride-th control step, as `RackServingBackend.motion_state`, and the
        header's `scene.motion` says so -- what `RackServingReplayRenderer` plays."""
        env = RackServingStepProtocol.env(problem=problem)
        scene: dict[str, Any] = {
            **env.config.model_dump(mode="json"),
            "scene_bg": env.scene_bg,
            "canonical_seed": env.canonical_seed,
            "skill_step_limit": env.skill_step_limit,
            "test_env_seed_offset": args.test_env_seed_offset,
        }
        state: Callable[[], dict[str, list[float]]] = functools.partial(
            RackServingStepProtocol.live_state, env=env
        )
        if env.motion_recording:
            # Every state in a motion recording has one shape, boundaries included.
            state = env.backend().motion_state
            scene["motion"] = {
                "fps": RackServingReplayRenderer.MOTION_FPS,
                "stride": RackServingReplayRenderer.motion_stride(),
                "control_frequency": SweepOffShelf3DBackend.CONTROL_FREQUENCY,
                "robot_joints": list(SweepOffShelf3DBackend.ROBOT_JOINTS),
            }
        writer = RunRecordWriter(
            output_path=output_path,
            header=RecordingHeader(
                environment="racktoserving3d",
                method=args.method,
                run_seed=args.seed,
                scene=scene,
            ),
            state_provider=state,
            task_seed_provider=lambda task: RackServingStepProtocol.task_seed(task=task),
        )
        if env.motion_recording:
            env.attach_motion_sink(sink=writer, stride=scene["motion"]["stride"])
        return writer

    @staticmethod
    def live_state(*, env: RackServingEnvironment) -> dict[str, list[float]]:
        return {
            name: [float(value) for value in values]
            for name, values in env.backend().features().items()
        }

    @staticmethod
    def isolate_evaluation_task(*, problem: Problem) -> None:
        """PickFood refusals outlive tasks on one instance (`pick_plannable.py`), as
        Pouring3D's kitchen pick refusals do; each evaluation task starts without them."""
        RackServingStepProtocol.env(problem=problem).forget_pick_refusals()

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        return {
            HUMAN_BATCH_RETURN: HUMAN_BATCH_RETURN,
            HUMAN_TOOL_RECOVERY: HUMAN_TOOL_RECOVERY,
        }

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(round(task.initial_state.get(obj=Objects.scene, feature_name="seed")))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """Use the same live rack verdict as skill-step evaluation, not KINDER's shelf goal."""
        del task
        env = RackServingStepProtocol.env(problem=problem)
        if env.config.two_route_rewards:
            # The verdict is the bulk skill's own, taken when it returns
            # (`evaluation_solved_after_action`): no tick mid-skill scores the task or
            # stops the controller.
            return lambda: False
        backend = env.backend()
        return lambda: bool(backend.task_succeeded())

    @staticmethod
    def evaluation_solved_after_action(*, problem: Problem, action: Action) -> bool | None:
        """Under `two_route_rewards`, a Sweep or GatherAndSweep that has returned decides
        the task: solved when its own outcome holds (`served_verdict`)."""
        env = RackServingStepProtocol.env(problem=problem)
        bulk = {SKILL_IDS["Sweep"], SKILL_IDS["GatherAndSweep"]}
        if not (env.config.two_route_rewards and action.size and int(action.flat[0]) in bulk):
            return None
        return ("Served", ()) in env.bulk_verdict()

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        """Evaluation buffers no ticks; its next task resets the scene."""
        del problem, action

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        return RackServingStepProtocol.env(problem=problem).adopt_interrupted_skill(
            action=action, reason=reason
        )

    @staticmethod
    def evaluation_terminal_reason(*, problem: Problem, action: Action) -> str | None:
        """Under `bulk_transfer_only` a task's one Sweep or GatherAndSweep ends it."""
        env = RackServingStepProtocol.env(problem=problem)
        bulk = {SKILL_IDS["Sweep"], SKILL_IDS["GatherAndSweep"]}
        if env.config.bulk_transfer_only and action.size and int(action.flat[0]) in bulk:
            return "bulk_attempt_finished"
        return None

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return RackServingStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        """No rack-specific trace builder exists; shelf skill IDs are not interchangeable.

        Under `two_route_rewards` every robot event carries the plate count after it,
        so an evaluation task's reward by count can be read from its last event."""
        env = RackServingStepProtocol.env(problem=problem)
        if not env.config.two_route_rewards:
            return None
        return {"plate_count": env.backend().plate_count()}
