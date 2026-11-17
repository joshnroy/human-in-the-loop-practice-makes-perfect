"""Serving's hooks for the control-step practice runner: the toy's problem, provider,
human reset and one-attempt evaluation over the 3D scene."""

import argparse
import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

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
from .skill_provider import RackServingSkillProvider
from .toy_environment import TOY_SKILL_NAMES, RackServingToyEnvironment
from .toy_interface import GATHER, HUMAN_RESET, SWEEP3, SWEEP5, ServingToyTransitions


class RackServingStepProtocol(SimulatedStepProtocolHooks):
    """A spawn-importable adapter; all dynamics remain in the rack environment."""

    @staticmethod
    def env(*, problem: Problem) -> RackServingToyEnvironment:
        assert isinstance(problem.env, RackServingToyEnvironment)
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
        return {HUMAN_RESET: HUMAN_RESET}

    @staticmethod
    def task_seed(*, task: Task) -> int:
        return int(round(task.initial_state.get(obj=Objects.scene, feature_name="seed")))

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """`Served`, the toy's goal: already held, or measured as the running sweep3 or
        sweep5 lands its count. The step protocol then stops the skill, and its
        transition is taken from the scene as it stands (`take_action`)."""
        del task
        env = RackServingStepProtocol.env(problem=problem)

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
        return "Served" in RackServingStepProtocol.env(problem=problem).toy_flags()

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
        del problem
        return RackServingStepProtocol.one_shot_failure_reason(action=action)

    @staticmethod
    def one_shot_failure_reason(*, action: Action) -> str | None:
        """A route skill's attempt is the task's one attempt, as in the toy."""
        values = np.asarray(action, dtype=float)
        if values.shape != (5,) or not np.isfinite(values[0]):
            return None
        skill = TOY_SKILL_NAMES.get(int(values[0]))
        return "route_failed" if skill in (SWEEP3, GATHER, SWEEP5) else None

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None:
        return RackServingStepProtocol.env(problem=problem).last_skill_error()

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        """The toy state after the skill and the measured scene its outcome was read
        from, so every robot event says what the physics decided."""
        env = RackServingStepProtocol.env(problem=problem)
        measured = env.toy_measurement()
        return {
            "toy_state": sorted(env.toy_flags()),
            "measured": None if measured is None else measured.model_dump(),
        }
