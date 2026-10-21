"""The structural shapes the global CLI (`cli.py`) wires together: what an
environment-CLI (`ENVIRONMENTS` value) and a method-CLI (`METHODS` value) each must
expose. Kept in their own module, below both `environments/` and `methods/` in the
import layering, so a method-CLI can be typed against `EnvironmentCli` (it drives the
selected env's `run_method`) without importing `cli.py` -- which would be a cycle,
since `cli.py` imports every concrete method-CLI. Both are Protocols (structural), so
a concrete CLI satisfies them just by having matching staticmethods, no subclassing.
"""

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from hitl_pmp.core.method.method import Method
from hitl_pmp.core.method.skill_provider import DomainContext, SkillProvider
from hitl_pmp.core.problem.environment.types import Action, State
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.core.problem.tasks.types import Task
from hitl_pmp.recording.run_record.writer import RunRecordWriter


class EnvironmentCli(Protocol):
    """The shape every environments/<domain>/cli.py entry in `ENVIRONMENTS` must
    expose: `add_arguments` (cli.py registers the selected env's config flags) and
    `run_method` (its composition root -- builds env/tasks/problem plus the domain's
    SkillProvider/OraclePolicyProvider, then calls the method-CLI's method_factory)."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    def run_method(
        *,
        args: argparse.Namespace,
        method_factory: Callable[[DomainContext], Method],
        num_cycles: int,
        max_steps_per_interaction: int,
    ) -> None: ...


class MethodCli(Protocol):
    """The shape every methods/<name>/cli.py entry in `METHODS` must expose:
    `add_arguments` (its method-specific flags) and `run` (which drives the selected
    `env_cli.run_method` with a method_factory that builds this method from the
    env's DomainContext). A method-CLI never imports a specific environment -- it
    receives `env_cli` from cli.py, the one place allowed to import both sides."""

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None: ...

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: type[EnvironmentCli]) -> None: ...


class StepProtocolEnvironment(Protocol):
    """What the step-based practice protocol (`step_protocol.py`) needs from a domain,
    registered per `--env` in `cli.py`'s `STEP_PROTOCOL_ENVIRONMENTS`. Every method
    then shares one x-axis -- counted practice steps, i.e. robot control steps plus
    each human skill's duration -- one budget and one evaluation schedule.

    Beyond this, the domain's `Environment` must implement
    `set_control_step_observers`, and its practice `Problem` must carry a `human`
    whose `execute_movables_reset(destination=...)` performs each human skill.

    A static-method container like `EnvironmentCli`. It must be importable by
    reference (a module-level class), because each evaluation runs in a spawned
    process that receives this class, not an instance."""

    @staticmethod
    def build_practice_problem(*, args: argparse.Namespace) -> Problem: ...

    @staticmethod
    def build_evaluation_problem(*, args: argparse.Namespace) -> Problem:
        """Wholly separate from practice, with the same fixed test-task stream."""
        ...

    @staticmethod
    def practice_context(*, args: argparse.Namespace, problem: Problem) -> DomainContext:
        """The practice skill provider (human skills offered as configured) and oracle."""
        ...

    @staticmethod
    def resume_practice_environment(*, problem: Problem, state: State) -> None:
        """Carry a resumed run on from the scene as it is now, re-observed; `state`, the
        checkpoint's last, supplies only the episode bookkeeping."""
        ...

    @staticmethod
    def run_saved(*, args: argparse.Namespace, output: Path, final: bool) -> None:
        """A cycle checkpoint was just written to `output` (`final`: the run ended); the
        real robot copies the skill server's logs beside it."""
        ...

    @staticmethod
    def evaluation_pool() -> Any:
        """Where evaluations run: `None` for the protocol's default (a fresh spawned
        process per evaluation, beside practice); a domain whose evaluation cannot run
        beside practice (one real robot) returns a pool that runs it in place."""
        ...

    @staticmethod
    def evaluation_skill_provider(*, problem: Problem) -> SkillProvider:
        """Deployment offers no human skill."""
        ...

    @staticmethod
    def attach_recording(
        *, args: argparse.Namespace, problem: Problem, output_path: Path
    ) -> RunRecordWriter:
        """Start the run recording (`recording/run_record/`) at `output_path`, with the
        domain's live state as its tick source; the caller closes it. A domain that
        cannot replay its states returns `RunRecordWriter.disabled()`."""
        ...

    @staticmethod
    def human_skill_names() -> dict[str, str]:
        """Each `HumanCubeBinResetRequested.destination` this domain accepts, mapped to
        the human skill it invokes (the name costs and the event log use). The key
        order is the order of `human_resets_by_destination` in every record."""
        ...

    @staticmethod
    def task_seed(*, task: Task) -> int:
        """The identifier an evaluation records for each held-out task."""
        ...

    @staticmethod
    def goal_check(*, problem: Problem, task: Task) -> Callable[[], bool]:
        """Whether `task` is solved in the live scene, callable after any control
        step -- mid-skill, before `take_action` has returned a new `State`."""
        ...

    @staticmethod
    def interrupted(*, problem: Problem, action: Action) -> None:
        """`take_action(action)` was cut short by the control-step budget; flush what
        the domain buffers per action."""
        ...

    @staticmethod
    def adopt_interrupted_skill(*, problem: Problem, action: Action, reason: str) -> State:
        """`take_action(action)` was cut short during practice: record `reason` as the
        skill's outcome and adopt the simulator's actual state, which `take_action`
        never reached its own bookkeeping to do."""
        ...

    @staticmethod
    def last_skill_error(*, problem: Problem) -> str | None: ...

    @staticmethod
    def practice_progress(
        *, problem: Problem, cycle: int, action: int, max_actions: int, steps: int, budget: int
    ) -> None:
        """Practice's place as its next action starts: `action` of `max_actions` in
        session `cycle`, at `steps` of `budget`."""
        ...

    @staticmethod
    def evaluation_progress(
        *, problem: Problem, index: int, label: str, task: int, num_tasks: int
    ) -> None:
        """Evaluation `index` (`label`: which update it measures, e.g. `after cycle 3`)
        is at `task` of `num_tasks`, told before each task's scene and each action."""
        ...

    @staticmethod
    def practice_scene(*, problem: Problem) -> Any:
        """Practice's scene, as plain data, before an in-process evaluation uses the room
        (`--measure-every-cycles`)."""
        ...

    @staticmethod
    def restore_practice_scene(*, problem: Problem, scene: Any) -> dict[str, Any]:
        """Put `practice_scene`'s `scene` back after that evaluation -- the person's job,
        charged nothing -- and carry practice on from it as observed. Returns the
        `practice_scene_restored` event's fields."""
        ...

    @staticmethod
    def last_skill_trace(*, problem: Problem) -> dict[str, Any] | None:
        """A compact, JSON-ready account of the last robot skill -- its outcome and
        every candidate it rejected -- logged as the step event's `skill_trace`; None
        for a domain that keeps none (the event then carries no such field)."""
        ...


class SimulatedStepProtocolHooks:
    """`StepProtocolEnvironment`'s real-robot hooks, for a simulated domain: nobody
    watches its progress, and its evaluations run in their own process and scene, so
    practice's scene is never disturbed (the flags that need one are refused first)."""

    @staticmethod
    def practice_progress(
        *, problem: Problem, cycle: int, action: int, max_actions: int, steps: int, budget: int
    ) -> None:
        del problem, cycle, action, max_actions, steps, budget

    @staticmethod
    def evaluation_progress(
        *, problem: Problem, index: int, label: str, task: int, num_tasks: int
    ) -> None:
        del problem, index, label, task, num_tasks

    @staticmethod
    def practice_scene(*, problem: Problem) -> Any:
        del problem
        raise NotImplementedError("--measure-every-cycles is for --env tossing3d_real only")

    @staticmethod
    def restore_practice_scene(*, problem: Problem, scene: Any) -> dict[str, Any]:
        del problem, scene
        raise NotImplementedError("--measure-every-cycles is for --env tossing3d_real only")

    @staticmethod
    def resume_practice_environment(*, problem: Problem, state: State) -> None:
        """A fresh process has a fresh simulator: no scene is left to re-observe."""
        del problem, state
        raise NotImplementedError("--resume is supported for --env tossing3d_real only")

    @staticmethod
    def run_saved(*, args: argparse.Namespace, output: Path, final: bool) -> None:
        del args, output, final
