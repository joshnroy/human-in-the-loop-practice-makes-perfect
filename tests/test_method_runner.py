import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


@pytest.fixture(autouse=True)
def _wire_lightswitch() -> Iterator[None]:
    original_problem_env = getattr(Problem, "env", None)
    original_problem_tasks = getattr(Problem, "tasks", None)
    Problem.env = LightSwitchEnvironment
    Problem.tasks = LightSwitchTasks
    try:
        yield
    finally:
        if original_problem_env is not None:
            Problem.env = original_problem_env
        if original_problem_tasks is not None:
            Problem.tasks = original_problem_tasks
        Metrics.reset()


def _args(*, num_test_tasks: int = 5, output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(num_test_tasks=num_test_tasks, output_dir=output_dir)


def test_run_prints_success_rate(*, capsys: pytest.CaptureFixture[str]) -> None:
    MethodRunner.run(
        args=_args(num_test_tasks=5),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_records_one_evaluation_per_cycle_plus_the_initial_one() -> None:
    MethodRunner.run(
        args=_args(num_test_tasks=3),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    # num_cycles/max_steps_per_interaction are forwarded to PracticeLoop.run, not
    # hardcoded inside MethodRunner -- one initial evaluation plus one per cycle.
    assert len(Metrics.evaluations) == 3


def test_run_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_with_output_dir_and_renderer_writes_a_video_file(*, tmp_path: Path) -> None:
    MethodRunner.run(
        args=_args(output_dir=tmp_path),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_resets_metrics_so_evaluations_dont_leak_between_calls() -> None:
    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(Metrics.evaluations) == 1

    MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod,
        problem=LightSwitchProblem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(Metrics.evaluations) == 1
