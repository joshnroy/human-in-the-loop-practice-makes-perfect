import argparse
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.problem import LightSwitchProblem
from hitl_pmp.environments.lightswitch.renderer import LightSwitchRenderer
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.method_runner import MethodRunner
from hitl_pmp.methods.oracle.skill_oracle_method import SkillOracleMethod


def _args(*, num_test_tasks: int = 5, output_dir: Path | None = None) -> argparse.Namespace:
    return argparse.Namespace(num_test_tasks=num_test_tasks, output_dir=output_dir)


def _build_problem() -> LightSwitchProblem:
    env = LightSwitchEnvironment()
    return LightSwitchProblem(env=env, tasks=LightSwitchTasks(env=env))


def test_run_prints_success_rate(*, capsys: pytest.CaptureFixture[str]) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(num_test_tasks=5),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out


def test_run_records_one_evaluation_per_cycle_plus_the_initial_one() -> None:
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=3),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=None,
        render_fps=2,
    )
    # num_cycles/max_steps_per_interaction are forwarded to PracticeLoop.run, not
    # hardcoded inside MethodRunner -- one initial evaluation plus one per cycle.
    assert len(metrics.evaluations) == 3


def test_run_without_output_dir_writes_no_files(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    assert list(tmp_path.iterdir()) == []


def test_run_without_output_dir_writes_no_stats_json(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=None),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert not (tmp_path / "stats.json").exists()


def test_run_with_output_dir_and_renderer_writes_a_video_file(*, tmp_path: Path) -> None:
    problem = _build_problem()
    MethodRunner.run(
        args=_args(output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    video_path = tmp_path / "episode.mp4"
    assert video_path.exists()
    assert video_path.stat().st_size > 0


def test_run_with_output_dir_writes_stats_json_that_round_trips(*, tmp_path: Path) -> None:
    problem = _build_problem()
    metrics = MethodRunner.run(
        args=_args(num_test_tasks=3, output_dir=tmp_path),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=2,
        max_steps_per_interaction=2,
        renderer=LightSwitchRenderer,
        render_fps=2,
    )
    stats_path = tmp_path / "stats.json"
    assert stats_path.exists()

    loaded = Metrics.model_validate_json(stats_path.read_text())
    assert loaded == metrics
    assert loaded.evaluations == metrics.evaluations
    assert loaded.task_name == metrics.task_name


def test_run_does_not_leak_evaluations_between_calls() -> None:
    """MethodRunner constructs a fresh Metrics() per call now (see its own
    docstring) -- there's no reset() step to forget, and no shared state two
    back-to-back calls could leak through."""
    problem = _build_problem()
    first = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(first.evaluations) == 1

    second = MethodRunner.run(
        args=_args(num_test_tasks=2),
        method=SkillOracleMethod(env=problem.env),
        problem=problem,
        num_cycles=0,
        max_steps_per_interaction=0,
        renderer=None,
        render_fps=2,
    )
    assert len(second.evaluations) == 1
    assert first is not second
