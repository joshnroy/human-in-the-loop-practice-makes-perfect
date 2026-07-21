import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.core.metrics.metrics import Metrics
from hitl_pmp.core.problem.problem import Problem
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.oracle.cli import SkillOracleCli


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """SkillOracleCli.run() delegates to LightSwitchCli.run_method(), which
    mutates shared ClassVar state as a side effect; snapshot and restore it
    around every test in this file, same as
    tests/environments/lightswitch/test_cli.py."""
    original_grid_size = LightSwitchEnvironment.grid_size
    original_seed = LightSwitchTasks.seed
    original_problem_env = getattr(Problem, "env", None)
    original_problem_tasks = getattr(Problem, "tasks", None)
    try:
        yield
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size
        LightSwitchTasks.set_seed(seed=original_seed)
        if original_problem_env is not None:
            Problem.env = original_problem_env
        if original_problem_tasks is not None:
            Problem.tasks = original_problem_tasks
        Metrics.reset()


def _build_parser() -> argparse.ArgumentParser:
    """Mimics hitl_pmp/cli.py's global flags plus --env lightswitch's own, so
    SkillOracleCli can be exercised the same way the real global CLI drives it."""
    from hitl_pmp.environments.lightswitch.cli import LightSwitchCli

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-test-tasks", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    LightSwitchCli.add_arguments(parser=parser)
    SkillOracleCli.add_arguments(parser=parser)
    return parser


def test_add_arguments_adds_no_flags_of_its_own() -> None:
    """SkillOracleMethod hardcodes Light Switch internally, so it needs nothing
    beyond --env lightswitch's own flags -- this is a no-op, unlike a future
    method-CLI with real hyperparameters."""
    before = argparse.ArgumentParser()
    after = argparse.ArgumentParser()
    SkillOracleCli.add_arguments(parser=after)
    assert [action.dest for action in after._actions] == [action.dest for action in before._actions]


def test_run_solves_every_sampled_task(*, capsys: pytest.CaptureFixture[str]) -> None:
    args = _build_parser().parse_args(["--num-test-tasks", "5"])
    SkillOracleCli.run(args=args)
    assert "success rate: 5/5 (100%)" in capsys.readouterr().out
