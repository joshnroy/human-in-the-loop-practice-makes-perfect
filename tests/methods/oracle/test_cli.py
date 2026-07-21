import argparse
from collections.abc import Iterator
from pathlib import Path

import pytest

from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.methods.oracle.cli import SkillOracleCli


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """SkillOracleCli.run() delegates to LightSwitchCli.run_method(), which still
    mutates LightSwitchEnvironment.light_on_tolerance/.same_position_tolerance as a
    ClassVar side effect via apply_config (see that method's own docstring for why
    those two specifically stay ClassVar rather than becoming constructor
    arguments). Everything else run_method touches -- env/tasks/problem/method -- is
    now a freshly constructed instance per call, with nothing left over to restore."""
    original_light_on_tolerance = LightSwitchEnvironment.light_on_tolerance
    original_same_position_tolerance = LightSwitchEnvironment.same_position_tolerance
    try:
        yield
    finally:
        LightSwitchEnvironment.light_on_tolerance = original_light_on_tolerance
        LightSwitchEnvironment.same_position_tolerance = original_same_position_tolerance


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
