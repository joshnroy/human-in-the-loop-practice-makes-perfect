import argparse

import pytest

import hitl_pmp.cli as cli_module
from hitl_pmp.cli import ENVIRONMENTS, METHODS, Cli, MethodCli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.methods.oracle.cli import SkillOracleCli


class _FakeMethodCli(MethodCli):
    run_calls: list[argparse.Namespace] = []

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fake-flag", type=int, default=0)

    @staticmethod
    def run(*, args: argparse.Namespace) -> None:
        _FakeMethodCli.run_calls.append(args)


@pytest.fixture
def _registered_fake_method(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """Registers an extra fake entry into METHODS (alongside the real
    "skill-oracle") for the duration of a test, so the --method discovery/
    dispatch mechanism itself can be exercised end to end without depending on
    a real Method's own argparse flags or run() behavior."""
    _FakeMethodCli.run_calls = []
    monkeypatch.setitem(cli_module.METHODS, "fake-method", _FakeMethodCli)


def test_environments_registry_contains_lightswitch() -> None:
    assert ENVIRONMENTS["lightswitch"] is LightSwitchCli


def test_methods_registry_contains_skill_oracle() -> None:
    assert METHODS["skill-oracle"] is SkillOracleCli


def test_parse_args_has_no_positional_arguments() -> None:
    args = Cli.parse_args(
        argv=["--env", "lightswitch", "--method", "skill-oracle", "--num-test-tasks", "3"]
    )
    assert args.env == "lightswitch"
    assert args.method == "skill-oracle"
    assert args.num_test_tasks == 3


def test_parse_args_exposes_both_global_and_environment_specific_flags() -> None:
    args = Cli.parse_args(
        argv=[
            "--seed",
            "7",
            "--num-test-tasks",
            "3",
            "--env",
            "lightswitch",
            "--method",
            "skill-oracle",
            "--grid-size",
            "5",
        ]
    )
    assert args.seed == 7
    assert args.num_test_tasks == 3
    assert args.grid_size == 5


def test_parse_args_rejects_a_non_positive_num_test_tasks() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=["--env", "lightswitch", "--method", "skill-oracle", "--num-test-tasks", "0"]
        )


def test_parse_args_help_after_env_shows_environment_specific_flags(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "skill-oracle", "--help"])
    assert "--grid-size" in capsys.readouterr().out


def test_parse_args_requires_env() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--method", "skill-oracle"])


def test_parse_args_requires_method() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch"])


def test_main_runs_the_selected_method_end_to_end() -> None:
    Cli.main(argv=["--env", "lightswitch", "--method", "skill-oracle", "--num-test-tasks", "4"])


def test_parse_args_rejects_an_unregistered_method_choice() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "not-a-real-method"])


@pytest.mark.usefixtures("_registered_fake_method")
def test_parse_args_exposes_method_specific_flags_once_method_is_known() -> None:
    args = Cli.parse_args(
        argv=["--env", "lightswitch", "--method", "fake-method", "--fake-flag", "3"]
    )
    assert args.method == "fake-method"
    assert args.fake_flag == 3


@pytest.mark.usefixtures("_registered_fake_method")
def test_parse_args_help_after_method_shows_method_specific_flags(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "fake-method", "--help"])
    assert "--fake-flag" in capsys.readouterr().out


@pytest.mark.usefixtures("_registered_fake_method")
def test_main_dispatches_to_the_selected_methods_own_run() -> None:
    Cli.main(argv=["--env", "lightswitch", "--method", "fake-method"])
    assert len(_FakeMethodCli.run_calls) == 1
