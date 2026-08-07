import argparse
from pathlib import Path

import pytest

import hitl_pmp.cli as cli_module
from hitl_pmp.cli import ENVIRONMENTS, METHODS, Cli, MethodCli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.tossingroomsplit.cli import TossingRoomSplitCli
from hitl_pmp.methods.oracle.cli import SkillOracleCli
from hitl_pmp.practice_loop import PracticeResetPolicy


class _FakeMethodCli(MethodCli):
    run_calls: list[argparse.Namespace] = []

    @staticmethod
    def add_arguments(*, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--fake-flag", type=int, default=0)

    @staticmethod
    def run(*, args: argparse.Namespace, env_cli: object) -> None:
        # env_cli is accepted to match the MethodCli protocol's run() signature
        # (cli.py hands the selected env-CLI to the method-CLI); this fake only
        # exercises dispatch, so it records the call and ignores env_cli.
        del env_cli
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


def test_main_runs_tossingroom_skill_oracle_end_to_end() -> None:
    Cli.main(
        argv=[
            "--env",
            "tossingroom",
            "--method",
            "skill-oracle",
            "--num-test-tasks",
            "4",
            "--goal-type",
            "recycling",
        ]
    )


def test_main_runs_tossingroomsplit_skill_oracle_end_to_end() -> None:
    """The split-throw domain is reachable by name from the global CLI, on the same flag
    set as `tossingroom` -- which is what lets one `scripts/run_sweep.py` command target
    either domain."""
    Cli.main(
        argv=[
            "--env",
            "tossingroomsplit",
            "--method",
            "skill-oracle",
            "--num-test-tasks",
            "4",
            "--goal-type",
            "recycling",
        ]
    )


def test_tossingroomsplit_registered_under_its_own_name() -> None:
    assert ENVIRONMENTS["tossingroomsplit"] is TossingRoomSplitCli
    assert ENVIRONMENTS["tossingroomsplit"] is not ENVIRONMENTS["tossingroom"]


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


def test_parse_args_defaults_practice_reset_interval_to_none() -> None:
    """None is "reset only at the cycle boundary", i.e. exactly the behaviour that
    predates the flag -- every run that does not ask for it is unchanged."""
    args = Cli.parse_args(argv=["--env", "tossingroom", "--method", "ees"])
    assert args.practice_reset_interval is None


def test_parse_args_accepts_a_practice_reset_interval() -> None:
    args = Cli.parse_args(
        argv=["--env", "tossingroom", "--method", "ees", "--practice-reset-interval", "10"]
    )
    assert args.practice_reset_interval == 10


def test_parse_args_rejects_a_non_positive_practice_reset_interval() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=["--env", "tossingroom", "--method", "ees", "--practice-reset-interval", "0"]
        )


def test_parse_args_defaults_practice_reset_policy_to_scheduled() -> None:
    """'scheduled' is the behaviour that predates the flag -- one reset at the top
    of every period -- so every run that does not ask for it is unchanged."""
    args = Cli.parse_args(argv=["--env", "tossingroom", "--method", "ees"])
    assert args.practice_reset_policy is PracticeResetPolicy.SCHEDULED


def test_parse_args_accepts_never_as_a_practice_reset_policy() -> None:
    args = Cli.parse_args(
        argv=["--env", "tossingroomsplit", "--method", "ees", "--practice-reset-policy", "never"]
    )
    assert args.practice_reset_policy is PracticeResetPolicy.NEVER


def test_parse_args_rejects_an_unknown_practice_reset_policy() -> None:
    """The arm names are the experiment's own vocabulary; a typo must fail loudly
    rather than fall back to the default and quietly run the wrong arm."""
    with pytest.raises(SystemExit):
        Cli.parse_args(
            argv=["--env", "tossingroom", "--method", "ees", "--practice-reset-policy", "sometimes"]
        )


def test_parse_args_defaults_record_full_loop_to_none() -> None:
    """Off unless asked for: a run that does not pass it is unchanged, right down
    to taking no rendering path at all."""
    args = Cli.parse_args(argv=["--env", "tossingroom", "--method", "ees"])
    assert args.record_full_loop is None


def test_parse_args_accepts_a_record_full_loop_path() -> None:
    args = Cli.parse_args(
        argv=[
            "--env",
            "tossingroom",
            "--method",
            "ees",
            "--record-full-loop",
            "/tmp/loop.mp4",
        ]
    )
    assert args.record_full_loop == Path("/tmp/loop.mp4")


def test_main_records_a_full_loop_end_to_end(*, tmp_path: Path) -> None:
    """The flag is wired all the way through the real CLI (not only MethodRunner):
    one video covering the baseline sweep, the practice period and the post-cycle
    sweep."""
    output = tmp_path / "loop.mp4"
    Cli.main(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "random-skills",
            "--num-test-tasks",
            "1",
            "--num-cycles",
            "1",
            "--max-steps-per-interaction",
            "2",
            "--record-full-loop",
            str(output),
        ]
    )
    assert output.exists()
