from collections.abc import Iterator

import pytest

from hitl_pmp.cli import ENVIRONMENTS, METHODS, Cli
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli
from hitl_pmp.environments.lightswitch.environment import LightSwitchEnvironment
from hitl_pmp.environments.lightswitch.tasks import LightSwitchTasks
from hitl_pmp.methods.practice_makes_perfect.cli import RandomSkillsCli


@pytest.fixture(autouse=True)
def _restore_lightswitch_config() -> Iterator[None]:
    """Cli.main(--method ...) mutates shared LightSwitch/RandomSkillsMethod
    ClassVar state as a side effect (same reason
    tests/environments/lightswitch/test_cli.py restores it); snapshot/restore
    around every test in this file too."""
    original_grid_size = LightSwitchEnvironment.grid_size
    original_seed = LightSwitchTasks.seed
    try:
        yield
    finally:
        LightSwitchEnvironment.grid_size = original_grid_size
        LightSwitchTasks.set_seed(seed=original_seed)


def test_environments_registry_contains_lightswitch() -> None:
    assert ENVIRONMENTS["lightswitch"] is LightSwitchCli


def test_methods_registry_contains_random_skills() -> None:
    assert METHODS["random-skills"] is RandomSkillsCli


def test_parse_args_has_no_positional_arguments() -> None:
    args = Cli.parse_args(argv=["--env", "lightswitch", "--num-test-tasks", "3"])
    assert args.env == "lightswitch"
    assert args.num_test_tasks == 3


def test_parse_args_exposes_both_global_and_environment_specific_flags() -> None:
    args = Cli.parse_args(
        argv=["--seed", "7", "--num-test-tasks", "3", "--env", "lightswitch", "--grid-size", "5"]
    )
    assert args.seed == 7
    assert args.num_test_tasks == 3
    assert args.grid_size == 5


def test_parse_args_rejects_a_non_positive_num_test_tasks() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--num-test-tasks", "0"])


def test_parse_args_help_after_env_shows_environment_specific_flags(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--help"])
    assert "--grid-size" in capsys.readouterr().out


def test_main_runs_the_selected_environment_end_to_end() -> None:
    Cli.main(argv=["--env", "lightswitch", "--num-test-tasks", "4"])


def test_parse_args_exposes_method_specific_flags_once_method_is_known() -> None:
    args = Cli.parse_args(
        argv=["--env", "lightswitch", "--method", "random-skills", "--num-cycles", "2"]
    )
    assert args.method == "random-skills"
    assert args.num_cycles == 2


def test_parse_args_rejects_an_unknown_method_choice() -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "not-a-real-method"])


def test_parse_args_help_after_method_shows_method_specific_flags(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        Cli.parse_args(argv=["--env", "lightswitch", "--method", "random-skills", "--help"])
    assert "--num-cycles" in capsys.readouterr().out


def test_main_with_method_runs_random_skills_end_to_end() -> None:
    Cli.main(
        argv=[
            "--env",
            "lightswitch",
            "--method",
            "random-skills",
            "--grid-size",
            "5",
            "--num-cycles",
            "1",
            "--steps-per-cycle",
            "2",
            "--num-test-tasks",
            "2",
        ]
    )


def test_main_without_method_still_runs_the_environments_own_policy_path() -> None:
    # No --method: dispatches to ENVIRONMENTS[env].run, exactly as before --method
    # existed -- confirms the new dispatch branch didn't change this default path.
    Cli.main(argv=["--env", "lightswitch", "--num-test-tasks", "2"])
