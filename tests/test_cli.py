from hitl_pmp.cli import ENVIRONMENTS, _parse_args, main
from hitl_pmp.environments.lightswitch.cli import LightSwitchCli


def test_environments_registry_contains_lightswitch() -> None:
    assert ENVIRONMENTS["lightswitch"] is LightSwitchCli


def test_parse_args_has_no_positional_arguments() -> None:
    args = _parse_args(argv=["--env", "lightswitch", "--num-test-tasks", "3"])
    assert args.env == "lightswitch"
    assert args.num_test_tasks == 3


def test_parse_args_exposes_both_global_and_environment_specific_flags() -> None:
    args = _parse_args(
        argv=["--seed", "7", "--num-test-tasks", "3", "--env", "lightswitch", "--grid-size", "5"]
    )
    assert args.seed == 7
    assert args.num_test_tasks == 3
    assert args.grid_size == 5


def test_main_runs_the_selected_environment_end_to_end() -> None:
    main(argv=["--env", "lightswitch", "--num-test-tasks", "4"])
