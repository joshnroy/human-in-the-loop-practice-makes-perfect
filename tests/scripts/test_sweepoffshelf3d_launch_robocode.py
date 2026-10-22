"""The launcher's RoboCode (Full-Agentic, LLLL) arm: agni-only, one seed, explicit hours."""

from pathlib import Path

import pytest

from hitl_pmp.environments.sweepoffshelf3d.tasks import SweepOffShelf3DTasks
from hitl_pmp.full_agentic.domains import SweepOffShelf3DFullAgentic
from hitl_pmp.full_agentic.runner import FullAgenticRunner
from scripts.sweepoffshelf3d_launch import (
    DRY,
    FULL,
    REPO,
    ROBOCODE,
    ROBOCODE_COSTS,
    ShelfLaunch,
    main,
)

RUNNER = ["python", "-m", "hitl_pmp.full_agentic.runner"]


def _value(*, argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _argv(**overrides: object) -> list[str]:
    kwargs: dict[str, object] = dict(
        size=FULL,
        seed=0,
        output=Path("/r/robocode/full_agentic/0"),
        sandbox_settings=Path("/r/sandbox.json"),
        hours=6.0,
        model_budget=20.0,
    )
    kwargs.update(overrides)
    return ShelfLaunch.robocode_argv(**kwargs)


def _runner_args(*, argv: list[str]) -> list[str]:
    return argv[argv.index("hitl_pmp.full_agentic.runner") + 1 :]


@pytest.fixture
def on_agni(*, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ShelfLaunch, "hostname", lambda: "agni")


def test_the_argv_is_the_runner_module_with_the_decided_budgets() -> None:
    argv = _runner_args(argv=_argv())
    args = FullAgenticRunner.parse_args(argv=argv)
    assert args.env == "sweepoffshelf3d"
    assert (
        args.practice_step_budget,
        args.measurement_interval_steps,
        args.evaluation_control_steps,
        args.num_test_tasks,
    ) == (85000, 15000, 3000, 10)
    assert args.stop_after_perfect_evaluations == 3
    assert args.reset_cubes_success_probability == 1.0
    assert args.model_budget == 20
    assert args.hours == 6
    assert args.seed == 0
    assert args.sandbox_settings == Path("/r/sandbox.json")
    assert args.output == Path("/r/robocode/full_agentic/0")
    # The CLI's default effort: the arm never passes one.
    assert "--effort" not in argv


def test_the_runner_runs_as_a_module_through_the_token_loader() -> None:
    argv = _argv(token_file=Path("/home/x/.bashrc"))
    start = argv.index("python")
    assert argv[start : start + 3] == [
        "python",
        str(REPO / "scripts/with_claude_token.py"),
        "/home/x/.bashrc",
    ]
    assert argv[start + 3 : start + 6] == ["-m", "hitl_pmp.full_agentic.runner", "--env"]


def test_the_cost_file_is_the_planner_arms_reset_500_file() -> None:
    # PR #528's file: reset_cubes and reset_wiper cost 500 and take 1 step.
    assert ROBOCODE_COSTS == REPO / "configs/sweepoffshelf3d_practice_costs_planner.json"
    argv = _runner_args(argv=_argv())
    assert _value(argv=argv, flag="--practice-cost-config") == str(ROBOCODE_COSTS)


def test_the_dry_size_shrinks_only_the_budgets() -> None:
    args = FullAgenticRunner.parse_args(argv=_runner_args(argv=_argv(size=DRY, model_budget=3.0)))
    assert (
        args.practice_step_budget,
        args.measurement_interval_steps,
        args.evaluation_control_steps,
        args.num_test_tasks,
    ) == (1500, 1500, 300, 1)
    assert args.model_budget == 3


def test_the_arm_refuses_to_build_without_hours() -> None:
    with pytest.raises(SystemExit, match="--hours"):
        _argv(hours=None)


def test_the_launcher_has_no_default_hours() -> None:
    assert ShelfLaunch.parse(argv=["print", "--arms", ROBOCODE]).hours is None
    with pytest.raises(SystemExit, match="--hours"):
        main(argv=["print", "--arms", ROBOCODE])


def test_print_shows_the_hours_it_was_given(*, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv=["print", "--arms", ROBOCODE, "--hours", "6"]) == 0
    out = capsys.readouterr().out
    assert "--hours 6" in out and "hitl_pmp.full_agentic.runner" in out


def test_robocode_is_one_seed_only() -> None:
    assert ShelfLaunch.robocode_seeds() == [0]


def test_robocode_is_not_in_the_default_plan() -> None:
    assert ROBOCODE not in ShelfLaunch.arms(expectimax_depths=[6])


def test_della_never_emits_robocode(*, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="agni"):
        main(argv=["della", "--tag", "t", "--out", str(tmp_path), "--arms", ROBOCODE])
    assert not list(tmp_path.iterdir())


def test_launch_refuses_off_agni(*, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ShelfLaunch, "hostname", lambda: "della-r1c1")
    with pytest.raises(SystemExit, match="agni"):
        ShelfLaunch.robocode_service(
            argv=_argv(), unit="u", results_root=tmp_path, working_directory=tmp_path
        )


def test_launch_is_a_user_service_in_agents_slice_without_swap(
    *, on_agni: None, tmp_path: Path
) -> None:
    del on_agni
    command = ShelfLaunch.robocode_service(
        argv=_argv(), unit="hitl-shelf-robocode-s0", results_root=tmp_path, working_directory=REPO
    )
    assert command[:3] == ["systemd-run", "--user", "--unit=hitl-shelf-robocode-s0"]
    assert "--slice=agents.slice" in command
    assert "MemorySwapMax=0" in command
    assert f"--working-directory={REPO}" in command
    assert command[command.index("--") + 1 :] == _argv()


def test_launch_robocode_refuses_without_go(*, on_agni: None, tmp_path: Path) -> None:
    del on_agni
    with pytest.raises(SystemExit, match="--go"):
        main(
            argv=[
                "launch", "--arms", ROBOCODE, "--hours", "6", "--results-root", str(tmp_path),
                "--sandbox-settings", str(tmp_path / "s.json"),
            ]
        )  # fmt: skip


def test_robocode_cannot_be_launched_with_the_planner_arms(*, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="on its own"):
        main(argv=["print", "--arms", ROBOCODE, "ees", "--hours", "6"])


def test_held_out_tasks_match_the_planner_arms() -> None:
    """Ten tasks from seed 0's test stream, offset 10000, as `--test-env-seed-offset
    10000` gives every planner arm."""
    from scripts.sweepoffshelf3d_launch import OCT13

    assert _value(argv=OCT13, flag="--test-env-seed-offset") == "10000"
    assert SweepOffShelf3DTasks.model_fields["test_env_seed_offset"].default == 10000
    planner = ShelfLaunch.method_args(arm=ShelfLaunch.arms(expectimax_depths=[6])["ees"], size=FULL)
    args = FullAgenticRunner.parse_args(argv=_runner_args(argv=_argv()))
    assert args.num_test_tasks == int(_value(argv=planner, flag="--num-test-tasks")) == 10
    assert args.evaluation_control_steps == int(
        _value(argv=planner, flag="--evaluation-control-steps")
    )


def test_the_runner_draws_held_out_scenes_from_the_planners_stream(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: list[int] = []
    monkeypatch.setattr(
        SweepOffShelf3DTasks, "build_task", lambda self, *, scene_seed: drawn.append(scene_seed)
    )
    from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment

    env = SweepOffShelf3DEnvironment.model_construct()
    robocode = SweepOffShelf3DFullAgentic.make_tasks(env=env, seed=0)
    for _ in range(10):
        robocode.sample_test_task()
    planner = SweepOffShelf3DTasks.model_construct(env=env, seed=0, test_env_seed_offset=10000)
    planner.model_post_init(None)
    ours, drawn[:] = list(drawn), []
    for _ in range(10):
        planner.sample_test_task()
    assert ours == drawn and len(set(ours)) == 10
