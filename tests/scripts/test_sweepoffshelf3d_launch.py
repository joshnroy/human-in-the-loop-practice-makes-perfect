import json
import shlex
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from scripts.run_sweep import SweepRunner
from scripts.sweepoffshelf3d_launch import (
    ALWAYS_ASK_COSTS,
    COSTS,
    FULL,
    REPO,
    ShelfLaunch,
    main,
)

CANONICAL_ARMS = ["ees", "cccc-determinized", "cccc-expectimax-d6", "cccc-expectimax-d7"]
ALWAYS_ASK_ARMS = [f"{arm}-always-ask" for arm in CANONICAL_ARMS]


def _value(*, argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _shared(*, command: list[str]) -> list[str]:
    return shlex.split(_value(argv=command, flag="--shared-args"))


@pytest.fixture
def always_ask_missing(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout from before #477, whose CLI has no `--always-ask-reset`."""
    monkeypatch.setattr(ShelfLaunch, "always_ask_supported", lambda *, method: False)


def test_the_decided_plan_is_eight_arms() -> None:
    arms = ShelfLaunch.arms(expectimax_depths=[6, 7])
    assert list(arms) == [*CANONICAL_ARMS, *ALWAYS_ASK_ARMS]
    assert {arm.method for arm in arms.values()} == {"ees", "pomdp"}
    assert [name for name, arm in arms.items() if arm.always_ask] == ALWAYS_ASK_ARMS


def test_the_default_depths_are_six_and_seven() -> None:
    assert ShelfLaunch.parse(argv=["print"]).expectimax_depths == [6, 7]


@pytest.mark.parametrize("depth", [5, 6, 7])
def test_the_expectimax_depth_is_a_flag_not_a_constant(*, depth: int) -> None:
    arms = ShelfLaunch.arms(expectimax_depths=[depth])
    assert [name for name in arms if "expectimax" in name] == [
        f"cccc-expectimax-d{depth}",
        f"cccc-expectimax-d{depth}-always-ask",
    ]
    argv = ShelfLaunch.cli_argv(
        arm=arms[f"cccc-expectimax-d{depth}"], size=FULL, seed=0, output_dir=Path("/x")
    )
    assert Cli.parse_args(argv=argv).pomdp_search_depth == depth


def test_the_determinized_arm_carries_no_depth() -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[7])["cccc-determinized"]
    assert "--pomdp-search-depth" not in arm.flags
    assert _value(argv=arm.flags, flag="--pomdp-solver") == "determinized_astar"


@pytest.mark.parametrize("name", [*CANONICAL_ARMS, *ALWAYS_ASK_ARMS])
def test_every_arm_parses_to_the_decided_settings(*, name: str) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[6, 7])[name]
    argv = ShelfLaunch.cli_argv(arm=arm, size=FULL, seed=3, output_dir=Path("/out"))
    args = Cli.parse_args(argv=argv)
    assert args.env == "sweepoffshelf3d"
    assert args.seed == 3
    assert args.practice_step_budget == 85000
    assert args.measurement_interval_steps == 15000
    assert args.evaluation_control_steps == 3000
    assert args.num_test_tasks == 10
    assert args.stop_after_perfect_evaluations == 3
    assert args.reset_cubes_success_probability == 1.0
    assert args.practice_cost_config == (ALWAYS_ASK_COSTS if arm.always_ask else COSTS)
    assert args.always_ask_reset is arm.always_ask
    assert args.scene_bg is True  # the lab2 scene
    assert args.skip_skill_benchmark_gate is False
    if arm.method == "pomdp":
        assert args.pomdp_observation_probability_weight == 0.0
        assert args.pomdp_competence_model == "local_trend"
    else:
        assert (args.ees_prior_alpha, args.ees_prior_beta) == (1.0, 3.0)


def test_p_is_one_and_the_plain_scene_is_never_requested() -> None:
    for arm in ShelfLaunch.arms(expectimax_depths=[6, 7]).values():
        shared = ShelfLaunch.method_args(arm=arm, size=FULL)
        assert _value(argv=shared, flag="--reset-cubes-success-probability") == "1"
        assert "--no-scene-bg" not in shared


def test_the_sweep_runs_seeds_zero_to_five_into_arm_method_seed(*, tmp_path: Path) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[7])["cccc-expectimax-d7"]
    command = ShelfLaunch.sweep_command(
        arm=arm, size=FULL, results_root=tmp_path, max_workers=3, skip_gate=False
    )
    assert command[1:3] == ["-m", "scripts.run_sweep"]
    assert _value(argv=command, flag="--num-seeds") == "6"
    assert _value(argv=command, flag="--methods") == "pomdp"
    runs = SweepRunner.plan(
        env=_value(argv=command, flag="--env"),
        methods=[_value(argv=command, flag="--methods")],
        seeds=SweepRunner.default_seeds(num_seeds=int(_value(argv=command, flag="--num-seeds"))),
        results_root=Path(_value(argv=command, flag="--results-root")),
        shared_args=_shared(command=command),
        method_args={},
    )
    assert [run.output_dir for run in runs] == [
        tmp_path / "cccc-expectimax-d7" / "pomdp" / str(seed) for seed in range(6)
    ]
    assert Cli.parse_args(argv=runs[0].command[3:]).pomdp_observation_probability_weight == 0.0


def test_the_gate_is_never_skipped_by_default(*, tmp_path: Path) -> None:
    assert ShelfLaunch.parse(argv=["launch"]).skip_skill_benchmark_gate is False
    for arm in ShelfLaunch.arms(expectimax_depths=[6, 7]).values():
        command = ShelfLaunch.sweep_command(
            arm=arm, size=FULL, results_root=tmp_path, max_workers=3, skip_gate=False
        )
        assert not any("skip-skill-benchmark-gate" in token for token in command)


def test_skipping_the_gate_is_an_explicit_flag(*, tmp_path: Path) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[7])["ees"]
    command = ShelfLaunch.sweep_command(
        arm=arm, size=FULL, results_root=tmp_path, max_workers=3, skip_gate=True
    )
    assert command[-1] == "--skip-skill-benchmark-gate"
    assert ShelfLaunch.parse(
        argv=["launch", "--skip-skill-benchmark-gate"]
    ).skip_skill_benchmark_gate


@pytest.mark.usefixtures("always_ask_missing")
@pytest.mark.parametrize("name", ALWAYS_ASK_ARMS)
def test_an_always_ask_arm_refuses_while_the_flag_is_missing(*, name: str) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[6, 7])[name]
    with pytest.raises(SystemExit, match=r"--always-ask-reset.*#477"):
        ShelfLaunch.require_runnable(arm=arm)


@pytest.mark.usefixtures("always_ask_missing")
def test_print_refuses_an_always_ask_arm_while_the_flag_is_missing(
    *, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit, match="ees-always-ask"):
        main(argv=["print", "--arms", "ees-always-ask"])
    assert capsys.readouterr().out == ""


@pytest.mark.usefixtures("always_ask_missing")
def test_canonical_arms_do_not_depend_on_the_always_ask_flag() -> None:
    for name in CANONICAL_ARMS:
        ShelfLaunch.require_runnable(arm=ShelfLaunch.arms(expectimax_depths=[6, 7])[name])


def test_main_accepts_the_always_ask_flag_so_no_arm_refuses() -> None:
    assert ShelfLaunch.always_ask_supported(method="ees")
    assert ShelfLaunch.always_ask_supported(method="pomdp")
    for arm in ShelfLaunch.arms(expectimax_depths=[6, 7]).values():
        ShelfLaunch.require_runnable(arm=arm)


def test_an_always_ask_arm_adds_the_flag_and_the_sampled_start_costs() -> None:
    arms = ShelfLaunch.arms(expectimax_depths=[6, 7])
    for name in CANONICAL_ARMS:
        plain = ShelfLaunch.method_args(arm=arms[name], size=FULL)
        asked = ShelfLaunch.method_args(arm=arms[f"{name}-always-ask"], size=FULL)
        assert arms[f"{name}-always-ask"].method == arms[name].method
        assert "--always-ask-reset" in asked and "--always-ask-reset" not in plain
        assert _value(argv=asked, flag="--practice-cost-config") == str(ALWAYS_ASK_COSTS)
        rest = [t for t in asked if t not in {"--always-ask-reset", str(ALWAYS_ASK_COSTS)}]
        assert rest == [t for t in plain if t != str(COSTS)]


def test_launch_refuses_without_go(*, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="--go"):
        main(argv=["launch", "--arms", "ees", "--results-root", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_the_cost_files_are_tracked_and_price_the_stage5_accounting() -> None:
    assert COSTS.parent == ALWAYS_ASK_COSTS.parent == REPO / "configs"
    canonical = json.loads(COSTS.read_text())
    sampled = json.loads(ALWAYS_ASK_COSTS.read_text())
    assert canonical["objective_lambda"] == sampled["objective_lambda"] == 3e-6
    assert canonical["robot_step"] == {"value": 1}
    assert set(canonical["human_skills"]) == {"reset_cubes", "reset_wiper"}
    assert set(sampled["human_skills"]) == {"reset_cubes", "reset_wiper", "reset_to_sampled_start"}


def test_dependencies_outside_the_checkout_are_named(*, tmp_path: Path) -> None:
    inside = tmp_path / "reference" / "kindergarden" / "src" / "kinder" / "__init__.py"
    outside = Path("/elsewhere/kinder_models/__init__.py")
    assert ShelfLaunch.outside(repo=tmp_path, files={"kinder": inside}) == []
    assert ShelfLaunch.outside(repo=tmp_path, files={"kinder": inside, "km": outside}) == [
        f"km: {outside}"
    ]
