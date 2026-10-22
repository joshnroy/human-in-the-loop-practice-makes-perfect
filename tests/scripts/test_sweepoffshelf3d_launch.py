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
DEFAULT_CANONICAL_ARMS = ["ees", "cccc-determinized", "cccc-expectimax-d6"]
ALWAYS_ASK_ARMS = [f"{arm}-always-ask" for arm in CANONICAL_ARMS]


def _value(*, argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _shared(*, command: list[str]) -> list[str]:
    return shlex.split(_value(argv=command, flag="--shared-args"))


@pytest.fixture
def always_ask_missing(*, monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout from before #477, whose CLI has no `--always-ask-reset`."""
    monkeypatch.setattr(ShelfLaunch, "always_ask_supported", lambda *, method: False)


def test_depths_six_and_seven_give_eight_arms() -> None:
    arms = ShelfLaunch.arms(expectimax_depths=[6, 7])
    assert list(arms) == [*CANONICAL_ARMS, *ALWAYS_ASK_ARMS]
    assert {arm.method for arm in arms.values()} == {"ees", "pomdp"}
    assert [name for name, arm in arms.items() if arm.always_ask] == ALWAYS_ASK_ARMS


def test_the_default_depth_is_six_only() -> None:
    # Josh, 2026-10-20: "let's drop d7"; depth 7 stays selectable by flag.
    assert ShelfLaunch.parse(argv=["print"]).expectimax_depths == [6]


def test_the_default_plan_is_six_arms() -> None:
    depths = ShelfLaunch.parse(argv=["print"]).expectimax_depths
    assert list(ShelfLaunch.arms(expectimax_depths=depths)) == [
        *DEFAULT_CANONICAL_ARMS,
        *(f"{arm}-always-ask" for arm in DEFAULT_CANONICAL_ARMS),
    ]


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
        # A deviation from predicators (whose default is True): a skill can measure 1.0
        # after a few samples, and skip_perfect would then treat it as mastered.
        assert args.reproduce_predicators_skip_perfect is False
        assert "--reproduce-predicators-skip-perfect" not in argv


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


DELLA = "/scratch/gpfs/TSILVER/jr2860"
SHA = "0123456789abcdef0123456789abcdef01234567"
GROUPS = ["ees", "determinized", "expectimax"]


def _della(
    *, tmp_path: Path, extra: list[str] | None = None
) -> dict[str, tuple[list[list[str]], str]]:
    """Each array group's manifest rows and sbatch text."""
    out = tmp_path / "della"
    argv = ["della", "--tag", "shelf-20261020", "--sha", SHA, "--out", str(out), *(extra or [])]
    assert main(argv=argv) == 0
    groups = {}
    for manifest in sorted(out.glob("manifest-*.tsv")):
        group = manifest.stem.removeprefix("manifest-")
        rows = [line.split("\t") for line in manifest.read_text().splitlines() if line]
        groups[group] = (rows, (out / f"array-{group}.sbatch").read_text())
    return groups


def _all_rows(*, groups: dict[str, tuple[list[list[str]], str]]) -> list[list[str]]:
    return [row for rows, _ in groups.values() for row in rows]


def _directives(*, sbatch: str) -> list[str]:
    return [line for line in sbatch.splitlines() if line.startswith("#SBATCH")]


def test_della_writes_36_tasks_in_three_time_groups(*, tmp_path: Path) -> None:
    groups = _della(tmp_path=tmp_path)
    assert sorted(groups) == sorted(GROUPS)
    rows = _all_rows(groups=groups)
    assert len(rows) == 36
    arms = ShelfLaunch.arms(expectimax_depths=[6])
    assert sorted((row[2], row[3]) for row in rows) == sorted(
        (name, str(seed)) for name in arms for seed in range(6)
    )
    assert {row[1] for row in rows} == {SHA}
    assert len({row[0] for row in rows}) == 36
    assert {group: sorted({row[2] for row in groups[group][0]}) for group in groups} == {
        "ees": ["ees", "ees-always-ask"],
        "determinized": ["cccc-determinized", "cccc-determinized-always-ask"],
        "expectimax": ["cccc-expectimax-d6", "cccc-expectimax-d6-always-ask"],
    }


def test_della_d7_is_absent_by_default_and_present_on_request(*, tmp_path: Path) -> None:
    default = _all_rows(groups=_della(tmp_path=tmp_path / "default"))
    assert not any("d7" in row[2] for row in default)
    both = _della(tmp_path=tmp_path / "both", extra=["--expectimax-depths", "6", "7"])
    assert len(_all_rows(groups=both)) == 48
    rows, sbatch = both["expectimax"]
    assert len(rows) == 24
    assert {row[2] for row in rows} >= {"cccc-expectimax-d7", "cccc-expectimax-d7-always-ask"}
    assert "#SBATCH --array=1-24" in _directives(sbatch=sbatch)


def test_della_each_group_gets_its_own_time_limit(*, tmp_path: Path) -> None:
    groups = _della(tmp_path=tmp_path)
    shared = [
        "#SBATCH --account=tsilver",
        "#SBATCH --partition=cpu",
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=12G",
        "#SBATCH --array=1-12",
    ]
    times = {"ees": "24:00:00", "determinized": "72:00:00", "expectimax": "72:00:00"}
    for group, (_, sbatch) in groups.items():
        directives = _directives(sbatch=sbatch)
        for directive in shared:
            assert directive in directives
        assert f"#SBATCH --time={times[group]}" in directives
        assert f"#SBATCH --job-name=shelf-20261020-{group}" in directives
        # Della's cpu partition picks the QOS from --time unless one is set.
        assert not any(line.startswith("#SBATCH --qos") for line in directives)
        assert sbatch.startswith("#!/bin/bash\n")


def test_della_time_and_qos_are_overridable_per_group(*, tmp_path: Path) -> None:
    groups = _della(
        tmp_path=tmp_path,
        extra=["--della-time", "determinized=36:00:00", "--della-qos", "expectimax=medium"],
    )
    determinized = _directives(sbatch=groups["determinized"][1])
    assert "#SBATCH --time=36:00:00" in determinized
    expectimax = _directives(sbatch=groups["expectimax"][1])
    assert "#SBATCH --qos=medium" in expectimax
    assert "#SBATCH --time=72:00:00" in expectimax
    assert "#SBATCH --time=24:00:00" in _directives(sbatch=groups["ees"][1])


@pytest.mark.parametrize("flag", [["--della-time", "nosuch=1:00:00"], ["--della-qos", "ees"]])
def test_della_rejects_an_unknown_group_or_a_malformed_override(
    *, flag: list[str], tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        main(argv=["della", "--tag", "t", "--sha", SHA, "--out", str(tmp_path), *flag])


def test_della_tasks_run_the_launchers_own_argv_with_della_paths(*, tmp_path: Path) -> None:
    rows = _all_rows(groups=_della(tmp_path=tmp_path))
    arms = ShelfLaunch.arms(expectimax_depths=[6])
    code = f"{DELLA}/code/{SHA}"
    for name, _sha, arm_name, seed, output_dir, joined in rows:
        arm = arms[arm_name]
        assert output_dir == f"{DELLA}/results/shelf-20261020/{arm_name}/{arm.method}/{seed}"
        argv = shlex.split(joined)
        local = ShelfLaunch.cli_argv(
            arm=arm, size=FULL, seed=int(seed), output_dir=Path(output_dir)
        )
        costs = ALWAYS_ASK_COSTS if arm.always_ask else COSTS
        assert argv == [
            f"{code}/{costs.relative_to(REPO)}" if token == str(costs) else token for token in local
        ]
        assert not any(str(REPO) in token for token in argv)
        assert Cli.parse_args(argv=argv).seed == int(seed)
        assert name == f"shelf-20261020-{arm_name}-s{seed}"


def test_della_never_skips_the_gate_unless_asked(*, tmp_path: Path) -> None:
    groups = _della(tmp_path=tmp_path)
    assert all("--skip-skill-benchmark-gate" not in row[5] for row in _all_rows(groups=groups))
    assert all("skip-skill-benchmark-gate" not in sbatch for _, sbatch in groups.values())
    skipped = _della(tmp_path=tmp_path / "skip", extra=["--skip-skill-benchmark-gate"])
    assert all(
        shlex.split(row[5])[-1] == "--skip-skill-benchmark-gate"
        for row in _all_rows(groups=skipped)
    )


def test_della_sbatch_preflights_then_runs_under_the_watchdog(*, tmp_path: Path) -> None:
    for _, sbatch in _della(tmp_path=tmp_path).values():
        zstd = sbatch.index("import zstandard")
        isolation = sbatch.index("ShelfLaunch.require_isolated()")
        watchdog = sbatch.index("scripts/scratch_watchdog.py")
        assert zstd < watchdog and isolation < watchdog
        assert "pip install zstandard" in sbatch
        # One logical bash line: backslash-newline continuations joined.
        logical = sbatch.replace("\\\n", " ").splitlines()
        run = next(line for line in logical if "scripts/scratch_watchdog.py" in line)
        assert "--path /scratch/gpfs/TSILVER" in run
        assert '--output-dir "$OUT"' in run
        assert "-m hitl_pmp.cli" in run
        assert run.index("--path") < run.index(" -- ") < run.index("hitl_pmp.cli")
        assert f'CODE="{DELLA}/code/$SHA"' in sbatch


def test_della_bash_text_is_syntactically_valid(*, tmp_path: Path) -> None:
    import subprocess

    _della(tmp_path=tmp_path)
    for sbatch in sorted((tmp_path / "della").glob("array-*.sbatch")):
        subprocess.run(["bash", "-n", str(sbatch)], check=True)


def test_della_refuses_without_a_tag(*, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="della needs"):
        main(argv=["della", "--sha", SHA, "--out", str(tmp_path)])


def test_della_sbatch_parses_a_manifest_line_back_to_the_same_argv(*, tmp_path: Path) -> None:
    import subprocess

    rows, sbatch = _della(tmp_path=tmp_path)["expectimax"]
    lines = sbatch.splitlines()
    read = next(line for line in lines if line.startswith("IFS="))
    split = next(line for line in lines if line.startswith('eval "ARGS='))
    manifest = tmp_path / "della" / "manifest-expectimax.tsv"
    script = (
        f'LINE=$(sed -n "7p" {manifest})\n{read}\n{split}\n'
        'printf "%s\\n" "$NAME" "$ARM" "$SEED" "$OUT"; printf "%s\\n" "${ARGS[@]}"\n'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    name, _sha, arm, seed, output_dir, joined = rows[6]
    assert out == [name, arm, seed, output_dir, *shlex.split(joined)]


def test_every_della_run_records_evaluation_motion_and_seed_zero_records_practice(
    *, tmp_path: Path
) -> None:
    rows = _all_rows(groups=_della(tmp_path=tmp_path))
    scopes = {
        (arm, int(seed)): Cli.parse_args(argv=shlex.split(joined)).record_motion
        for _name, _sha, arm, seed, _out, joined in rows
    }
    assert len(scopes) == 36
    for (_, seed), scope in scopes.items():
        assert scope == ("all" if seed == 0 else "evaluation")
    assert {arm for arm, seed in scopes if seed == 0} == set(
        ShelfLaunch.arms(expectimax_depths=[6])
    )


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_a_single_runs_argv_records_motion_by_seed(*, seed: int) -> None:
    arm = ShelfLaunch.arms(expectimax_depths=[6])["ees"]
    argv = ShelfLaunch.cli_argv(arm=arm, size=FULL, seed=seed, output_dir=Path("/x"))
    assert Cli.parse_args(argv=argv).record_motion == ("all" if seed == 0 else "evaluation")


def test_the_agni_sweep_records_evaluation_motion_for_every_seed(*, tmp_path: Path) -> None:
    """`run_sweep` shares one argv across seeds, so practice motion is della/print only."""
    arm = ShelfLaunch.arms(expectimax_depths=[6])["ees"]
    command = ShelfLaunch.sweep_command(
        arm=arm, size=FULL, results_root=tmp_path, max_workers=3, skip_gate=False
    )
    assert _value(argv=_shared(command=command), flag="--record-motion") == "evaluation"


def test_no_arm_asks_for_episode_traces_the_step_protocol_refuses() -> None:
    for arm in ShelfLaunch.arms(expectimax_depths=[6, 7]).values():
        argv = ShelfLaunch.cli_argv(arm=arm, size=FULL, seed=0, output_dir=Path("/x"))
        assert "--record-episode-traces" not in argv
