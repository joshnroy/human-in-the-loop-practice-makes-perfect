import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from scripts.tossing3d_launch import DELLA, FULL, SEEDS, Tossing3DLaunch

ARMS = [
    "ees",
    "expectimax",
    "determinized",
    "determinized-curtis",
    "ees-always-ask",
    "expectimax-always-ask",
    "determinized-always-ask",
    "determinized-curtis-always-ask",
]
CURTIS_ARMS = ["determinized-curtis", "determinized-curtis-always-ask"]


def _parse(*, name: str, seed: int = 0) -> object:
    arm = Tossing3DLaunch.arms()[name]
    return Cli.parse_args(
        argv=Tossing3DLaunch.cli_argv(arm=arm, size=FULL, seed=seed, output_dir=Path("/out"))
    )


def test_the_plan_is_eight_arms_over_seeds_zero_to_five() -> None:
    # Josh, 2026-10-22: six seeds per classical arm, so a paired sign test can reach 0.031.
    assert list(Tossing3DLaunch.arms()) == ARMS
    assert SEEDS == 6


@pytest.mark.parametrize("name", ARMS)
def test_every_arm_runs_the_canonical_budget_under_the_step_protocol(*, name: str) -> None:
    args = _parse(name=name)
    assert args.env == "tossing3d"
    assert args.practice_step_budget == 160000
    assert args.measurement_interval_steps == 3200
    assert args.num_test_tasks == 10
    assert args.evaluation_control_steps == 500
    assert args.layout == "barrier" and args.evaluation_layout == "barrier"
    assert args.max_steps_per_interaction == 20


@pytest.mark.parametrize("name", [n for n in ARMS if not n.startswith("ees")])
def test_the_planner_arms_are_the_canonical_preset(*, name: str) -> None:
    args = _parse(name=name)
    assert args.method == "pomdp"
    assert args.preset == "tossing3d-canonical"
    solver = "determinized_astar" if name.startswith("determinized") else "expectimax"
    assert args.pomdp_solver == solver
    # Josh, 2026-10-23: determinized also as Curtis et al.'s PO_DAStar (#577), every
    # outcome enumerated at lambda = 0.01, the value that passed the 3,200-step gate.
    curtis = name in CURTIS_ARMS
    assert args.pomdp_determinized_variant == ("curtis" if curtis else "algorithm3")
    assert args.pomdp_num_rollouts is None
    assert args.pomdp_observation_probability_weight == (0.01 if curtis else 0.0)
    assert args.pomdp_draw_scope == "node"


@pytest.mark.parametrize("name", ARMS)
def test_each_arm_prices_human_resets_with_its_own_cost_file(*, name: str) -> None:
    args = _parse(name=name)
    method = "ees" if name.startswith("ees") else "pomdp"
    always_ask = name.endswith("always-ask")
    assert args.always_ask_reset is always_ask
    expected = Tossing3DArmCosts.cost_config(method=method, always_ask=always_ask)
    assert Path(args.practice_cost_config).resolve() == expected
    PracticeCosts.load(path=args.practice_cost_config)


def test_ees_runs_without_skip_perfect_and_with_its_default_prior() -> None:
    # The shelf's labelled deviation (Josh, 2026-10-21), applied here too; EXP-22c's
    # prior, since only the shelf chose Beta(1, 3).
    args = _parse(name="ees")
    assert args.method == "ees"
    assert args.reproduce_predicators_skip_perfect is False
    assert "--ees-prior-alpha" not in Tossing3DLaunch.cli_argv(
        arm=Tossing3DLaunch.arms()["ees"], size=FULL, seed=0, output_dir=Path("/out")
    )


@pytest.mark.parametrize("name", ARMS)
def test_every_arm_stops_after_three_perfect_evaluations_in_a_row(*, name: str) -> None:
    # Josh, 2026-10-22: "if it gets 3 10/10 evals in a row, we can terminate early".
    # EvaluationStopping counts a streak (reset by any imperfect evaluation).
    assert _parse(name=name).stop_after_perfect_evaluations == 3


def test_the_della_manifest_has_one_task_per_arm_and_seed() -> None:
    arms = list(Tossing3DLaunch.arms().values())
    lines = Tossing3DLaunch.della_manifest(arms=arms, size=FULL, sha="abc123", tag="t3")
    assert len(lines) == 48
    seen = set()
    for line in lines:
        name, sha, arm, seed, out, argv = line.split("\t")
        assert sha == "abc123"
        assert out == f"{DELLA}/results/t3/{arm}/{Tossing3DLaunch.arms()[arm].method}/{seed}"
        parsed = shlex.split(argv)
        assert parsed[parsed.index("--seed") + 1] == seed
        assert parsed[parsed.index("--output-dir") + 1] == out
        seen.add((arm, int(seed)))
    assert seen == {(arm, seed) for arm in ARMS for seed in range(6)}


def test_the_curtis_arms_share_the_determinized_della_group() -> None:
    groups = {
        name: Tossing3DLaunch.della_group(arm=arm) for name, arm in Tossing3DLaunch.arms().items()
    }
    assert [name for name, group in groups.items() if group == "determinized"] == [
        "determinized",
        "determinized-curtis",
        "determinized-always-ask",
        "determinized-curtis-always-ask",
    ]


def test_the_curtis_arms_alone_are_one_twelve_task_manifest() -> None:
    arms = [Tossing3DLaunch.arms()[name] for name in CURTIS_ARMS]
    lines = Tossing3DLaunch.della_manifest(arms=arms, size=FULL, sha="abc123", tag="t3c")
    assert [(line.split("\t")[2], int(line.split("\t")[3])) for line in lines] == [
        (arm, seed) for arm in CURTIS_ARMS for seed in range(6)
    ]


def test_the_della_time_limits_are_twice_the_dry_run_projection() -> None:
    # Josh, 2026-10-22: 2x the dry run's projection per group.
    from scripts.tossing3d_launch import DELLA_GROUPS

    assert {group: limits.time for group, limits in DELLA_GROUPS.items()} == {
        "ees": "08:00:00",
        "determinized": "16:00:00",
        "expectimax": "48:00:00",
    }


def test_all_arms_make_the_new_physics_explicit():
    for name in ARMS:
        args = _parse(name=name)
        assert args.bin_mass_kg == 1.0
        assert args.cube_mass_kg == 0.1
        assert args.enable_bin_pouring
        costs = PracticeCosts.load(path=args.practice_cost_config)
        assert costs.robot_step.value == 0.01
        assert costs.objective_lambda == 0.0003


def test_pilots_cover_the_six_default_arms_without_early_stopping():
    from scripts.tossing3d_launch import DEFAULT_ARMS, PILOT

    arms = [Tossing3DLaunch.arms()[name] for name in DEFAULT_ARMS]
    lines = Tossing3DLaunch.della_manifest(
        arms=arms, size=PILOT, sha="abc123", tag="pilot", seeds=(0, 2)
    )
    assert len(lines) == 12
    for line in lines:
        args = Cli.parse_args(argv=shlex.split(line.split("\t")[-1]))
        assert args.practice_step_budget == 6400
        assert args.stop_after_perfect_evaluations == 0
        assert args.num_test_tasks == 10


def test_della_workers_use_gpfs_tmp_and_parse_arguments_without_eval():
    from scripts.tossing3d_launch import DELLA_GROUPS, della_sbatch

    script = della_sbatch(
        tag="pour", group="ees", limits=DELLA_GROUPS["ees"], tasks=12, sha="a" * 40
    )
    assert "HITL_PMP_TMPDIR=" + DELLA + "/tmp/" in script
    assert "export TMP=" in script
    assert 'TMPDIR="$HITL_PMP_TMPDIR"' in script
    assert 'TEMP="$HITL_PMP_TMPDIR"' in script
    assert "eval " not in script
    assert "--mem=16G" in script


@pytest.mark.parametrize(("size", "count"), [("pilot", 12), ("full", 36)])
def test_default_generated_launch_uses_the_six_requested_arms(*, tmp_path, size, count):
    from scripts.tossing3d_launch import DEFAULT_ARMS, main

    assert (
        main(
            argv=[
                "della",
                "--size",
                size,
                "--tag",
                "checked",
                "--sha",
                "a" * 40,
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    rows = [
        line.split("\t")
        for p in tmp_path.glob("manifest-*.tsv")
        for line in p.read_text().splitlines()
    ]
    assert len(rows) == count
    assert {row[2] for row in rows} == set(DEFAULT_ARMS)
    assert {int(row[3]) for row in rows} == ({0, 2} if size == "pilot" else set(range(6)))
    for p in tmp_path.glob("array-*.sbatch"):
        assert "--time=02:00:00" in p.read_text() if size == "pilot" else True


def test_long_pilot_can_probe_only_the_two_planners(*, tmp_path):
    from scripts.tossing3d_launch import main

    assert (
        main(
            argv=[
                "della",
                "--size",
                "long-pilot",
                "--arms",
                "expectimax",
                "determinized-curtis",
                "--seeds",
                "0",
                "--tag",
                "longchecked",
                "--sha",
                "a" * 40,
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    rows = [
        line.split("\t")
        for p in tmp_path.glob("manifest-*.tsv")
        for line in p.read_text().splitlines()
    ]
    assert len(rows) == 2
    for row in rows:
        args = Cli.parse_args(argv=shlex.split(row[-1]))
        assert args.practice_step_budget == 25600
        assert args.stop_after_perfect_evaluations == 0


def test_worker_rejects_an_incorrect_manifest_revision_before_simulation(*, tmp_path):
    from scripts.tossing3d_launch import run_worker

    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("name\t" + "a" * 40 + "\tees\t0\t/unused\t--env tossing3d\n")
    with pytest.raises(RuntimeError, match="manifest revision"):
        run_worker(manifest=manifest, index=1, expected_sha="b" * 40)


@pytest.mark.parametrize("dirty", [False, True])
def test_preflight_rejects_wrong_or_modified_source_before_building_physics(
    *, monkeypatch, tmp_path, dirty
):
    import subprocess

    from hitl_pmp.config_snapshot import ConfigSnapshot
    from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
    from scripts.tossing3d_launch import PILOT, preflight

    monkeypatch.setattr(
        ConfigSnapshot,
        "collect",
        lambda **kwargs: SimpleNamespace(git_commit="a" * 40 if dirty else "b" * 40),
    )

    def no_scene(**kwargs):
        pytest.fail("rejected source must not build a simulator")

    monkeypatch.setattr(Tossing3DCli, "build_practice_problem", no_scene)
    if dirty:

        def modified(*args, **kwargs):
            raise subprocess.CalledProcessError(1, args[0])

        monkeypatch.setattr(subprocess, "run", modified)
    argv = Tossing3DLaunch.cli_argv(
        arm=Tossing3DLaunch.arms()["ees"], size=PILOT, seed=0, output_dir=tmp_path
    )
    with pytest.raises(subprocess.CalledProcessError if dirty else RuntimeError):
        preflight(argv=argv, output=tmp_path, frozen_sha="a" * 40)


def test_worker_preserves_quoted_manifest_arguments_without_a_shell(*, monkeypatch, tmp_path):
    from scripts import tossing3d_launch

    sha = "a" * 40
    output = tmp_path / "results with spaces"
    argv = ["--env", "tossing3d", "--method", "ees", "--output-dir", str(output)]
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\t".join(["pilot", sha, "ees", "0", str(output), shlex.join(argv)]) + "\n")
    checked = []
    commands = []
    monkeypatch.setattr(tossing3d_launch, "preflight", lambda **kwargs: checked.append(kwargs))

    def run(command, **kwargs):  # noqa: PLR0917 (subprocess signature)
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tossing3d_launch.subprocess, "run", run)
    assert tossing3d_launch.run_worker(manifest=manifest, index=1, expected_sha=sha) == 0
    assert checked[0]["argv"] == argv
    assert commands[0][0][-len(argv) :] == argv
    assert commands[0][1].get("shell", False) is False
