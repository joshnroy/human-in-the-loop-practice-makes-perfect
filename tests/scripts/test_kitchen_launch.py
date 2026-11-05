import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from scripts.kitchen_launch import KitchenLaunch


@pytest.mark.parametrize("stage,budget", [("pilot", 6400), ("overnight", 25600)])
def test_paired_matrix_and_evaluation_contract(*, stage, budget):
    runs = KitchenLaunch.runs(stage=stage, results_root=Path("/unused"))
    assert len(runs) == 6
    assert len({r.output_dir for r in runs}) == 6
    assert [(r.output_dir.parts[-3], r.seed) for r in runs] == [
        (arm, seed) for arm in ("ees", "curtis", "expectimax") for seed in (0, 2)
    ]
    for run in runs:
        args = Cli.parse_args(argv=run.command[3:])
        assert args.env == "pouring3d_kitchen"
        assert args.practice_step_budget == budget
        assert args.measurement_interval_steps == 6400
        assert args.initial_measurement
        assert args.num_test_tasks == (3 if stage == "pilot" else 9)
        assert args.evaluation_control_steps == 7500
        assert args.requested_counts == [1, 2, 3]
        assert args.payload_count == 3
        assert args.practice_reset_policy == "never"
        assert not args.always_ask_reset
        assert args.stop_after_perfect_evaluations == 0
        assert args.record_motion == ("all" if run.seed == 0 else "evaluation")
        assert args.scene_bg
        assert args.exploration_epsilon == 0.5
        assert args.sampler_max_train_iters == 10000
        assert args.reproduce_predicators_explore_target_only
        assert not args.reproduce_predicators_double_observe
        assert args.surface_mu == 0.1
        assert args.pour_ramp_rate_bounds == [0.25, 5.0]
        assert args.pour_tilt_bounds is None
        assert args.pour_amplitude_bounds is None
        assert args.pour_duration_bounds is None
        costs = PracticeCosts.load(path=args.practice_cost_config)
        assert costs.robot_step.value == 0.01
        assert costs.objective_lambda == 0.0003
        assert set(costs.human_skills) == {"human_bulk_pour_back"}
        assert costs.human_skills["human_bulk_pour_back"].cost.value == 5
        assert costs.human_skills["human_bulk_pour_back"].duration.value == 1
        if args.method == "ees":
            assert args.ees_score_human_skills
            assert args.practice_groundings == "observed"
            assert args.ees_preview_test_goals
            assert args.record_ees_decisions
            assert not args.reproduce_predicators_skip_perfect
            assert args.reproduce_predicators_random_when_stranded
        else:
            # Curtis is bounded by its iteration budget, never a search depth.
            assert args.pomdp_search_depth == (
                None if args.pomdp_solver == "determinized_astar" else 6
            )
            assert args.pomdp_num_particles == 1024
            assert args.pomdp_competence_model == "local_trend"
            if run.output_dir.parts[-3] == "curtis":
                assert args.pomdp_solver == "determinized_astar"
                assert args.pomdp_determinized_variant == "curtis"
                assert args.pomdp_max_search_iterations == 1600
                assert args.pomdp_draw_scope == "node"
                assert args.pomdp_num_rollouts is None
                assert args.pomdp_observation_probability_weight == 0.01
            else:
                assert args.pomdp_solver == "expectimax"
                assert args.pomdp_observation_probability_weight == 0


def test_manifest_is_exclusive_and_contains_resolved_commands(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "planned"
    assert len(manifest["runs"]) == 6
    assert all(row["resolved_args"]["initial_measurement"] for row in manifest["runs"])
    assert len(manifest["cost_sha256"]) == 64
    with pytest.raises(FileExistsError):
        KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")


def test_run_refuses_changed_code_before_spawning(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "b" * 40})
    with pytest.raises(RuntimeError, match="identity"):
        KitchenLaunch.run(manifest=path, max_workers=1)


def test_run_refuses_modified_manifest_commands(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    data = json.loads(path.read_text())
    data["runs"][0]["argv"].append("--always-ask-reset")
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="commands"):
        KitchenLaunch.run(manifest=path, max_workers=1)


def test_run_refuses_existing_results(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    runs = KitchenLaunch.runs(stage="pilot", results_root=tmp_path / "results")
    runs[-1].output_dir.mkdir(parents=True)
    (runs[-1].output_dir / "stats.json").write_text("{}")
    with pytest.raises(FileExistsError):
        KitchenLaunch.run(manifest=path, max_workers=1)


def test_array_worker_only_executes_its_row(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    monkeypatch.setattr(KitchenLaunch, "native_check", lambda: {"ready": True})
    path = tmp_path / "manifest.json"
    root = tmp_path / "results"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=root)
    runs = KitchenLaunch.runs(stage="pilot", results_root=root)
    # A completed sibling task must not block this array worker.
    runs[0].output_dir.mkdir(parents=True)
    observed = []

    def execute(**kwargs):
        observed.extend(kwargs["runs"])
        return []

    monkeypatch.setattr("scripts.kitchen_launch.SweepRunner.execute", execute)
    assert KitchenLaunch.run(manifest=path, max_workers=1, task=3) == 0
    assert observed == [runs[2]]
    receipt = json.loads((runs[2].output_dir / "launch.json").read_text())
    assert receipt["run"]["arm"] == "curtis"
    assert receipt["run"]["seed"] == 0
    assert not runs[3].output_dir.exists()


@pytest.mark.parametrize("task", [0, 7])
def test_array_index_is_checked(*, task, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    with pytest.raises(ValueError, match="Task index"):
        KitchenLaunch.run(manifest=path, max_workers=1, task=task)


def test_latest_method_factories_accept_every_pouring_arm(*, tmp_path, monkeypatch):
    from hitl_pmp.cli import METHODS
    from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli

    built = []

    def capture(*, args, method_factory, **kwargs):
        problem = KitchenCli.build_problem(args=args)
        method = method_factory(KitchenCli.context(args=args, problem=problem))
        built.append(method)
        assert method.goal_pursuit_horizon == 0
        assert method.reproduce_predicators_explore_target_only
        assert method.practice_groundings == "observed"
        assert {skill.name for skill in method.skills() if skill.name.startswith("Pour")} == {
            "Pour1",
            "Pour2",
            "Pour3",
        }
        if args.method == "ees":
            assert method.ees_decision_log == args.output_dir / "ees_decisions.jsonl"
            assert [goal[0][1][0] for goal in method.preview_test_goals] == [
                "request_1",
                "request_2",
                "request_3",
            ]
        else:
            assert method.pomdp_solver == args.pomdp_solver
            assert method.pomdp_search_depth == (
                None if args.pomdp_solver == "determinized_astar" else 6
            )
            assert method.pomdp_domain.sampler_skills == {"Pour1", "Pour2", "Pour3"}
            assert method.pomdp_domain.human_skill_names == {"human_bulk_pour_back"}
            assert "PickShrimpFromSurface" in dict(method.pomdp_domain.belief_sources)
            assert method.ees_decision_log is None
            assert method.preview_test_goals == ()

    monkeypatch.setattr(KitchenCli, "run_method", staticmethod(capture))
    for run in KitchenLaunch.runs(stage="pilot", results_root=tmp_path):
        args = Cli.parse_args(argv=run.command[3:])
        METHODS[args.method].run(args=args, env_cli=KitchenCli)
    assert len(built) == 6


DELLA_SHA = "a" * 40


def _della(*, tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": DELLA_SHA})
    out = tmp_path / "della"
    KitchenLaunch.della(out=out, tag="kp-test", **kwargs)
    return json.loads((out / "manifest.json").read_text()), (out / "array.sbatch").read_text()


def test_della_manifest_is_frozen_for_the_della_checkout(*, tmp_path, monkeypatch):
    from scripts.kitchen_launch import DELLA

    manifest, _ = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="overnight", seeds=(0, 1, 2)
    )
    code = f"{DELLA}/code/{DELLA_SHA}"
    assert manifest["working_directory"] == code
    assert manifest["results_root"] == f"{DELLA}/results/kp-test"
    assert manifest["seeds"] == [0, 1, 2]
    assert manifest["identity"] == {"commit": DELLA_SHA}
    assert [(row["arm"], row["seed"]) for row in manifest["runs"]] == [
        (arm, seed) for arm in ("ees", "curtis", "expectimax") for seed in (0, 1, 2)
    ]
    for row in manifest["runs"]:
        method = "ees" if row["arm"] == "ees" else "pomdp"
        assert row["output_dir"] == f"{DELLA}/results/kp-test/{row['arm']}/{method}/{row['seed']}"
        cost = row["argv"][row["argv"].index("--practice-cost-config") + 1]
        assert cost == f"{code}/configs/kitchen_pilot_costs.json"
        assert row["resolved_args"]["practice_step_budget"] == 25600


def test_della_rows_carry_no_path_from_the_submitting_checkout(*, tmp_path, monkeypatch):
    """`run` recomputes the rows inside the Della checkout and demands equality."""
    from scripts.kitchen_launch import REPO

    manifest, _ = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="overnight", seeds=(0, 1, 2)
    )
    assert str(REPO) not in json.dumps(manifest["runs"])


def test_della_sbatch_runs_one_manifest_row_per_array_task(*, tmp_path, monkeypatch):
    from scripts.kitchen_launch import DELLA

    _, script = _della(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="overnight",
        seeds=(0, 1, 2),
        time="96:00:00",
        mem="12G",
    )
    code = f"{DELLA}/code/{DELLA_SHA}"
    assert "#SBATCH --partition=cpu" in script
    assert "#SBATCH --array=1-9" in script
    assert "#SBATCH --time=96:00:00" in script
    assert "#SBATCH --mem=12G" in script
    assert "#SBATCH --job-name=kp-test" in script
    assert f"cd {code}" in script
    assert f"export HITL_PMP_CONDA_ENV={DELLA}/envs/hitl-pmp" in script
    assert f"export FD_EXEC_PATH={DELLA}/downward" in script
    assert "export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1" in script
    assert 'export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"' in script
    assert 'export XDG_RUNTIME_DIR="$HITL_PMP_TMPDIR/xdg"' in script
    assert "OPENBLAS_NUM_THREADS=1" in script
    assert (
        script
        .replace(" \\\n  ", " ")
        .rstrip()
        .endswith(
            f"{code}/scripts/with_env.sh python -m scripts.kitchen_launch run "
            '--manifest "$1" --task "$SLURM_ARRAY_TASK_ID" --go'
        )
    )


def test_della_sbatch_pins_the_genoa_nodes_right_after_the_partition(*, tmp_path, monkeypatch):
    _, script = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="overnight", seeds=(0, 1, 2)
    )
    lines = script.splitlines()
    partition = lines.index("#SBATCH --partition=cpu")
    assert lines[partition + 1] == "#SBATCH --constraint=genoa"
    assert script.count("--constraint") == 1


@pytest.mark.parametrize(("constraint", "line"), [("cascade", "#SBATCH --constraint=cascade")])
def test_della_sbatch_takes_another_constraint(*, tmp_path, monkeypatch, constraint, line):
    _, script = _della(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="overnight",
        seeds=(0,),
        constraint=constraint,
    )
    assert line in script.splitlines()
    assert script.count("--constraint") == 1


def test_della_sbatch_can_drop_the_constraint(*, tmp_path, monkeypatch):
    _, script = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="overnight", seeds=(0,), constraint=""
    )
    assert "--constraint" not in script


def test_the_della_cli_passes_the_constraint_through(*, tmp_path, monkeypatch):
    import sys

    seen = {}

    def della(**kwargs):
        seen.update(kwargs)
        return tmp_path / "array.sbatch"

    monkeypatch.setattr(KitchenLaunch, "della", della)
    argv = ["kitchen_launch", "della", "--out", str(tmp_path), "--tag", "t"]
    monkeypatch.setattr(sys, "argv", argv)
    assert KitchenLaunch.main() == 0
    assert seen["constraint"] == "genoa"
    monkeypatch.setattr(sys, "argv", [*argv, "--constraint", ""])
    assert KitchenLaunch.main() == 0
    assert seen["constraint"] == ""


def test_della_refuses_a_tag_that_is_not_a_plain_name(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": DELLA_SHA})
    with pytest.raises(ValueError, match="tag"):
        KitchenLaunch.della(out=tmp_path, tag="../x", stage="overnight", seeds=(0,))


def test_dry_stage_keeps_the_protocol_but_shrinks_practice():
    for run in KitchenLaunch.runs(stage="dry", results_root=Path("/unused"), seeds=(0,)):
        args = Cli.parse_args(argv=run.command[3:])
        assert args.practice_step_budget == 1600
        assert args.measurement_interval_steps == 1600
        assert args.num_test_tasks == 3
        assert args.initial_measurement
        assert args.evaluation_control_steps == 7500


def test_a_started_sibling_task_does_not_block_a_later_array_task(*, tmp_path, monkeypatch):
    """Della starts array tasks at different times; a sibling's live run is not ours."""
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    monkeypatch.setattr(KitchenLaunch, "native_check", lambda: {"ready": True})
    path = tmp_path / "manifest.json"
    root = tmp_path / "results"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=root)
    runs = KitchenLaunch.runs(stage="pilot", results_root=root)
    (runs[2].output_dir / "snapshots").mkdir(parents=True)
    observed = []
    monkeypatch.setattr(
        "scripts.kitchen_launch.SweepRunner.execute",
        lambda **kwargs: observed.extend(kwargs["runs"]) or [],
    )
    assert KitchenLaunch.run(manifest=path, max_workers=1, task=1) == 0
    assert observed == [runs[0]]
    with pytest.raises(FileExistsError):
        KitchenLaunch.run(manifest=path, max_workers=1, task=3)


def test_run_honours_the_manifest_seed_list(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    monkeypatch.setattr(KitchenLaunch, "native_check", lambda: {"ready": True})
    path = tmp_path / "manifest.json"
    root = tmp_path / "results"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=root, seeds=(0, 1, 2))
    observed = []
    monkeypatch.setattr(
        "scripts.kitchen_launch.SweepRunner.execute",
        lambda **kwargs: observed.extend(kwargs["runs"]) or [],
    )
    assert KitchenLaunch.run(manifest=path, max_workers=1, task=9) == 0
    assert [(r.output_dir.parts[-3], r.seed) for r in observed] == [("expectimax", 2)]
    with pytest.raises(ValueError, match="between 1 and 9"):
        KitchenLaunch.run(manifest=path, max_workers=1, task=10)


# --- parallel evaluation ----------------------------------------------------------------


def test_local_runs_evaluate_one_task_at_a_time_unless_asked():
    for stage in ("pilot", "overnight", "dry"):
        for run in KitchenLaunch.runs(stage=stage, results_root=Path("/unused")):
            args = Cli.parse_args(argv=run.command[3:])
            assert args.evaluation_workers == 1
            assert args.evaluation_task_rng == "shared"
            assert args.evaluation_task_timeout_seconds is None


def test_parallel_evaluation_flags_take_per_task_rngs_and_a_timeout():
    runs = KitchenLaunch.runs(stage="overnight", results_root=Path("/unused"), evaluation_workers=9)
    for run in runs:
        args = Cli.parse_args(argv=run.command[3:])
        assert args.evaluation_workers == 9
        assert args.evaluation_task_rng == "per-task"
        assert args.evaluation_task_timeout_seconds == 4 * 3600


def test_della_overnight_evaluates_all_nine_tasks_at_once(*, tmp_path, monkeypatch):
    manifest, script = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="overnight", seeds=(0, 1, 2)
    )
    assert manifest["evaluation_workers"] == 9
    for row in manifest["runs"]:
        assert row["resolved_args"]["evaluation_workers"] == 9
        assert row["resolved_args"]["evaluation_task_rng"] == "per-task"
    assert "#SBATCH --cpus-per-task=10" in script
    assert f"#SBATCH --mem={KitchenLaunch.della_memory(evaluation_workers=9)}" in script
    assert "#SBATCH --time=12:00:00" in script


@pytest.mark.parametrize("workers", [1, 3, 9])
def test_della_memory_covers_practice_plus_every_evaluation_process(*, workers):
    from scripts.kitchen_launch import DELLA_EVALUATION_PROCESS_GB, DELLA_PRACTICE_GB

    gigabytes = int(KitchenLaunch.della_memory(evaluation_workers=workers).removesuffix("G"))
    assert gigabytes >= 1.25 * (DELLA_PRACTICE_GB + workers * DELLA_EVALUATION_PROCESS_GB)


def test_della_sizing_follows_an_explicit_worker_count(*, tmp_path, monkeypatch):
    manifest, script = _della(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        stage="overnight",
        seeds=(0,),
        evaluation_workers=3,
    )
    assert manifest["evaluation_workers"] == 3
    assert "#SBATCH --cpus-per-task=4" in script
    assert f"#SBATCH --mem={KitchenLaunch.della_memory(evaluation_workers=3)}" in script


def test_run_rebuilds_the_rows_with_the_manifests_worker_count(*, tmp_path, monkeypatch):
    monkeypatch.setattr(KitchenLaunch, "identity", lambda **_: {"commit": "a" * 40})
    monkeypatch.setattr(KitchenLaunch, "native_check", lambda: {"ready": True})
    path = tmp_path / "manifest.json"
    root = tmp_path / "results"
    KitchenLaunch.write_manifest(path=path, stage="pilot", results_root=root, evaluation_workers=3)
    observed = []
    monkeypatch.setattr(
        "scripts.kitchen_launch.SweepRunner.execute",
        lambda **kwargs: observed.extend(kwargs["runs"]) or [],
    )
    assert KitchenLaunch.run(manifest=path, max_workers=1, task=1) == 0
    assert "--evaluation-workers" in observed[0].command


def test_long_stage_practises_200k_ticks_with_nine_measurements():
    for run in KitchenLaunch.runs(stage="long", results_root=Path("/unused")):
        args = Cli.parse_args(argv=run.command[3:])
        assert args.practice_step_budget == 200_000
        assert args.measurement_interval_steps == 25_000
        assert args.num_test_tasks == 9
        assert args.initial_measurement
        assert args.evaluation_control_steps == 7500
        # The initial measurement, one every 25k ticks short of the budget, the final one.
        interior = len(range(25_000, 200_000, 25_000))
        assert 1 + interior + 1 == 9


def test_della_long_is_sized_for_the_slowest_arm_within_48_hours(*, tmp_path, monkeypatch):
    manifest, script = _della(
        tmp_path=tmp_path, monkeypatch=monkeypatch, stage="long", seeds=(0, 1, 2)
    )
    assert manifest["evaluation_workers"] == 9
    assert "#SBATCH --constraint=genoa" in script
    assert "#SBATCH --cpus-per-task=10" in script
    assert "#SBATCH --time=48:00:00" in script
    # Practice for 8x the overnight budget gets 12 GB, not 3: tonight's 27G job peaked
    # at 17.9 GB MaxRSS, and evaluations never overlap (one queue, one at a time).
    assert "#SBATCH --mem=38G" in script
    for row in manifest["runs"]:
        assert row["resolved_args"]["practice_step_budget"] == 200_000


def test_della_overnight_memory_is_unchanged_by_the_long_stage():
    assert KitchenLaunch.della_memory(evaluation_workers=9) == "27G"
    assert KitchenLaunch.della_memory(evaluation_workers=9, stage="long") == "38G"


@pytest.mark.parametrize("stage", ["pilot", "overnight", "long", "dry"])
def test_only_curtis_carries_the_1600_iteration_search_budget(*, stage):
    # At 100 iterations the zero-heuristic search never reaches reset -> pour behind the
    # no-op Wait skill, so STOP wins and the run ends on the repeated-STOP limit.
    for run in KitchenLaunch.runs(stage=stage, results_root=Path("/unused")):
        argv = run.command[3:]
        arm = run.output_dir.parts[-3]
        if arm == "curtis":
            at = argv.index("--pomdp-max-search-iterations")
            assert argv[at + 1] == "1600"
            assert argv.count("--pomdp-max-search-iterations") == 1
        else:
            assert "--pomdp-max-search-iterations" not in argv
