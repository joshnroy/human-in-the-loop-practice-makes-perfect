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
        assert args.pour_tilt_bounds == [75, 95]
        assert args.pour_amplitude_bounds == [0, 0.1]
        assert args.pour_duration_bounds == [0, 60]
        costs = PracticeCosts.load(path=args.practice_cost_config)
        assert costs.robot_step.value == 0.01
        assert costs.objective_lambda == 0.0003
        assert set(costs.human_skills) == {"human_bulk_pour_back"}
        assert costs.human_skills["human_bulk_pour_back"].cost.value == 5
        assert costs.human_skills["human_bulk_pour_back"].duration.value == 1
        if args.method == "ees":
            assert args.ees_score_human_skills
            assert args.practice_groundings == "observed"
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
                assert args.pomdp_max_search_iterations == 100
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
        else:
            assert method.pomdp_solver == args.pomdp_solver
            assert method.pomdp_search_depth == (
                None if args.pomdp_solver == "determinized_astar" else 6
            )
            assert method.pomdp_domain.sampler_skills == {"Pour1", "Pour2", "Pour3"}
            assert method.pomdp_domain.human_skill_names == {"human_bulk_pour_back"}
            assert "PickShrimpFromSurface" in dict(method.pomdp_domain.belief_sources)
            assert method.ees_decision_log is None

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
