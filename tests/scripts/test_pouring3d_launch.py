import json
from pathlib import Path

import pytest

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from scripts.pouring3d_launch import PouringLaunch


@pytest.mark.parametrize("stage,budget", [("pilot", 6400), ("long-pilot", 25600)])
def test_paired_matrix_and_evaluation_contract(*, stage, budget):
    runs = PouringLaunch.runs(stage=stage, results_root=Path("/unused"))
    assert len(runs) == 6
    assert len({r.output_dir for r in runs}) == 6
    assert [(r.output_dir.parts[-3], r.seed) for r in runs] == [
        (arm, seed) for arm in ("ees", "curtis", "expectimax") for seed in (0, 2)
    ]
    for run in runs:
        args = Cli.parse_args(argv=run.command[3:])
        assert args.practice_step_budget == budget
        assert args.measurement_interval_steps == 3200
        assert args.initial_measurement
        assert args.num_test_tasks == 9
        assert args.evaluation_control_steps == 2500
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
        assert args.pour_tilt_bounds == [87, 95]
        assert args.pour_amplitude_bounds == [0, 0.025]
        assert args.pour_duration_bounds == [0, 30]
        costs = PracticeCosts.load(path=args.practice_cost_config)
        assert costs.robot_step.value == 0.01
        assert costs.objective_lambda == 0.0003
        assert set(costs.human_skills) == {"reset_shrimp"}
        assert costs.human_skills["reset_shrimp"].cost.value == 5
        assert costs.human_skills["reset_shrimp"].duration.value == 1
        if args.method == "ees":
            assert args.ees_score_human_skills
            assert not args.reproduce_predicators_skip_perfect
            assert args.reproduce_predicators_random_when_stranded
        else:
            assert args.pomdp_search_depth == 6
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
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "planned"
    assert len(manifest["runs"]) == 6
    assert all(row["resolved_args"]["initial_measurement"] for row in manifest["runs"])
    assert len(manifest["cost_sha256"]) == 64
    with pytest.raises(FileExistsError):
        PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")


def test_run_refuses_changed_code_before_spawning(*, tmp_path, monkeypatch):
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "b" * 40})
    with pytest.raises(RuntimeError, match="identity"):
        PouringLaunch.run(manifest=path, max_workers=1)


def test_run_refuses_modified_manifest_commands(*, tmp_path, monkeypatch):
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    data = json.loads(path.read_text())
    data["runs"][0]["argv"].append("--always-ask-reset")
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="commands"):
        PouringLaunch.run(manifest=path, max_workers=1)


def test_run_refuses_existing_results(*, tmp_path, monkeypatch):
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    runs = PouringLaunch.runs(stage="pilot", results_root=tmp_path / "results")
    runs[-1].output_dir.mkdir(parents=True)
    (runs[-1].output_dir / "stats.json").write_text("{}")
    with pytest.raises(FileExistsError):
        PouringLaunch.run(manifest=path, max_workers=1)


def test_array_worker_only_executes_its_row(*, tmp_path, monkeypatch):
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    monkeypatch.setattr(PouringLaunch, "native_check", lambda: {"ready": True})
    path = tmp_path / "manifest.json"
    root = tmp_path / "results"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=root)
    runs = PouringLaunch.runs(stage="pilot", results_root=root)
    # A completed sibling task must not block this array worker.
    runs[0].output_dir.mkdir(parents=True)
    observed = []

    def execute(**kwargs):
        observed.extend(kwargs["runs"])
        return []

    monkeypatch.setattr("scripts.pouring3d_launch.SweepRunner.execute", execute)
    assert PouringLaunch.run(manifest=path, max_workers=1, task=3) == 0
    assert observed == [runs[2]]
    receipt = json.loads((runs[2].output_dir / "launch.json").read_text())
    assert receipt["run"]["arm"] == "curtis"
    assert receipt["run"]["seed"] == 0
    assert not runs[3].output_dir.exists()


@pytest.mark.parametrize("task", [0, 7])
def test_array_index_is_checked(*, task, tmp_path, monkeypatch):
    monkeypatch.setattr(PouringLaunch, "identity", lambda **_: {"commit": "a" * 40})
    path = tmp_path / "manifest.json"
    PouringLaunch.write_manifest(path=path, stage="pilot", results_root=tmp_path / "results")
    with pytest.raises(ValueError, match="Task index"):
        PouringLaunch.run(manifest=path, max_workers=1, task=task)
