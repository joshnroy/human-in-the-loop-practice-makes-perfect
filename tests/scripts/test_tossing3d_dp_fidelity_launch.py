import copy
import json
from itertools import product
from pathlib import Path

import pytest

from hitl_pmp.cli import METHODS, Cli
from hitl_pmp.config_snapshot import ConfigSnapshot
from hitl_pmp.core.method.skill_provider import DomainContext
from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DOracle, Tossing3DSkillProvider
from scripts import tossing3d_dp_fidelity_launch as launcher
from scripts.tossing3d_dp_priors import FIT_SEEDS, SIMULATOR_PINS, Tossing3DDpPriors
from tests.scripts.test_tossing3d_dp_priors import curves as curves
from tests.scripts.test_tossing3d_dp_priors import open_probes as open_probes


@pytest.fixture
def options(*, curves, open_probes, monkeypatch):
    report = Tossing3DDpPriors.fit(
        root=curves,
        open_root=open_probes,
        enable_bin_pouring=False,
        open_aggregation="equal-context",
    )
    path = curves / "prior.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(
        ConfigSnapshot,
        "_repo_state",
        staticmethod(lambda *, package: (SIMULATOR_PINS[package], False)),
    )
    return {
        "size": "full",
        "results_root": curves / "results",
        "prior_calibration": path,
        "calibration_root": curves,
        "open_root": open_probes,
    }


@pytest.mark.parametrize("size", ["full", "pilot"])
def test_complete_matrix_seed_major_and_common_settings(*, options, size):
    options["size"] = size
    manifest = launcher.build_manifest(**options)
    rows = manifest["runs"]
    seeds = (0, 1, 2) if size == "full" else (0, 1)
    assert [(row["arm"], row["seed"]) for row in rows] == [
        (arm[0], seed) for seed in seeds for arm in launcher.ARMS
    ]
    assert len(rows) == 4 * len(seeds)
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["output_dir"] for row in rows}) == len(rows)
    assert {
        (
            row["config"]["competence_sharing"],
            row["config"]["dp_budget_unit"],
            row["config"]["dp_allocation_schedule"],
        )
        for row in rows
    } == {
        (sharing, unit, schedule)
        for sharing, (unit, schedule) in product(
            ("grounded", "parent"), (("episodes", "once"), ("cost", "period"))
        )
    }
    for row in rows:
        args = Cli.parse_args(argv=row["argv"])
        assert args.method == "dp"
        assert args.dp_budget == 2000
        assert args.dp_solver == "gurobi"
        assert args.dp_prior_calibration == options["prior_calibration"]
        assert not args.dp_allow_grounding_substitution
        assert args.practice_groundings == "all"
        assert not args.reproduce_predicators_skip_perfect
        assert args.practice_step_budget == (160000 if size == "full" else 6400)
        assert (
            args.measurement_interval_steps,
            args.num_test_tasks,
            args.evaluation_control_steps,
        ) == (3200, 10, 500)
        assert args.stop_after_perfect_evaluations == 0
        assert args.goal_pursuit_horizon == 0
        assert args.goal_pursuit_init_cycles == args.goal_pursuit_interval == 1
        assert args.reproduce_predicators_explore_target_only
        assert args.ees_score_human_skills
        assert not args.ees_reset_gate and not args.always_ask_reset
        assert not args.enable_bin_pouring
        assert (args.bin_mass_kg, args.cube_mass_kg) == (1.0, 0.1)
        assert args.exploration_epsilon == 0.5
        assert args.sampler_max_train_iters == 10000
        assert args.seed == row["seed"]
        assert str(args.output_dir) == row["output_dir"]
    assert manifest["schedule"] == {
        "ordering": "seed-major",
        "dependencies": [],
        "barriers": [],
        "max_concurrency": None,
        "concurrency_policy": "maximum scheduler permits",
    }
    assert len(manifest["calibration"]["raw_evidence_sha256"]) == 24
    launcher.validate_manifest(manifest=manifest)
    assert not options["results_root"].exists()


def test_actual_dp_cli_factory_preserves_protocol(*, options, monkeypatch):
    captured = []

    def capture(*, args, method_factory, **unused):
        env = Tossing3DEnvironment(scene_bg=False, enable_bin_pouring=False)
        method = method_factory(
            DomainContext(
                env=env,
                skill_provider=Tossing3DSkillProvider(env=env),
                oracle=Tossing3DOracle(env=env),
            )
        )
        captured.append(method)
        for name in (
            "competence_sharing",
            "practice_groundings",
            "dp_budget",
            "dp_budget_unit",
            "dp_allocation_schedule",
            "dp_allow_grounding_substitution",
            "dp_solver",
            "goal_pursuit_horizon",
            "goal_pursuit_interval",
            "goal_pursuit_init_cycles",
            "reproduce_predicators_explore_target_only",
            "reproduce_predicators_skip_perfect",
            "exploration_epsilon",
            "sampler_max_train_iters",
            "competence_window_size",
            "competence_recency_size",
            "reproduce_predicators_double_observe",
            "reproduce_predicators_practice_target_history",
            "reproduce_predicators_seen_task_order",
            "reproduce_predicators_random_when_stranded",
        ):
            assert getattr(method, name) == getattr(args, name)
        assert method.score_human_skills is True
        assert method.reset_cost_gate is False
        assert method.always_ask_reset is None
        assert method.draw_recorder is not None
        assert method.ees_decision_log is None
        assert method.dp_solver_log == args.output_dir / "dp_allocations.jsonl"
        assert method.dp_prior_calibration == options["prior_calibration"]
        assert not method.pursues_goal_in_period(period=0)

    monkeypatch.setattr(Tossing3DCli, "run_method", staticmethod(capture))
    for row in launcher.build_manifest(**options)["runs"][:4]:
        args = Cli.parse_args(argv=row["argv"])
        METHODS[args.method].run(args=args, env_cli=Tossing3DCli)
    assert len(captured) == 4
    assert not options["results_root"].exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "order", "argv", "config", "source"])
def test_tampered_manifest_rejected(*, options, change):
    manifest = copy.deepcopy(launcher.build_manifest(**options))
    rows = manifest["runs"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "argv":
        rows[0]["argv"].append("--dp-allow-grounding-substitution")
    elif change == "config":
        rows[0]["config"]["goal_pursuit_horizon"] = 5
    else:
        manifest["source"]["head"] = "0" * 40
    with pytest.raises(ValueError, match="frozen matrix"):
        launcher.validate_manifest(manifest=manifest)


@pytest.mark.parametrize(
    "change",
    [
        "old_driver",
        "old_open_driver",
        "missing_sources",
        "stale_source",
        "missing_pins",
        "pouring",
        "old_open_context",
        "fit_value",
        "failed_validation",
        "missing_skill",
    ],
)
def test_old_incomplete_or_modified_prior_rejected(*, options, change):
    path = options["prior_calibration"]
    report = json.loads(path.read_text())
    if change in {"old_driver", "old_open_driver"}:
        key = (
            "collection_driver_sha256"
            if change == "old_driver"
            else "open_gripper_collection_driver_sha256"
        )
        report[key] = ["0" * 64]
    elif change == "missing_sources":
        report["source_sha256"].pop("src/hitl_pmp/step_protocol.py")
    elif change == "stale_source":
        report["source_sha256"]["src/hitl_pmp/step_protocol.py"] = "0" * 64
    elif change == "missing_pins":
        report["simulator_pins"] = {}
    elif change == "pouring":
        report["enable_bin_pouring"] = True
    elif change == "old_open_context":
        report["diagnostics"]["OpenGripper"].pop("aggregation")
    elif change == "fit_value":
        report["skills"]["OpenGripper"]["alpha"] += 0.1
    elif change == "failed_validation":
        report["validation_passed"] = False
    else:
        report["skills"].pop("OpenGripper")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        launcher.build_manifest(**options)


def test_raw_probe_tampering_rejected(*, options):
    path = options["open_root"] / str(FIT_SEEDS[0]) / "rollouts.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(row) for row in rows if row["context"] != "holding"))
    with pytest.raises(ValueError, match="OpenGripper probes"):
        launcher.build_manifest(**options)


def test_modified_simulator_rejected(*, options, monkeypatch):
    monkeypatch.setattr(ConfigSnapshot, "_repo_state", staticmethod(lambda **unused: ("bad", True)))
    with pytest.raises(ValueError, match="simulator mismatch"):
        launcher.build_manifest(**options)


def test_changed_source_during_preparation_rejected(*, options, monkeypatch):
    snapshots = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(launcher, "source_hashes", lambda: next(snapshots))
    with pytest.raises(ValueError, match="source changed"):
        launcher.build_manifest(**options)


def test_relative_paths_rejected(*, options):
    options["prior_calibration"] = Path("prior.json")
    with pytest.raises(ValueError, match="absolute"):
        launcher.build_manifest(**options)


def test_cli_prepares_checks_validates_without_overwrite_or_runs(*, options):
    output = options["calibration_root"] / "manifest.json"
    argv = [
        item
        for name, value in options.items()
        for item in ("--" + name.replace("_", "-"), str(value))
    ]
    assert launcher.main(argv=["check", *argv]) == 0
    assert launcher.main(argv=["prepare", *argv, "--out", str(output)]) == 0
    assert launcher.main(argv=["validate", "--manifest", str(output)]) == 0
    assert len(json.loads(output.read_text())["runs"]) == 12
    with pytest.raises(SystemExit):
        launcher.main(argv=["prepare", *argv, "--out", str(output)])
    assert not options["results_root"].exists()


def test_cli_requires_calibration_evidence(*, tmp_path):
    with pytest.raises(SystemExit):
        launcher.main(
            argv=[
                "prepare",
                "--results-root",
                str(tmp_path),
                "--out",
                str(tmp_path / "manifest.json"),
            ]
        )
    assert not (tmp_path / "manifest.json").exists()
