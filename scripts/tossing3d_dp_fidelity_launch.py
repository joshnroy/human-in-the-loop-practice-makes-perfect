"""Prepare and validate source-bound DP fidelity manifests; never run or submit jobs."""

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path

from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
from hitl_pmp.methods.practice_makes_perfect.dp_method import DpMethod
from hitl_pmp.methods.practice_makes_perfect.dp_priors import DpPriorCalibration
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler
from scripts.tossing3d_dp_priors import (
    FIT_SEEDS,
    SIMULATOR_PINS,
    VALIDATION_SEEDS,
    Tossing3DDpPriors,
)
from scripts.tossing3d_ees_fidelity_launch import SEEDS, SIZES, parsed_config
from scripts.tossing3d_launch import REPO, Tossing3DLaunch

ARMS = (
    ("dp-grounded-episodes-once2000", "grounded", "episodes", "once"),
    ("dp-shared-episodes-once2000", "parent", "episodes", "once"),
    ("dp-grounded-cost-period2000", "grounded", "cost", "period"),
    ("dp-shared-cost-period2000", "parent", "cost", "period"),
)


def digest(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    sources = {
        path for path in (REPO / "src/hitl_pmp").rglob("*") if path.suffix in {".py", ".json"}
    }
    sources.update(
        REPO / name
        for name in (
            "scripts/tossing3d_dp_fidelity_launch.py",
            "scripts/tossing3d_dp_priors.py",
            "scripts/tossing3d_ees_fidelity_launch.py",
            "scripts/tossing3d_launch.py",
            "scripts/sweepoffshelf3d_launch.py",
            "pyproject.toml",
        )
    )
    return {str(path.relative_to(REPO)): digest(path=path) for path in sorted(sources)}


def verify_calibration(*, prior_calibration: Path, calibration_root: Path, open_root: Path) -> dict:
    """Re-fit saved probes, without simulation, and require current collection sources."""
    calibration = DpPriorCalibration.load(path=prior_calibration)
    report = json.loads(prior_calibration.read_text())
    current = Tossing3DDpPriors.source_hashes()
    if calibration.source_sha256 != current:
        raise ValueError("fresh calibration must cover all current calibration sources")
    driver = current["scripts/tossing3d_dp_priors.py"]
    for field in ("collection_driver_sha256", "open_gripper_collection_driver_sha256"):
        if report.get(field) != [driver]:
            raise ValueError("fresh calibration requires current collection-driver provenance")
    if calibration.simulator_pins != SIMULATOR_PINS:
        raise ValueError("fresh calibration requires the complete simulator pins")
    calibration.verify_simulators()
    if (calibration.environment, calibration.layout, calibration.variant) != (
        "tossing3d",
        "barrier",
        "o1",
    ) or report.get("enable_bin_pouring") is not False:
        raise ValueError("calibration must match barrier/o1 with pouring disabled")
    if calibration.fit_seeds != list(FIT_SEEDS) or calibration.validation_seeds != list(
        VALIDATION_SEEDS
    ):
        raise ValueError("calibration requires the independent reserved fit/validation splits")
    provider = Tossing3DSkillProvider(env=Tossing3DEnvironment(enable_bin_pouring=False))
    calibration.verify_robot_skills(skills=provider.skills())
    aggregation = report.get("diagnostics", {}).get("OpenGripper", {}).get("aggregation", {})
    if aggregation.get("name") != "equal-context":
        raise ValueError("calibration must explicitly record equal-context OpenGripper aggregation")
    evidence_files = [
        root / str(seed) / name
        for root in (calibration_root, open_root)
        for seed in (*FIT_SEEDS, *VALIDATION_SEEDS)
        for name in ("metadata.json", "rollouts.jsonl")
    ]
    evidence = {str(path): digest(path=path) for path in evidence_files}
    refitted = Tossing3DDpPriors.fit(
        root=calibration_root,
        open_root=open_root,
        enable_bin_pouring=False,
        open_aggregation=aggregation["name"],
    )
    if report != refitted:
        raise ValueError("calibration differs from independently re-fitted raw evidence")
    if evidence != {str(path): digest(path=path) for path in evidence_files}:
        raise ValueError("calibration evidence changed during verification")
    return {
        "path": str(prior_calibration),
        "sha256": digest(path=prior_calibration),
        "raw_evidence_sha256": evidence,
        "simulator_pins": calibration.simulator_pins,
        "open_aggregation": aggregation,
        "verification": "current-source collection, raw-evidence refit, held-out validation",
    }


def check_config(*, config: dict, sharing: str, unit: str, schedule: str, size: str) -> None:
    expected = {
        "env": "tossing3d",
        "variant": "o1",
        "method": "dp",
        "competence_sharing": sharing,
        "practice_groundings": "all",
        "dp_budget": 2000.0,
        "dp_budget_unit": unit,
        "dp_allocation_schedule": schedule,
        "dp_allow_grounding_substitution": False,
        "dp_solver": "gurobi",
        "goal_pursuit_horizon": 0,
        "goal_pursuit_interval": 1,
        "goal_pursuit_init_cycles": 1,
        "reproduce_predicators_explore_target_only": True,
        "reproduce_predicators_skip_perfect": False,
        "reproduce_predicators_double_observe": False,
        "reproduce_predicators_practice_target_history": True,
        "reproduce_predicators_seen_task_order": True,
        "reproduce_predicators_random_when_stranded": True,
        "exploration_epsilon": 0.5,
        "sampler_max_train_iters": 10000,
        "competence_recency_size": 5,
        "competence_window_size": 5,
        "planning_timeout": 10.0,
        "practice_step_budget": SIZES[size].practice_steps,
        "measurement_interval_steps": 3200,
        "num_test_tasks": 10,
        "evaluation_control_steps": 500,
        "stop_after_perfect_evaluations": 0,
        "ees_score_human_skills": True,
        "ees_reset_gate": False,
        "always_ask_reset": False,
        "enable_bin_pouring": False,
        "bin_mass_kg": 1.0,
        "cube_mass_kg": 0.1,
        "layout": "barrier",
        "evaluation_layout": "barrier",
        "max_steps_per_interaction": 20,
        "practice_reset_policy": "never",
        "test_env_seed_offset": 10000,
        "record_sampler_draws": True,
        "record_ees_decisions": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"DP protocol drift: {key} must be {value!r}")
    for model, defaults in (
        (DpMethod, {"num_candidates": 100, "max_proposals_per_candidate": 100}),
        (LearnedSkillSampler, {"learning_rate": 0.001, "num_candidates": 100}),
    ):
        for key, value in defaults.items():
            if model.model_fields[key].default != value:
                raise ValueError(f"inherited learner source drift: {model.__name__}.{key}")
    costs = PracticeCosts.load(path=Path(config["practice_cost_config"]))
    if costs.robot_step.value != 0.01 or costs.objective_lambda != 0.0003:
        raise ValueError("canonical cost source drift")


def build_manifest(
    *,
    size: str,
    results_root: Path,
    prior_calibration: Path,
    calibration_root: Path,
    open_root: Path,
) -> dict:
    if size not in SIZES:
        raise ValueError("size must be full or pilot")
    paths = {
        "results_root": results_root,
        "prior_calibration": prior_calibration,
        "calibration_root": calibration_root,
        "open_root": open_root,
    }
    if not all(path.is_absolute() for path in paths.values()):
        raise ValueError("all paths must be absolute; prepare on the execution checkout")
    paths = {name: path.resolve() for name, path in paths.items()}
    if REPO.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError("canonical launcher imported from the wrong checkout")
    sources = source_hashes()
    prior = verify_calibration(**{
        name: path for name, path in paths.items() if name != "results_root"
    })
    rows = []
    for seed, (name, sharing, unit, schedule) in product(SEEDS[size], ARMS):
        output_dir = paths["results_root"] / size / name / "dp" / str(seed)
        argv = Tossing3DLaunch.cli_argv(
            arm=Tossing3DLaunch.arms()["ees"], size=SIZES[size], seed=seed, output_dir=output_dir
        )
        argv[argv.index("--method") + 1] = "dp"
        argv += [
            "--competence-sharing",
            sharing,
            "--practice-groundings",
            "all",
            "--dp-budget",
            "2000",
            "--dp-budget-unit",
            unit,
            "--dp-allocation-schedule",
            schedule,
            "--no-dp-allow-grounding-substitution",
            "--dp-solver",
            "gurobi",
            "--dp-prior-calibration",
            str(paths["prior_calibration"]),
            "--goal-pursuit-horizon",
            "0",
            "--goal-pursuit-interval",
            "1",
            "--goal-pursuit-init-cycles",
            "1",
            "--reproduce-predicators-explore-target-only",
            "--no-reproduce-predicators-skip-perfect",
            "--ees-score-human-skills",
            "--no-ees-reset-gate",
            "--no-bin-pouring",
            "--stop-after-perfect-evaluations",
            "0",
        ]
        config = parsed_config(argv=argv)
        check_config(config=config, sharing=sharing, unit=unit, schedule=schedule, size=size)
        rows.append({
            "id": f"{size}-{name}-s{seed}",
            "arm": name,
            "seed": seed,
            "output_dir": str(output_dir),
            "argv": argv,
            "config": config,
        })
    cells = {(sharing, unit, schedule) for _, sharing, unit, schedule in ARMS}
    if (
        cells
        != set(product(("grounded", "parent"), ("episodes",), ("once",)))
        | set(product(("grounded", "parent"), ("cost",), ("period",)))
        or len(ARMS) != 4
    ):
        raise ValueError("DP matrix mismatch")
    for key in ("id", "output_dir"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {key}")
    if sources != source_hashes():
        raise ValueError("source changed during preparation; retry after freeze")
    return {
        "schema_version": 1,
        "status": "preparation-only; no jobs submitted",
        "options": {"size": size, **{name: str(path) for name, path in paths.items()}},
        "schedule": {
            "ordering": "seed-major",
            "dependencies": [],
            "barriers": [],
            "max_concurrency": None,
            "concurrency_policy": "maximum scheduler permits",
        },
        "source": {
            "checkout": str(REPO),
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "sha256": sources,
            "scope": (
                "local Python/JSON sources and preparation inputs; "
                "not a clean-deployment attestation"
            ),
        },
        "calibration": prior,
        "eligibility": "deployment-reachable untried robot skills included; no skip-perfect axis",
        "provenance_pending": [
            "clean integration/launch revision, matching runtime dependencies and simulator assets",
            "validate on execution checkout: absolute prior, cost and evidence paths are local",
            "solver license/certification, pilot results and resource measurements",
        ],
        "runs": rows,
    }


def validate_manifest(*, manifest: dict) -> None:
    options = {
        name: value if name == "size" else Path(value)
        for name, value in manifest["options"].items()
    }
    if manifest != build_manifest(**options):
        raise ValueError(
            "manifest differs from current source, calibration or frozen matrix; regenerate"
        )


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("check", "prepare"):
        command = modes.add_parser(mode)
        command.add_argument("--size", choices=SIZES, default="full")
        for name in ("results-root", "prior-calibration", "calibration-root", "open-root"):
            command.add_argument(f"--{name}", type=Path, required=True)
        if mode == "prepare":
            command.add_argument("--out", type=Path, required=True)
    validate = modes.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            manifest = json.loads(args.manifest.read_text())
            validate_manifest(manifest=manifest)
        else:
            manifest = build_manifest(
                size=args.size,
                results_root=args.results_root,
                prior_calibration=args.prior_calibration,
                calibration_root=args.calibration_root,
                open_root=args.open_root,
            )
            if args.mode == "prepare":
                with args.out.open("x") as output:
                    output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(
        f"Validated {len(manifest['runs'])} DP runs; no jobs submitted; deployment gates pending."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
