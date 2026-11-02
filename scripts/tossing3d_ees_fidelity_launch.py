"""Prepare and validate the Tossing3D fidelity matrix; never execute or submit runs."""

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.cli import Cli
from hitl_pmp.core.practice_costs import PracticeCosts
from hitl_pmp.methods.belief_space.tossing3d_method import Tossing3DPomdpMethod
from hitl_pmp.methods.practice_makes_perfect.ees_method import EesMethod
from hitl_pmp.methods.practice_makes_perfect.wrapped_sampler import LearnedSkillSampler
from scripts.tossing3d_launch import EES, FULL, PILOT, REPO, Tossing3DArm, Tossing3DLaunch

SIZES = {"full": FULL, "pilot": PILOT}
SEEDS = {"full": (0, 1, 2), "pilot": (0, 1)}
SOURCES = (
    "scripts/tossing3d_ees_fidelity_launch.py",
    "scripts/tossing3d_launch.py",
    "scripts/sweepoffshelf3d_launch.py",
    "src/hitl_pmp/cli.py",
    "src/hitl_pmp/methods/practice_makes_perfect/cli.py",
    "src/hitl_pmp/methods/practice_makes_perfect/ees_method.py",
    "src/hitl_pmp/methods/practice_makes_perfect/wrapped_sampler.py",
    "src/hitl_pmp/methods/belief_space/tossing3d_method.py",
    "src/hitl_pmp/methods/belief_space/practice_domain.py",
    "src/hitl_pmp/environments/tossing3d/cli.py",
    "src/hitl_pmp/environments/tossing3d/step_protocol.py",
    "src/hitl_pmp/presets/preset.py",
    "src/hitl_pmp/presets/arm_costs.py",
    "src/hitl_pmp/presets/tossing3d-canonical.json",
    "src/hitl_pmp/presets/tossing3d_ees_tick_priced_costs.json",
    "src/hitl_pmp/presets/tossing3d_tick_priced_costs.json",
)
PENDING_PROVENANCE = (
    "freeze-override rationale/approval if deviating from the approved horizon-5 baseline",
    "final clean deployed revision and complete source/diff digest",
    "reference/submodule revisions, simulator assets and dependency/environment versions",
    "runtime config snapshots and cost-file checksums from deployed checkout",
    "diagnostic schema/flag review with agent A and goal/CLI behavioral test evidence",
    "pilot results, resource measurements and explicit launch authorization",
)


class FreezeOverrides(BaseModel):
    """Optional shared numerical freezes, applied identically to every arm."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    exploration_epsilon: float = Field(default=0.5, ge=0, le=1)
    sampler_max_train_iters: int = Field(default=10000, gt=0)
    competence_window_size: int = Field(default=5, gt=0)
    competence_recency_size: int = Field(default=5, gt=0)
    planning_timeout: float = Field(default=10.0, gt=0)


def positive_horizon(*, value: int | None) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("an explicitly supplied positive integer --goal-horizon is required")
    return value


def arms(*, goal_horizon: int) -> list[Tossing3DArm]:
    """Sixteen EES factorial cells, then the two paired planner comparisons."""
    positive_horizon(value=goal_horizon)
    canonical = Tossing3DLaunch.arms()
    result = []
    for sharing, groundings, skip, goal in product(
        ("parent", "grounded"), ("observed", "all"), (True, False), (False, True)
    ):
        name = (
            f"ees-{sharing}-{groundings}-skipperfect{str(skip).lower()}-"
            f"goal{'on' if goal else 'off'}"
        )
        flags = [
            *canonical["ees"].flags,
            "--competence-sharing",
            sharing,
            "--practice-groundings",
            groundings,
            "--ees-score-human-skills",
            "--record-ees-decisions",
            "--reproduce-predicators-skip-perfect"
            if skip
            else "--no-reproduce-predicators-skip-perfect",
            "--goal-pursuit-horizon",
            str(goal_horizon if goal else 0),
        ]
        result.append(Tossing3DArm(name=name, method="ees", flags=flags, always_ask=False))
    for name, goal in product(("expectimax", "determinized-curtis"), (False, True)):
        base = canonical[name]
        flags = [
            *base.flags,
            *EES,
            "--goal-pursuit-horizon",
            str(goal_horizon if goal else 0),
        ]
        if name == "determinized-curtis":
            flags += ["--pomdp-max-search-iterations", "100", "--pomdp-draw-scope", "node"]
        result.append(
            base.model_copy(
                update={
                    "name": f"{name}-goal{'on' if goal else 'off'}",
                    "flags": flags,
                }
            )
        )
    return result


def source_checks() -> dict:
    """Check inherited learner defaults without constructing a simulator or learner."""
    if REPO.resolve() != Path(__file__).resolve().parents[1]:
        raise ValueError("canonical launcher imported from the wrong checkout")
    inherited = {
        "exploration_epsilon": 0.5,
        "sampler_max_train_iters": 10000,
        "num_candidates": 100,
        "max_proposals_per_candidate": 100,
        "score_human_skills": True,
    }
    for model in (EesMethod, Tossing3DPomdpMethod):
        for name, expected in inherited.items():
            if model.model_fields[name].default != expected:
                raise ValueError(f"source default drift: {model.__name__}.{name}")
    sampler = {"exploration_epsilon": 0.5, "num_candidates": 100, "learning_rate": 0.001}
    for name, expected in sampler.items():
        if LearnedSkillSampler.model_fields[name].default != expected:
            raise ValueError(f"source default drift: LearnedSkillSampler.{name}")
    return {"inherited_method_defaults": inherited, "sampler_defaults": sampler}


def parsed_config(*, argv: list[str]) -> dict:
    return json.loads(json.dumps(vars(Cli.parse_args(argv=argv)), default=str))


def check_config(
    *, config: dict, arm: Tossing3DArm, size: str, horizon: int, freeze: FreezeOverrides
) -> None:
    expected = {
        **freeze.model_dump(),
        "env": "tossing3d",
        "variant": "o1",
        "method": arm.method,
        "goal_pursuit_horizon": horizon if arm.name.endswith("-goalon") else 0,
        "goal_pursuit_interval": 1,
        "goal_pursuit_init_cycles": 1,
        "reproduce_predicators_explore_target_only": True,
        "reproduce_predicators_double_observe": False,
        "reproduce_predicators_practice_target_history": True,
        "reproduce_predicators_seen_task_order": True,
        "reproduce_predicators_random_when_stranded": True,
        "record_sampler_draws": True,
        "practice_step_budget": SIZES[size].practice_steps,
        "measurement_interval_steps": 3200,
        "num_test_tasks": 10,
        "evaluation_control_steps": 500,
        "stop_after_perfect_evaluations": 0,
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
    }
    if arm.method == "ees":
        _, sharing, groundings, skip, _ = arm.name.split("-")
        expected.update(
            competence_sharing=sharing,
            practice_groundings=groundings,
            reproduce_predicators_skip_perfect=skip == "skipperfecttrue",
            ees_score_human_skills=True,
            record_ees_decisions=True,
        )
    else:
        curtis = arm.name.startswith("determinized-curtis")
        expected.update(
            preset="tossing3d-canonical",
            pomdp_search_depth=6,
            pomdp_solver="determinized_astar" if curtis else "expectimax",
            pomdp_observation_probability_weight=0.01 if curtis else 0.0,
            pomdp_linear_cost_lambda=0.0003,
            pomdp_inference_engine="grid",
            pomdp_num_particles=1024,
            ees_score_human_skills=None,
            record_ees_decisions=False,
            competence_sharing="grounded",
            practice_groundings="observed",
        )
        if curtis:
            expected.update(
                pomdp_determinized_variant="curtis",
                pomdp_max_search_iterations=100,
                pomdp_draw_scope="node",
                pomdp_num_rollouts=None,
            )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{arm.name}: {key} must be {value!r}, got {config.get(key)!r}")
    costs = PracticeCosts.load(path=Path(config["practice_cost_config"]))
    if costs.robot_step.value != 0.01 or costs.objective_lambda != 0.0003:
        raise ValueError("canonical cost source drift")


def build_manifest(
    *, size: str, goal_horizon: int, results_root: Path, freeze_overrides: dict | None = None
) -> dict:
    """Deterministic seed-major JSON data, with no worker or scheduler side effects."""
    positive_horizon(value=goal_horizon)
    if size not in SIZES:
        raise ValueError("size must be full or pilot")
    if not results_root.is_absolute():
        raise ValueError("results-root must be an absolute path")
    results_root = results_root.resolve()
    freeze = FreezeOverrides.model_validate(freeze_overrides or {})
    evidence = source_checks()
    matrix = arms(goal_horizon=goal_horizon)
    rows = []
    for seed, arm in product(SEEDS[size], matrix):
        output_dir = results_root / size / arm.name / arm.method / str(seed)
        argv = Tossing3DLaunch.cli_argv(
            arm=arm,
            size=SIZES[size],
            seed=seed,
            output_dir=output_dir,
        )
        argv += [
            "--goal-pursuit-init-cycles",
            "1",
            "--goal-pursuit-interval",
            "1",
            "--reproduce-predicators-explore-target-only",
            "--no-ees-reset-gate",
            "--no-bin-pouring",
            "--stop-after-perfect-evaluations",
            "0",
        ]
        for key, value in freeze.model_dump().items():
            argv += ["--" + key.replace("_", "-"), str(value)]
        config = parsed_config(argv=argv)
        check_config(config=config, arm=arm, size=size, horizon=goal_horizon, freeze=freeze)
        rows.append({
            "id": f"{size}-{arm.name}-s{seed}",
            "arm": arm.name,
            "seed": seed,
            "output_dir": str(output_dir),
            "argv": argv,
            "config": config,
        })
    expected_pairs = [(arm.name, seed) for seed in SEEDS[size] for arm in matrix]
    if len(matrix) != 20 or [(row["arm"], row["seed"]) for row in rows] != expected_pairs:
        raise ValueError("matrix/order mismatch")
    for key in ("id", "output_dir"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {key}")
    return {
        "schema_version": 1,
        "status": "preparation-only; not launch authorization",
        "options": {
            "size": size,
            "goal_horizon": goal_horizon,
            "results_root": str(results_root),
            "freeze_overrides": freeze_overrides or {},
        },
        "resolved_freeze": freeze.model_dump(),
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
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                text=True,
            ).strip(),
            "sha256": {
                name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCES
            },
            "checks": evidence,
            "scope": "selected local sources only; HEAD is not a clean/deployed attestation",
        },
        "provenance_pending": list(PENDING_PROVENANCE),
        "runs": rows,
    }


def validate_manifest(*, manifest: dict) -> None:
    """Rebuild against current sources; reject altered matrix, argv, configs or provenance."""
    options = dict(manifest["options"])
    options["results_root"] = Path(options["results_root"])
    expected = build_manifest(**options)
    if manifest != expected:
        raise ValueError("manifest differs from the frozen matrix or current source; regenerate")


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "check"):
        command = modes.add_parser(mode)
        command.add_argument("--size", choices=SIZES, default="full")
        command.add_argument("--goal-horizon", type=int, required=True)
        command.add_argument("--results-root", type=Path, required=True)
        command.add_argument("--freeze-overrides", type=Path, help="Shared numerical JSON object")
        if mode == "prepare":
            command.add_argument(
                "--out", type=Path, required=True, help="New JSON file; no overwrite"
            )
    validate = modes.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            manifest = json.loads(args.manifest.read_text())
            validate_manifest(manifest=manifest)
        else:
            overrides = (
                json.loads(args.freeze_overrides.read_text()) if args.freeze_overrides else {}
            )
            if not isinstance(overrides, dict):
                raise ValueError("freeze-overrides must be a JSON object")
            manifest = build_manifest(
                size=args.size,
                goal_horizon=args.goal_horizon,
                results_root=args.results_root,
                freeze_overrides=overrides,
            )
            if args.mode == "prepare":
                with args.out.open("x") as output:
                    output.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(f"Validated {len(manifest['runs'])} runs; preparation only, provenance pending.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
