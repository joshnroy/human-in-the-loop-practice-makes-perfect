"""Fail-closed worker for a frozen kitchen-only manifest; never submits jobs."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("ees", "expectimax", "curtis")
SMOKE_TIMEOUT_SECONDS = 1800
READINESS = (
    "selected_object_pickup",
    "single_object_train_eval",
    "seeded_scatter_reproducible",
    "scatter_across_kitchen",
    "training_deployment_isolation",
    "deployment_no_human_actions",
    "dependency_resolution",
    "final_refit_evaluation",
    "kitchen_costs_approved",
)


# Tossing3D's two human resets, as the kitchen charges them, each at Tossing3D's 5.
HUMAN_RESET_COSTS = {"ResetCubeBinSameSide": 5, "ResetCubeBinOppositeSide": 5}


def check_costs(*, costs: dict, cost: float = 5) -> None:
    priced = {name: entry["cost"]["value"] for name, entry in costs.get("human_skills", {}).items()}
    if priced != {name: cost for name in HUMAN_RESET_COSTS}:
        raise ValueError(f"Cost file must price exactly the two human resets at {cost}: {priced}")


def arms(*, manifest: dict) -> tuple[str, ...]:
    """The manifest's arms in row order; absent, the three every package had before DP."""
    return tuple(manifest.get("arms", ARMS))


def expected_rows(*, manifest: dict) -> list[tuple]:
    """(reset cost, seed, arm) per row: seed-major, then each swept cost in the manifest's
    order, then arm (`None`: no sweep, the nine runs at the cost file's 5)."""
    costs = manifest.get("human_reset_costs", [None])
    return [
        (cost, seed, arm) for seed in range(3) for cost in costs for arm in arms(manifest=manifest)
    ]


def check_dp_row(*, parsed: argparse.Namespace, manifest: dict, row: dict) -> None:
    """A DP row runs DP exactly as the manifest records it (episodes, allocated once,
    Gurobi, the manifest's B and initial competences, no prior calibration); no other row
    runs DP."""
    if row["arm"] != "dp":
        if parsed.method == "dp":
            raise ValueError(f"Only the DP arm may run --method dp, not {row['arm']}")
        return
    expected = {
        "method": "dp",
        "dp_budget_unit": "episodes",
        "dp_allocation_schedule": "once",
        "dp_solver": "gurobi",
        "dp_budget": float(manifest["dp"]["episode_budget"]),
        "dp_prior_calibration": None,
        "dp_initial_competence": manifest["dp"].get("initial_competence", {}),
    }
    actual = {name: getattr(parsed, name, None) for name in expected}
    if actual != expected:
        raise ValueError(f"DP row differs from the manifest's DP contract: {actual}")


def check_row_costs(*, parsed: argparse.Namespace, row: dict) -> None:
    """The row's cost file prices both resets at its cost and is the file the manifest
    bound (when swept). The cost file is the only reset price: both arms read it."""
    cost = row.get("human_reset_cost", 5)
    path = Path(parsed.practice_cost_config)
    path = path if path.is_absolute() else REPO / path
    if "costs_sha256" in row and sha(path=path) != row["costs_sha256"]:
        raise ValueError(f"Cost file changed since the manifest bound it: {path}")
    check_costs(costs=read(path=path), cost=cost)


def read(*, path: Path):
    return json.loads(path.read_text())


def sha(*, path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def contract(*, manifest: dict) -> tuple[int, int]:
    """The manifest's counted-step budget and evaluation workers; absent keys are the
    full contract's 160,000 steps and one evaluation process (prepare.py's defaults)."""
    return manifest.get("practice_step_budget", 160000), manifest.get("evaluation_workers", 1)


def check_budget_and_workers(*, parsed: argparse.Namespace, manifest: dict) -> None:
    budget, workers = contract(manifest=manifest)
    if parsed.practice_step_budget != budget:
        raise ValueError(f"Wrong full step ceiling: expected the manifest's {budget}")
    if parsed.evaluation_workers != workers:
        raise ValueError(f"Wrong evaluation workers: expected the manifest's {workers}")
    if workers > 1 and parsed.evaluation_task_rng != "per-task":
        raise ValueError("Parallel evaluation requires --evaluation-task-rng per-task")


def repeated_stop_limit(*, manifest: dict, row: dict | None = None) -> int | None:
    """A row's STOP-streak limit: its own (the DP rows'), else the manifest's."""
    if row is not None and "repeated_stop_limit" in row:
        return row["repeated_stop_limit"]
    return manifest.get("repeated_stop_limit")


def check_repeated_stop_limit(
    *, parsed: argparse.Namespace, manifest: dict, row: dict | None = None
) -> None:
    """Every row carries its STOP-streak limit, and none without one."""
    expected = repeated_stop_limit(manifest=manifest, row=row)
    if parsed.repeated_stop_limit != expected:
        raise ValueError(
            f"Wrong --repeated-stop-limit {parsed.repeated_stop_limit}: expected the "
            f"manifest's {expected}"
        )


def complete_endpoints(*, manifest: dict, row: dict | None = None) -> set[str]:
    """How a run may end and still count as complete: at the budget, or, under a
    STOP-streak limit, at that limit."""
    if repeated_stop_limit(manifest=manifest, row=row) is None:
        return {"practice_step_budget"}
    return {"practice_step_budget", "repeated_stop"}


SLURM_HOST_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURMD_NODENAME",
    "SLURM_JOB_PARTITION",
)


def cpu_model() -> str:
    """The CPU's model name as /proc/cpuinfo gives it; platform's guess elsewhere."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def host_record() -> dict:
    """Where a run ran. Search results are not bit-reproducible across CPU models, so
    the launch-time equivalence check (equivalence.py) compares only like with like."""
    return {
        "hostname": socket.gethostname(),
        "cpu_model": cpu_model(),
        "slurm": {key: os.environ[key] for key in SLURM_HOST_KEYS if key in os.environ},
    }


def check_equivalence(*, launch: Path, manifest: dict) -> None:
    """Mandatory from v11: the manifest binds an equivalence reference (or a waiver), and
    the gate's passing (or waived) record is bound to this launch and carried verbatim
    into the full authorization's provenance. Launches before v11 ran their own frozen
    copy of this file and are unaffected."""
    block = manifest.get("equivalence")
    if block is None:
        raise ValueError(
            "The manifest records neither an equivalence reference nor a waiver "
            "(prepare.py --equivalence-reference)"
        )
    record = read(path=launch / "equivalence-check.json")
    if record.get("binding") != binding(launch=launch):
        raise ValueError("The equivalence record's binding is not this launch's")
    if block["reference_root"] is None:
        if record.get("waived") is not True or record.get("waiver") != block["waiver"]:
            raise ValueError("The equivalence record does not carry the manifest's waiver")
    elif record.get("passed") is not True or (
        [run["index"] for run in record.get("runs", [])] != [run["index"] for run in block["runs"]]
    ):
        raise ValueError("The equivalence check did not pass for every bound smoke row")
    provenance = read(path=launch / "authorization-provenance.json")
    if provenance.get("equivalence") != record:
        raise ValueError("The authorization provenance must carry the equivalence record")


def binding(*, launch: Path) -> dict:
    return {
        name: sha(path=launch / name)
        for name in ("manifest.json", "snapshot.json", "runtime-lock.json")
    }


def verify(*, launch: Path, full: bool) -> dict:
    manifest = read(path=launch / "manifest.json")
    rows = manifest["runs"]
    expected = expected_rows(manifest=manifest)
    if [(row.get("human_reset_cost"), row["seed"], row["arm"]) for row in rows] != expected:
        raise ValueError("Only the nine seed-major kitchen runs (per swept cost) are permitted")
    if len({output(row=row) for row in rows}) != len(rows):
        raise ValueError("Every run needs its own output directory")
    snapshot = read(path=launch / "snapshot.json")
    if not snapshot["assets_included"]:
        raise ValueError("Portable simulator assets are required")
    for relative, expected_hash in snapshot["files"].items():
        if sha(path=REPO / relative) != expected_hash:
            raise ValueError(f"Frozen source/asset mismatch: {relative}")
    for package, relative in {
        "hitl_pmp": "src/hitl_pmp",
        "kinder": "reference/kindergarden/src/kinder",
        "kinder_models": "reference/kinder-baselines/kinder-models/src/kinder_models",
    }.items():
        module = importlib.import_module(package)
        if not Path(module.__file__).resolve().is_relative_to((REPO / relative).resolve()):
            raise ValueError(f"Import escaped snapshot: {package}: {module.__file__}")
    lock = read(path=launch / "runtime-lock.json")
    for package, version in lock["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"Dependency drift: {package}, expected {version}")
    evidence = read(path=launch / "readiness.json")
    if evidence.get("binding") != binding(launch=launch):
        raise ValueError("Readiness is not bound to this exact launch/snapshot/runtime")
    for check in READINESS if full else READINESS[:4]:
        entry = evidence.get(check, {})
        if entry.get("passed") is not True:
            raise ValueError(f"Missing readiness evidence: {check}")
        path = launch / entry["path"]
        if not path.resolve().is_relative_to(launch.resolve()):
            raise ValueError("Evidence path escapes launch directory")
        if sha(path=path) != entry["sha256"]:
            raise ValueError(f"Evidence changed: {check}")
    if full:
        if lock.get("status") != "RESOLVED" or lock.get("blockers"):
            raise ValueError("Dependency lock is observed-only, not resolved/validated")
        gate = read(path=launch / "smoke-passed.json")
        if gate.get("binding") != binding(launch=launch) or gate.get("passed") is not True:
            raise ValueError("Matching completed smoke gate required")
        if set(gate.get("arms", [])) != set(arms(manifest=manifest)):
            raise ValueError("Smoke must cover every method")
        authorization = read(path=launch / "authorization.json")
        if authorization != {
            "binding": binding(launch=launch),
            "authorized_by": "main",
            "submit_full": True,
        }:
            raise ValueError("Explicit main authorization for this exact package is required")
        check_equivalence(launch=launch, manifest=manifest)
    else:
        authorization = read(path=launch / "smoke-authorization.json")
        if authorization.get("authorized_by") not in {"main", "user"}:
            raise ValueError("Smoke authorization must come from main or the user")
        if authorization != {
            "binding": binding(launch=launch),
            "authorized_by": authorization["authorized_by"],
            "submit_smoke": True,
        }:
            raise ValueError("Explicit main smoke authorization for this package is required")
    from hitl_pmp.cli import Cli
    from hitl_pmp.environments.kitchen_recycling3d.step_protocol import (
        KitchenRecyclingStepProtocol,
    )
    from hitl_pmp.environments.kitchen_recycling3d.tasks import KitchenRecyclingTasks

    if set(KitchenRecyclingStepProtocol.human_skill_names().values()) != set(HUMAN_RESET_COSTS):
        raise ValueError("The kitchen's reset charge names drifted from the cost contract")

    if KitchenRecyclingTasks.model_fields["test_env_seed_offset"].default != 10000:
        raise ValueError("Held-out task seed offset drifted from the canonical 10000")

    for row in rows:
        parsed = Cli.parse_args(argv=[*row["argv"], "--output-dir", "/tmp/kitchen-parse-only"])
        if parsed.env != "kitchen_recycling3d":
            raise ValueError("Wrong environment")
        check_budget_and_workers(parsed=parsed, manifest=manifest)
        check_repeated_stop_limit(parsed=parsed, manifest=manifest, row=row)
        check_dp_row(parsed=parsed, manifest=manifest, row=row)
        if parsed.goal_pursuit_horizon != 0 or parsed.stop_after_perfect_evaluations != 0:
            raise ValueError("Goal pursuit/perfect-evaluation stopping must be disabled")
        if not parsed.no_early_stop:
            raise ValueError("Full-horizon runs require --no-early-stop")
        if parsed.scene_bg is not False:
            raise ValueError("Pilots and full runs must use --no-scene-bg")
        check_row_costs(parsed=parsed, row=row)
        config = read(path=REPO / parsed.kitchen_config)
        if config.get("scatter_objects") is not True:
            raise ValueError("Full experiment config must enable seeded kitchen scatter")
        objects = config.get("objects", [])
        if len(objects) != 1 or config.get("required_objects") != [objects[0]["name"]]:
            raise ValueError("Final config must contain exactly one object, also the sole goal")
        if evidence.get("selected_object") != objects[0]["name"]:
            raise ValueError("Pickup evidence must identify the final selected object")
    return manifest


def output(*, row: dict) -> str:
    """A run's directory under results/ (and smoke/): arm/seed, or the row's own."""
    return row.get("output", f"{row['arm']}/{row['seed']}")


def run(*, launch: Path, row: dict, smoke: bool, manifest: dict) -> dict:
    output_dir = launch.parent / ("smoke" if smoke else "results") / output(row=row)
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "hitl_pmp.cli", *row["argv"], "--output-dir", str(output_dir)]
    budget, workers = contract(manifest=manifest)
    if smoke:
        command += [
            "--practice-step-budget",
            "3200",
            "--measurement-interval-steps",
            "3200",
            # One task per evaluation worker, so the smoke runs them concurrently.
            "--num-test-tasks",
            str(workers),
            "--evaluation-control-steps",
            "500",
        ]
    start = time.monotonic()
    with (output_dir / "worker.log").open("w") as logfile:
        subprocess.run(
            command,
            cwd=REPO,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            timeout=SMOKE_TIMEOUT_SECONDS if smoke else None,
            check=True,
        )
    # After the CLI has exited, so the run never sees it.
    (output_dir / "host.json").write_text(json.dumps(host_record(), indent=2) + "\n")
    status = read(path=output_dir / "status.json")
    evaluations = read(path=output_dir / "step_evaluations.json")
    ends = [
        json.loads(line)
        for line in (output_dir / "step_events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ends = [event for event in ends if event.get("event") == "practice_end"]
    if status.get("phase") != "complete" or not evaluations:
        raise ValueError("Process exit alone is not experiment completion")
    if not all(record.get("complete") for record in evaluations):
        raise ValueError("Incomplete or failed held-out evaluation")
    if evaluations[-1].get("practice_steps") != status.get("practice_steps"):
        raise ValueError("Final held-out evaluation is missing")
    if not ends or ends[-1].get("reason") not in complete_endpoints(manifest=manifest, row=row):
        raise ValueError("Early STOP/stall is not full budget completion")
    # A run the STOP-streak limit ended has stopped practising: its horizon is its own.
    at_budget = ends[-1].get("reason") == "practice_step_budget"
    if smoke and at_budget and status.get("practice_steps", 0) < 3200:
        raise ValueError("Smoke did not reach its practice ceiling")
    if not smoke and at_budget and status.get("practice_steps", 0) < budget:
        raise ValueError(f"Run stopped before the full {budget} counted-step horizon")
    receipt = {
        "arm": row["arm"],
        "seed": row["seed"],
        "binding": binding(launch=launch),
        "elapsed_seconds": time.monotonic() - start,
        "status": status,
        "endpoint": ends[-1],
        "command": command,
    }
    (output_dir / "completion-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "smoke", "full"))
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--task", type=int, default=0)
    args = parser.parse_args()
    launch = args.launch.resolve()
    manifest = verify(launch=launch, full=args.mode == "full")
    if not 0 <= args.task < len(manifest["runs"]):
        parser.error(f"--task must index one of the manifest's {len(manifest['runs'])} runs")
    if args.mode == "check":
        print(json.dumps({"binding": binding(launch=launch), "parsed": len(manifest["runs"])}))
    elif args.mode == "smoke":
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Run physics smoke in an authorized compute allocation, not login")
        receipts = [
            run(launch=launch, row=row, smoke=True, manifest=manifest)
            for row in manifest["runs"][: len(arms(manifest=manifest))]
        ]
        gate = {
            "binding": binding(launch=launch),
            "passed": True,
            "arms": list(arms(manifest=manifest)),
            "receipts": receipts,
        }
        with (launch / "smoke-passed.json").open("x") as stream:
            json.dump(gate, stream, indent=2)
    else:
        run(launch=launch, row=manifest["runs"][args.task], smoke=False, manifest=manifest)


if __name__ == "__main__":
    main()
