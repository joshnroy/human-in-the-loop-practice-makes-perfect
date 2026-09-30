"""Run one frozen Sweep arm through the shared sweep runner."""

import argparse
import json
from pathlib import Path

from run_sweep import SpawnRetryPolicy, SweepRunner


def main():
    from check_slurm_memory import main as check_memory

    check_memory()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    environment = manifest.get("environment", "sweep_drawer3d")
    if environment not in ("sweep_drawer3d", "sweep_simple3d"):
        raise ValueError(f"Unsupported Sweep manifest environment {environment!r}")
    if environment == "sweep_simple3d":
        from hitl_pmp.environments.sweep_simple3d.cli import SweepSimpleCli

        SweepSimpleCli.validate_manifest(manifest=manifest)
    arm = next(a for a in manifest["arms"] if a["name"] == args.arm)
    method = arm["method"]
    flags = [
        "--sweep-manifest",
        str(args.manifest.resolve()),
        "--canonical-seed",
        str(args.seed),
        "--num-cycles",
        "1" if args.smoke else str(manifest["num_cycles"]),
        "--max-steps-per-interaction",
        str(manifest["max_steps_per_interaction"]),
        "--num-test-tasks",
        "1" if args.smoke else str(manifest["num_test_tasks"]),
        "--practice-reset-policy",
        "never",
        "--goal-pursuit-horizon",
        "0",
        "--no-ees-reset-gate",
        "--human-reset-practice-cost",
        str(arm["human_reset_practice_cost"]),
        "--record-episode-traces",
        "--record-sampler-draws",
        "--defer-rendering",
    ]
    if not arm["human_reset"]:
        flags.append("--no-human-reset")
    if method == "ees":
        flags += [
            "--reproduce-predicators-explore-target-only",
            "--reproduce-predicators-seen-task-order",
            "--reproduce-predicators-skip-perfect",
            "--reproduce-predicators-random-when-stranded",
        ]
    runs = SweepRunner.plan(
        env=environment,
        methods=[method],
        seeds=[args.seed],
        results_root=args.output,
        shared_args=flags,
        method_args={},
    )
    if any((run.output_dir / "timing.json").exists() for run in runs):
        raise RuntimeError("Refusing to overwrite a previously attempted run")
    results = SweepRunner.execute(
        runs=runs,
        max_workers=1,
        sweep_id=SweepRunner.new_sweep_id(),
        retry_policy=SpawnRetryPolicy(max_attempts=1),
    )
    return 0 if all(r.succeeded for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
