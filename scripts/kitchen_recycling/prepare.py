"""Write a kitchen-only launch proposal. Never contact Slurm or execute experiments."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = ("ees", "expectimax", "curtis")
# The full contract's budget, and one evaluation process per checkpoint. Left at these
# defaults the package is byte-identical to the one v5-v8 launched from.
PRACTICE_STEP_BUDGET = 160000
EVALUATION_WORKERS = 1
TEST_TASKS = 10
# The two human resets every row's cost file prices, and their full-contract price.
RESET_NAMES = ("ResetCubeBinSameSide", "ResetCubeBinOppositeSide")
HUMAN_RESET_COST = 5
COMMON = shlex.split(
    "--env kitchen_recycling3d --practice-step-budget 160000 "
    "--measurement-interval-steps 3200 --evaluation-control-steps 500 "
    "--num-test-tasks 10 --initial-measurement "
    "--max-steps-per-interaction 20 --practice-reset-policy never "
    "--stop-after-perfect-evaluations 0 --no-early-stop "
    "--defer-rendering --no-scene-bg --human-reset "
    "--no-ees-reset-gate --exploration-epsilon 0.5 --goal-pursuit-horizon 0 "
    "--goal-pursuit-interval 1 --goal-pursuit-init-cycles 1 "
    "--competence-recency-size 5 --competence-window-size 5 "
    "--planning-timeout 10.0 --sampler-max-train-iters 10000 "
    "--record-sampler-draws --no-reproduce-predicators-double-observe "
    "--reproduce-predicators-explore-target-only "
    "--reproduce-predicators-practice-target-history "
    "--reproduce-predicators-random-when-stranded "
    "--reproduce-predicators-seen-task-order --no-reproduce-predicators-skip-perfect"
)
PLANNER = shlex.split(
    "--pomdp-linear-cost-lambda 0.0003 "
    "--pomdp-inference-engine grid --pomdp-num-particles 1024 "
    "--practice-groundings observed "
    "--pomdp-num-rollouts all --pomdp-max-search-iterations 100 --pomdp-draw-scope node"
)


def digest(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost_value(*, cost: float) -> float | int:
    """A cost as JSON writes it: integral costs as integers, like costs.json's 5."""
    return int(cost) if float(cost).is_integer() else cost


def cost_token(*, cost: float) -> str:
    return str(cost_value(cost=cost))


def priced_costs(*, base: dict, cost: float) -> dict:
    """The base cost file with both human resets repriced, every other entry unchanged."""
    if set(base["human_skills"]) != set(RESET_NAMES):
        raise ValueError(f"Cost file must price exactly {RESET_NAMES}")
    result = json.loads(json.dumps(base))
    for name in RESET_NAMES:
        result["human_skills"][name]["cost"]["value"] = cost_value(cost=cost)
    return result


def argv(
    *,
    arm: str,
    seed: int,
    cost_file: str,
    kitchen_config: str,
    practice_step_budget: int = PRACTICE_STEP_BUDGET,
    evaluation_workers: int = EVALUATION_WORKERS,
    human_reset_cost: float | None = None,
) -> list[str]:
    common = list(COMMON)
    common[common.index("--practice-step-budget") + 1] = str(practice_step_budget)
    if evaluation_workers > 1:
        # Tasks run apart reproduce one process running them only under per-task RNG.
        common += [
            "--evaluation-workers",
            str(evaluation_workers),
            "--evaluation-task-rng",
            "per-task",
        ]
    result = [
        *common,
        "--seed",
        str(seed),
        "--practice-cost-config",
        cost_file,
        "--kitchen-config",
        kitchen_config,
    ]
    if human_reset_cost is not None:
        # The planner's own price for both resets, besides the charge the cost file
        # makes: every arm then sees the swept cost however it reads it.
        result += ["--human-reset-practice-cost", cost_token(cost=human_reset_cost)]
    if arm == "ees":
        return result + shlex.split(
            "--method ees --practice-groundings all --ees-score-human-skills --record-ees-decisions"
        )
    return result + [
        "--method",
        "pomdp",
        *PLANNER,
        "--pomdp-solver",
        "expectimax" if arm == "expectimax" else "determinized_astar",
        "--pomdp-observation-probability-weight",
        "0" if arm == "expectimax" else "0.01",
        # Expectimax's depth; Curtis is bounded by its iterations and refuses one.
        *(
            ["--pomdp-search-depth", "6"]
            if arm == "expectimax"
            else ["--pomdp-determinized-variant", "curtis"]
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--max-parallel", type=int, default=9)
    parser.add_argument("--cost-file", default="scripts/kitchen_recycling/costs.json")
    parser.add_argument(
        "--kitchen-config", default="scripts/kitchen_recycling/experiment-config.json"
    )
    parser.add_argument("--practice-step-budget", type=int, default=PRACTICE_STEP_BUDGET)
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=EVALUATION_WORKERS,
        choices=range(1, TEST_TASKS + 1),
        help="Each checkpoint's test tasks in up to this many concurrent processes "
        "(with --evaluation-task-rng per-task); the smoke then evaluates this many tasks.",
    )
    parser.add_argument("--cpus-per-task", type=int, default=2)
    parser.add_argument("--memory", default="16G")
    parser.add_argument(
        "--human-reset-costs",
        type=float,
        nargs="+",
        default=None,
        help="Sweep the human reset cost: every arm x seed once per cost, in this order, "
        "each row with its own cost file (both resets at that cost, written beside the "
        "manifest under costs/), --human-reset-practice-cost and output directory.",
    )
    parser.add_argument("--constraint", default=None, help="Slurm --constraint for every job")
    args = parser.parse_args()
    costs = args.human_reset_costs
    if costs is not None and (
        len(set(costs)) != len(costs) or any(cost < 0 or cost != cost for cost in costs)
    ):
        parser.error("--human-reset-costs must be distinct, non-negative numbers")
    if args.practice_step_budget <= 0 or args.practice_step_budget % 1000:
        parser.error("--practice-step-budget must be a positive whole number of thousands")
    if not args.tag.startswith("kitchen-recycling-") or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in args.tag
    ):
        parser.error("tag must be a new kitchen-recycling-* identifier with no path separators")
    args.out.mkdir(parents=True, exist_ok=False)
    root = f"/scratch/gpfs/TSILVER/jr2860/kitchen-recycling/{args.tag}"
    rows = []
    cost_files = {}
    if costs is not None:
        # Read from the repository, where the CLI resolves it, wherever this runs from.
        base = json.loads((REPO / args.cost_file).read_text())
        (args.out / "costs").mkdir()
        for cost in costs:
            name = f"costs/human-reset-cost-{cost_token(cost=cost)}.json"
            (args.out / name).write_text(
                json.dumps(priced_costs(base=base, cost=cost), indent=2) + "\n"
            )
            cost_files[cost] = name
    for cost in costs if costs is not None else [None]:
        for seed in range(3):
            for arm in ARMS:
                command = argv(
                    arm=arm,
                    seed=seed,
                    cost_file=args.cost_file
                    if cost is None
                    else f"{root}/launch/{cost_files[cost]}",
                    kitchen_config=args.kitchen_config,
                    practice_step_budget=args.practice_step_budget,
                    evaluation_workers=args.evaluation_workers,
                    human_reset_cost=cost,
                )
                row = {"index": len(rows), "arm": arm, "seed": seed, "argv": command}
                if cost is not None:
                    row |= {
                        "human_reset_cost": cost_value(cost=cost),
                        "costs_sha256": digest(path=args.out / cost_files[cost]),
                        "output": f"reset-cost-{cost_token(cost=cost)}/{arm}/{seed}",
                    }
                rows.append(row)
    if not 1 <= args.max_parallel <= len(rows):
        parser.error(f"--max-parallel must be between 1 and {len(rows)}")
    manifest = {
        "schema": 1,
        "tag": args.tag,
        "remote_root": root,
        "max_parallel": args.max_parallel,
        "budget_policy": f"{args.practice_step_budget // 1000}k counted steps; "
        "cost logged, not a hard cap",
        # Recorded only when not the default, so a default manifest keeps v8's bytes;
        # worker.contract reads an absent key as the default.
        **(
            {"practice_step_budget": args.practice_step_budget}
            if args.practice_step_budget != PRACTICE_STEP_BUDGET
            else {}
        ),
        **(
            {"evaluation_workers": args.evaluation_workers}
            if args.evaluation_workers != EVALUATION_WORKERS
            else {}
        ),
        **(
            {"human_reset_costs": [cost_value(cost=cost) for cost in costs]}
            if costs is not None
            else {}
        ),
        "runs": rows,
        "resources": {
            "account": "tsilver",
            "partition": "cpu",
            "cpus": args.cpus_per_task,
            "memory": args.memory,
            "time": "48:00:00",
            "gpus": 0,
            **({"constraint": args.constraint} if args.constraint else {}),
        },
        "status": "PROPOSAL_ONLY_NOT_AUTHORIZED",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "commands.txt").write_text(
        "\n".join(
            f"{row['index']} seed={row['seed']} arm={row['arm']} "
            + shlex.join([
                "python",
                "-m",
                "hitl_pmp.cli",
                *row["argv"],
                "--output-dir",
                f"{root}/results/" + row.get("output", f"{row['arm']}/{row['seed']}"),
            ])
            for row in rows
        )
        + "\n"
    )
    for phase in ("smoke", "full"):
        array = (
            f"#SBATCH --array=0-{len(rows) - 1}%{args.max_parallel}\n" if phase == "full" else ""
        )
        constraint = f"#SBATCH --constraint={args.constraint}\n" if args.constraint else ""
        script = (
            "#!/usr/bin/env bash\n"
            "#SBATCH --account=tsilver\n#SBATCH --partition=cpu\n"
            f"#SBATCH --cpus-per-task={args.cpus_per_task}\n#SBATCH --mem={args.memory}\n"
            f"{constraint}"
            f"#SBATCH --time={'02:00:00' if phase == 'smoke' else '48:00:00'}\n"
            f"#SBATCH --job-name={args.tag}-{phase}\n{array}"
            f"#SBATCH --output={root}/logs/%x-%A_%a.out\n"
            "set -euo pipefail\n"
            f"ROOT={shlex.quote(root)}\n"
            'cd "$ROOT/source"\n'
            'export PYTHONPATH="$ROOT/runtime-overlay:$ROOT/source/src:'
            "$ROOT/source/reference/kindergarden/src:"
            '$ROOT/source/reference/kinder-baselines/kinder-models/src:$ROOT/source"\n'
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1\n"
            "export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1\n"
            "export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl\n"
            "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n"
            "export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1\n"
            'export FD_EXEC_PATH="$ROOT/source/downward"\n'
            'export LD_LIBRARY_PATH="$ROOT/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            'export LD_PRELOAD="$ROOT/env/lib/libstdc++.so.6"\n'
            'export TMPDIR="$ROOT/tmp/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-smoke}"\n'
            'mkdir -p "$TMPDIR"\nexport TEMP="$TMPDIR" TMP="$TMPDIR"\n'
            'export XDG_RUNTIME_DIR="$TMPDIR/xdg"\nmkdir -m 700 "$XDG_RUNTIME_DIR"\n'
            'exec "$ROOT/env/bin/python" scripts/kitchen_recycling/worker.py '
            f'{phase} --launch "$ROOT/launch" --task "${{SLURM_ARRAY_TASK_ID:-0}}"\n'
        )
        (args.out / f"{phase}.sbatch").write_text(script)
    print(
        json.dumps({
            "manifest": str(args.out / "manifest.json"),
            "jobs": len(rows),
            "sha256": digest(path=args.out / "manifest.json"),
            "blocking": "Final single-object config, pinned snapshot, pickup/scatter "
            "evidence, CLI pilots and main authorization pending",
        })
    )


if __name__ == "__main__":
    main()
