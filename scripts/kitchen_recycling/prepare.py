"""Write a kitchen-only launch proposal. Never contact Slurm or execute experiments."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path

ARMS = ("ees", "expectimax", "curtis")
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


def argv(*, arm: str, seed: int, cost_file: str, kitchen_config: str) -> list[str]:
    result = [
        *COMMON,
        "--seed",
        str(seed),
        "--practice-cost-config",
        cost_file,
        "--kitchen-config",
        kitchen_config,
    ]
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
    parser.add_argument("--max-parallel", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--cost-file", default="scripts/kitchen_recycling/costs.json")
    parser.add_argument(
        "--kitchen-config", default="scripts/kitchen_recycling/experiment-config.json"
    )
    args = parser.parse_args()
    if not args.tag.startswith("kitchen-recycling-") or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in args.tag
    ):
        parser.error("tag must be a new kitchen-recycling-* identifier with no path separators")
    args.out.mkdir(parents=True, exist_ok=False)
    root = f"/scratch/gpfs/TSILVER/jr2860/kitchen-recycling/{args.tag}"
    rows = []
    for seed in range(3):
        for arm in ARMS:
            command = argv(
                arm=arm, seed=seed, cost_file=args.cost_file, kitchen_config=args.kitchen_config
            )
            rows.append({"index": len(rows), "arm": arm, "seed": seed, "argv": command})
    manifest = {
        "schema": 1,
        "tag": args.tag,
        "remote_root": root,
        "max_parallel": args.max_parallel,
        "budget_policy": "160k counted steps; cost logged, not a hard cap",
        "runs": rows,
        "resources": {
            "account": "tsilver",
            "partition": "cpu",
            "cpus": 2,
            "memory": "16G",
            "time": "48:00:00",
            "gpus": 0,
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
                f"{root}/results/{row['arm']}/{row['seed']}",
            ])
            for row in rows
        )
        + "\n"
    )
    for phase in ("smoke", "full"):
        array = f"#SBATCH --array=0-8%{args.max_parallel}\n" if phase == "full" else ""
        script = (
            "#!/usr/bin/env bash\n"
            "#SBATCH --account=tsilver\n#SBATCH --partition=cpu\n"
            "#SBATCH --cpus-per-task=2\n#SBATCH --mem=16G\n"
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
