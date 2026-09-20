"""Run the four competence model/inference combinations under one fixed protocol."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

MODELS = ("global_curve", "local_trend")
ENGINES = ("particle", "grid")


def experiment_commands(
    *, results_root: Path, num_seeds: int, num_cycles: int, python: str
) -> list[tuple[str, list[str]]]:
    """Pair task seeds and hold simulator, planner, costs and budgets fixed."""
    runs = []
    for model in MODELS:
        for engine in ENGINES:
            for seed in range(num_seeds):
                name = f"{model}-{engine}/seed_{seed:02d}"
                command = [
                    python,
                    "-m",
                    "hitl_pmp.cli",
                    "--env",
                    "tossing3d",
                    "--method",
                    "pomdp",
                    "--seed",
                    str(seed),
                    "--num-cycles",
                    str(num_cycles),
                    "--num-test-tasks",
                    "10",
                    "--max-steps-per-interaction",
                    "20",
                    "--canonical-seed",
                    "125",
                    "--layout",
                    "barrier",
                    "--evaluation-layout",
                    "barrier",
                    "--practice-reset-policy",
                    "never",
                    "--human-reset-practice-cost",
                    "5",
                    "--pomdp-linear-cost-lambda",
                    "0.0003",
                    "--pomdp-solver",
                    "determinized_astar",
                    "--pomdp-max-search-iterations",
                    "100",
                    "--pomdp-observation-probability-weight",
                    "0.001",
                    "--pomdp-num-particles",
                    "1024",
                    "--pomdp-num-samples",
                    "100",
                    "--pomdp-competence-model",
                    model,
                    "--pomdp-inference-engine",
                    engine,
                    "--pomdp-grid-competence-bins",
                    "25",
                    "--pomdp-grid-learning-rate-bins",
                    "16",
                    "--pomdp-competence-process-noise-std",
                    "0.03",
                    "--pomdp-learning-rate-process-noise-std",
                    "0.005",
                    "--pomdp-learning-rate-decay",
                    "0.9",
                    "--pomdp-learning-rate-max",
                    "0.15",
                    "--record-sampler-draws",
                    "--output-dir",
                    str(results_root / name),
                ]
                runs.append((name, command))
    return runs


def run_one(*, name: str, command: list[str], results_root: Path) -> dict[str, object]:
    log = results_root / "logs" / (name.replace("/", "-") + ".log")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": str(results_root / "matplotlib-cache"),
    }
    started = datetime.now(timezone.utc).isoformat()
    print(f"Starting {name}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, env=environment, stdout=stream, stderr=stream, check=False
        )
    record: dict[str, object] = {
        "name": name,
        "returncode": completed.returncode,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "log": str(log),
    }
    (results_root / "logs" / (name.replace("/", "-") + ".status.json")).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Finished {name}: exit {completed.returncode}", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--num-cycles", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.num_seeds, args.num_cycles, args.max_workers) < 1:
        parser.error("seeds, cycles and workers must be positive")
    root = args.results_root.resolve()
    runs = experiment_commands(
        results_root=root,
        num_seeds=args.num_seeds,
        num_cycles=args.num_cycles,
        python=sys.executable,
    )
    if args.dry_run:
        print(json.dumps(dict(runs), indent=2))
        return
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    source = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=source, text=True)
    (root / "source.patch").write_text(patch, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "source_commit": sha,
                "source_checkout": str(source),
                "python": sys.executable,
                "pythonpath": os.environ.get("PYTHONPATH", ""),
                "num_cycles": args.num_cycles,
                "num_seeds": args.num_seeds,
                "commands": dict(runs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(run_one, name=name, command=command, results_root=root)
            for name, command in runs
        ]
        results = [future.result() for future in as_completed(futures)]
    (root / "run_status.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    if any(result["returncode"] != 0 for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
