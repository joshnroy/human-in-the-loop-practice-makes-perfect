"""Freeze paired kitchen pilots; planning never submits or starts experiments.

Use `check`, `manifest --out FILE --results-root DIR`, then `run --manifest FILE --go`.
Run through scripts/with_env.sh in the frozen checkout. See docs/experiments/kitchen-pilots.md.

`della --tag TAG --out DIR` writes the same manifest frozen for the Della checkout
`<DELLA>/code/<HEAD sha>` plus `array.sbatch`, whose task N runs manifest row N; submit
with `sbatch DIR/array.sbatch <manifest on Della>`. Nothing is copied or submitted here.
Each Della run evaluates a checkpoint's test tasks in parallel (`--evaluation-workers`).
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.step_protocol import StepProtocolValidation
from scripts.run_sweep import SweepRun, SweepRunner

REPO = Path(__file__).resolve().parents[1]
COSTS_PATH = "configs/kitchen_pilot_costs.json"
COSTS = REPO / COSTS_PATH
# stage -> (practice ticks, measurement interval, test tasks). `dry` sizes a Della launch:
# one practice block between two full-length evaluations of one task per requested count.
# `long` is the overnight protocol at 8x the practice: 25,600 ticks is about 5 practice
# cycles, too few to learn the Pour's ramp rate.
STAGES = {
    "pilot": (6400, 6400, 3),
    "overnight": (25600, 6400, 9),
    "long": (200_000, 25_000, 9),
    "long_learning_early": (200_000, 25_000, 9),
    # `long`, deployed as one attempt: a test task ends at its first failed Pour.
    "long_one_shot": (200_000, 25_000, 9),
    "dry": (1600, 1600, 3),
}
# Learning updates measured right after, beside the interval multiples.
# `long_learning_early` is `long` plus measurements after updates 1, 2 and 4, so learning
# shows within the first hours; `long`'s argv is left exactly as it was. A fixed early
# tick does not do this: the samplers are fitted only at a learning update, and the
# first came at ~11.1k ticks (Della job 16931554), so 5k and 10k measured unfitted ones.
LEARNING_UPDATE_MEASUREMENTS = {"long_learning_early": (1, 2, 4)}
# Stages whose test tasks allow one Pour: at deployment the pan is hot, so a bad pour is
# committed. Every other stage's argv is unchanged.
ONE_SHOT_STAGES = frozenset({"long_one_shot"})
ARMS = ("ees", "curtis", "expectimax")
SEEDS = (0, 2)
DELLA = "/scratch/gpfs/TSILVER/jr2860"
# A Della job evaluates every test task of a checkpoint at once (`--evaluation-workers`):
# a checkpoint then takes its slowest task, not the sum of them. Local runs (`manifest`)
# keep one task at a time unless asked, so six runs fit agni's shared agents.slice.
DELLA_EVALUATION_WORKERS = {stage: tasks for stage, (_, _, tasks) in STAGES.items()}
# One task's ceiling: the dry run's slowest, 72 min at the 7,500-step limit, about 3x over.
EVALUATION_TASK_TIMEOUT_SECONDS = 4 * 3600
# Resident memory, GiB: the practice process, and one evaluation process (one task).
# Evaluations never overlap (`FreshProcessEvaluations` runs one at a time), so a job
# holds at most practice plus one evaluation's workers; overnight's 27G peaked at 17.9 GB.
DELLA_PRACTICE_GB = 3.0
# Practice keeps 8x the data over `long`'s 200k ticks; its growth is unmeasured.
DELLA_PRACTICE_GB_BY_STAGE = {"long": 12.0, "long_learning_early": 12.0, "long_one_shot": 12.0}
DELLA_EVALUATION_PROCESS_GB = 2.0
DELLA_MEMORY_HEADROOM = 1.25
# Practice runs alongside the evaluations: 4-7 h for the overnight budget, then the last
# checkpoint's evaluation, its slowest task (72 min at the limit), with headroom.
# Della's Slurm feature for its AMD Genoa CPU nodes: the launches are sized on them.
DELLA_CONSTRAINT = "genoa"
# `long` is sized on the slowest arm, expectimax (2026-11-04 job 16876353: 14,800 ticks
# in 6,978 s, ~2.1 ticks/s): ~26 h for 200k ticks plus the final ~1 h evaluation. EES
# (25,600 in 1.25-2 h) needs 10-16 h. 48 h is ~1.75x that projection, in the same QOS band as 36 h.
DELLA_TIME = {
    "pilot": "06:00:00",
    "overnight": "12:00:00",
    "long": "48:00:00",
    "long_learning_early": "48:00:00",
    "long_one_shot": "48:00:00",
    "dry": "06:00:00",
}
DEPENDENCIES = {
    "kinder": "reference/kindergarden",
    "kinder_models": "reference/kinder-baselines",
}


class KitchenLaunch:
    """One fixed comparison, shared by manifest validation and execution."""

    @staticmethod
    def flags(
        *, arm: str, stage: str, seed: int, repo: Path = REPO, evaluation_workers: int = 1
    ) -> list[str]:
        updates = LEARNING_UPDATE_MEASUREMENTS.get(stage, ())
        return [
            *KitchenLaunch.arm_flags(
                arm=arm, stage=stage, seed=seed, repo=repo, evaluation_workers=evaluation_workers
            ),
            *(["--measure-after-learning-updates", *map(str, updates)] if updates else []),
        ]

    @staticmethod
    def arm_flags(
        *, arm: str, stage: str, seed: int, repo: Path, evaluation_workers: int
    ) -> list[str]:
        budget, interval, tasks = STAGES[stage]
        if arm not in ARMS:
            raise ValueError("Only EES, Curtis and expectimax are admitted; always-ask is pending")
        common = [
            "--pour-payload", "potsticker",
            "--payload-count", "3", "--requested-counts", "1", "2", "3",
            "--receiving-vessel", "skillet",
            "--pour-layout", "stations",
            "--surface-mu", "0.1", "--canonical-seed", "0",
            "--pour-ramp-rate-bounds", "0.25", "5.0",
            "--test-env-seed-offset", "10000",
            "--practice-reset-policy", "never",
            "--max-steps-per-interaction", "20",
            "--practice-step-budget", str(budget),
            "--measurement-interval-steps", str(interval), "--initial-measurement",
            "--num-test-tasks", str(tasks),
            "--evaluation-control-steps", "7500",
            "--skill-step-limit", "2500", "--stop-after-perfect-evaluations", "0",
            "--practice-cost-config", str(repo / COSTS_PATH),
            "--defer-rendering", "--record-motion", "all" if seed == 0 else "evaluation",
            "--exploration-epsilon", "0.5", "--sampler-max-train-iters", "10000",
            "--record-sampler-draws", "--no-ees-reset-gate", "--goal-pursuit-horizon", "0",
            "--goal-pursuit-interval", "1",
            "--competence-recency-size", "5", "--competence-window-size", "5",
            "--planning-timeout", "10.0",
            "--no-reproduce-predicators-double-observe",
            "--reproduce-predicators-explore-target-only",
            "--reproduce-predicators-practice-target-history",
            "--reproduce-predicators-random-when-stranded",
            "--reproduce-predicators-seen-task-order",
            "--no-reproduce-predicators-skip-perfect",
        ]  # fmt: skip
        if stage in ONE_SHOT_STAGES:
            common += ["--evaluation-max-pour-attempts", "1"]
        if evaluation_workers > 1:
            common += [
                "--evaluation-workers", str(evaluation_workers),
                "--evaluation-task-rng", "per-task",
                "--evaluation-task-timeout-seconds", str(EVALUATION_TASK_TIMEOUT_SECONDS),
            ]  # fmt: skip
        if arm == "ees":
            return [
                *common,
                "--ees-score-human-skills",
                "--practice-groundings", "observed",
                "--ees-preview-test-goals",
                "--record-ees-decisions",
            ]  # fmt: skip
        planner = [
            *common, "--pomdp-competence-model", "local_trend",
            "--pomdp-linear-cost-lambda", "0.0003",
            "--pomdp-inference-engine", "grid", "--pomdp-num-particles", "1024",
            "--pomdp-observation-probability-weight", "0.01" if arm == "curtis" else "0",
            "--pomdp-solver", "determinized_astar" if arm == "curtis" else "expectimax",
        ]  # fmt: skip
        if arm == "curtis":
            planner += [
                "--pomdp-determinized-variant", "curtis", "--pomdp-num-rollouts", "all",
                "--pomdp-draw-scope", "node", "--pomdp-max-search-iterations", "100",
            ]  # fmt: skip
        else:
            # Expectimax only: Curtis is bounded by its iterations, never a depth.
            planner += ["--pomdp-search-depth", "6"]
        return planner

    @staticmethod
    def runs(
        *,
        stage: str,
        results_root: Path,
        seeds: tuple[int, ...] = SEEDS,
        repo: Path = REPO,
        evaluation_workers: int = 1,
        human_reset_success_probabilities: tuple[float, ...] | None = None,
    ) -> list[SweepRun]:
        """Every arm x seed. Given `human_reset_success_probabilities`, every p x arm x
        seed instead: each run adds `--human-reset-success-probability p` and writes
        under `human-reset-p<p>/`; without it the runs are exactly the stage's own."""
        sweep: tuple[float | None, ...] = human_reset_success_probabilities or (None,)
        return [
            run
            for p in sweep
            for arm in ARMS
            for seed in seeds
            for run in SweepRunner.plan(
                env="pouring3d_kitchen",
                methods=["ees" if arm == "ees" else "pomdp"],
                seeds=[seed],
                results_root=(results_root if p is None else results_root / f"human-reset-p{p}")
                / arm,
                shared_args=[
                    *KitchenLaunch.flags(
                        arm=arm,
                        stage=stage,
                        seed=seed,
                        repo=repo,
                        evaluation_workers=evaluation_workers,
                    ),
                    *([] if p is None else ["--human-reset-success-probability", str(p)]),
                ],
                method_args={},
            )
        ]

    @staticmethod
    def rows(*, runs: list[SweepRun], validate: bool = True) -> list[dict]:
        rows = []
        for run in runs:
            args = Cli.parse_args(argv=run.command[3:])
            if validate:
                StepProtocolValidation.check(args=args)
            rows.append({
                "arm": run.output_dir.parts[-3],
                "method": run.method,
                "seed": run.seed,
                "output_dir": str(run.output_dir),
                "argv": run.command[3:],
                "resolved_args": json.loads(json.dumps(vars(args), default=str)),
            })
        return rows

    @staticmethod
    def git(*, repo: Path = REPO, args: list[str]) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    @staticmethod
    def identity(*, imports: bool = False) -> dict:
        # Untracked documentation does not change execution, but untracked Python can.
        for root, paths in (
            (REPO, ["src", "scripts", "configs"]),
            *((REPO / rel, []) for rel in DEPENDENCIES.values()),
        ):
            subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD"], check=True)
            unknown = KitchenLaunch.git(
                repo=root, args=["ls-files", "--others", "--exclude-standard", "--", *paths]
            )
            if unknown:
                raise RuntimeError(f"Uncommitted runtime files in {root}: {unknown}")
        dependencies = {}
        for package, relative in DEPENDENCIES.items():
            expected = KitchenLaunch.git(args=["ls-tree", "HEAD", relative]).split()[2]
            actual = KitchenLaunch.git(repo=REPO / relative, args=["rev-parse", "HEAD"])
            if actual != expected:
                raise RuntimeError(f"{package} revision {actual} != pinned {expected}")
            dependencies[relative] = actual
        if imports:
            for package, relative in {"hitl_pmp": "src", **DEPENDENCIES}.items():
                source = importlib.import_module(package).__file__
                if source is None or not Path(source).resolve().is_relative_to(REPO / relative):
                    raise RuntimeError(f"{package} imports outside this checkout: {source}")
        return {"commit": KitchenLaunch.git(args=["rev-parse", "HEAD"]), **dependencies}

    @staticmethod
    def write_manifest(
        *,
        path: Path,
        stage: str,
        results_root: Path,
        seeds: tuple[int, ...] = SEEDS,
        repo: Path = REPO,
        evaluation_workers: int = 1,
        human_reset_success_probabilities: tuple[float, ...] | None = None,
    ) -> None:
        identity = KitchenLaunch.identity()
        root = results_root.resolve()
        runs = KitchenLaunch.runs(
            stage=stage,
            results_root=root,
            seeds=seeds,
            repo=repo,
            evaluation_workers=evaluation_workers,
            human_reset_success_probabilities=human_reset_success_probabilities,
        )
        data = {
            "status": "planned",
            "identity": identity,
            "stage": stage,
            "seeds": list(seeds),
            "evaluation_workers": evaluation_workers,
            **(
                {"human_reset_success_probabilities": list(human_reset_success_probabilities)}
                if human_reset_success_probabilities
                else {}
            ),
            "results_root": str(root),
            "working_directory": str(repo),
            "cost_sha256": hashlib.sha256(COSTS.read_bytes()).hexdigest(),
            "cost_config": json.loads(COSTS.read_text()),
            "runs": KitchenLaunch.rows(runs=runs),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")

    @staticmethod
    def della_memory(*, evaluation_workers: int, stage: str | None = None) -> str:
        """Slurm `--mem` for practice plus `evaluation_workers` evaluation processes."""
        practice = DELLA_PRACTICE_GB_BY_STAGE.get(stage or "", DELLA_PRACTICE_GB)
        gigabytes = DELLA_MEMORY_HEADROOM * (
            practice + evaluation_workers * DELLA_EVALUATION_PROCESS_GB
        )
        return f"{math.ceil(gigabytes)}G"

    @staticmethod
    def della(
        *,
        out: Path,
        tag: str,
        stage: str,
        seeds: tuple[int, ...],
        time: str | None = None,
        mem: str | None = None,
        evaluation_workers: int | None = None,
        constraint: str = DELLA_CONSTRAINT,
        human_reset_success_probabilities: tuple[float, ...] | None = None,
    ) -> Path:
        """Freeze the committed HEAD for `<DELLA>/code/<sha>`; Della results stay on Della.

        One CPU for practice and one per evaluation worker; memory and time sized from
        them unless given."""
        workers = (
            DELLA_EVALUATION_WORKERS[stage] if evaluation_workers is None else evaluation_workers
        )
        time = DELLA_TIME[stage] if time is None else time
        mem = (
            KitchenLaunch.della_memory(evaluation_workers=workers, stage=stage)
            if mem is None
            else mem
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
            raise ValueError("--tag must contain only letters, digits, underscores and hyphens")
        sha = KitchenLaunch.identity()["commit"]
        code = Path(f"{DELLA}/code/{sha}")
        KitchenLaunch.write_manifest(
            path=out / "manifest.json",
            stage=stage,
            results_root=Path(f"{DELLA}/results/{tag}"),
            seeds=seeds,
            repo=code,
            evaluation_workers=workers,
            human_reset_success_probabilities=human_reset_success_probabilities,
        )
        rows = len(ARMS) * len(seeds) * len(human_reset_success_probabilities or (None,))
        sbatch = out / "array.sbatch"
        node = f"#SBATCH --constraint={constraint}\n" if constraint else ""
        sbatch.write_text(f"""#!/usr/bin/env bash
#SBATCH --partition=cpu
{node}#SBATCH --cpus-per-task={workers + 1}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=1-{rows}
#SBATCH --job-name={tag}
#SBATCH --output={DELLA}/slurm-logs/%x-%A_%a.out
set -euo pipefail
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV={DELLA}/envs/hitl-pmp
export FD_EXEC_PATH={DELLA}/downward
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export HITL_PMP_TMPDIR={DELLA}/tmp/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$HITL_PMP_TMPDIR"
export XDG_RUNTIME_DIR="$HITL_PMP_TMPDIR/xdg"
mkdir -m 700 -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
cd {code}
exec {code}/scripts/with_env.sh python -m scripts.kitchen_launch run \\
  --manifest "$1" --task "$SLURM_ARRAY_TASK_ID" --go
""")
        return sbatch

    @staticmethod
    def native_check() -> dict:
        from hitl_pmp.environments.pouring3d.kitchen.cli import KitchenCli

        args = Cli.parse_args(
            argv=KitchenLaunch.runs(stage="pilot", results_root=Path("/unused"))[0].command[3:]
        )
        problem = KitchenCli.build_problem(args=args)
        try:
            problem.hard_reset()
            controller = problem.env.controller()
            if problem.env.payload_kind != "potsticker" or controller.mode != "joint_lateral":
                raise RuntimeError("Pilot requires joint-feedback dumpling controller")
            provider = KitchenCli.context(args=args, problem=problem).skill_provider
            skills = sorted(skill.name for skill in provider.skills())
            pours = [name for name in skills if name.startswith("Pour")]
            if pours != ["Pour1", "Pour2", "Pour3"]:
                raise RuntimeError(f"Unexpected learned skills: {pours}")
            if {s.param_dim for s in provider.skills() if s.name in pours} != {1}:
                raise RuntimeError("Each kitchen Pour learns exactly its ramp rate")
            locations = controller.scene.cube_locations()
            if locations != ["pitcher"] * 3 or not controller.ready():
                raise RuntimeError(f"Native initial scene is not ready: {locations}")
            return {"skills": skills, "initial_locations": locations, "ready": True}
        finally:
            problem.env.close()

    @staticmethod
    def run(*, manifest: Path, max_workers: int, task: int | None = None) -> int:
        data = json.loads(manifest.read_text())
        if KitchenLaunch.identity(imports=True) != data["identity"]:
            raise RuntimeError("Launch identity differs from the frozen manifest")
        if hashlib.sha256(COSTS.read_bytes()).hexdigest() != data["cost_sha256"]:
            raise RuntimeError("Launch cost file differs from the frozen manifest")
        runs = KitchenLaunch.runs(
            stage=data["stage"],
            results_root=Path(data["results_root"]),
            seeds=tuple(data.get("seeds", SEEDS)),
            evaluation_workers=data.get("evaluation_workers", 1),
            human_reset_success_probabilities=tuple(
                data.get("human_reset_success_probabilities", ())
            )
            or None,
        )
        # Array siblings may already be running, so their outputs exist; only the
        # rows this worker will execute must still be fresh (validated below).
        if KitchenLaunch.rows(runs=runs, validate=False) != data["runs"]:
            raise RuntimeError("Launch commands or resolved defaults differ from the manifest")
        rows = data["runs"]
        if task is not None:
            if not 1 <= task <= len(runs):
                raise ValueError(f"Task index must be between 1 and {len(runs)}")
            runs, rows = [runs[task - 1]], [rows[task - 1]]
        KitchenLaunch.rows(runs=runs)
        for run in runs:
            if run.output_dir.exists():
                raise FileExistsError(f"Refusing to overwrite prior run: {run.output_dir}")
        physics = KitchenLaunch.native_check()
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for run, row in zip(runs, rows, strict=True):
            run.output_dir.mkdir(parents=True)
            (run.output_dir / "launch.json").write_text(
                json.dumps(
                    {
                        "manifest_sha256": digest,
                        "identity": data["identity"],
                        "cost_sha256": data["cost_sha256"],
                        "cost_config": data["cost_config"],
                        "verified_initial_scene": physics,
                        "run": row,
                    },
                    indent=2,
                )
                + "\n"
            )
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        outcomes = SweepRunner.execute(runs=runs, max_workers=max_workers)
        for outcome in outcomes:
            print(f"{outcome.run.output_dir}: exit {outcome.returncode}", flush=True)
        return int(any(not outcome.succeeded for outcome in outcomes))

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mode", choices=("check", "manifest", "della", "run"))
        parser.add_argument("--stage", choices=tuple(STAGES), default="pilot")
        parser.add_argument("--out", type=Path)
        parser.add_argument("--results-root", type=Path)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--task", type=int, help="Run one manifest row (1-based array index)")
        parser.add_argument("--native", action="store_true")
        parser.add_argument("--max-workers", type=int, choices=(1, 2, 3, 4, 5, 6), default=1)
        parser.add_argument("--go", action="store_true")
        parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
        parser.add_argument("--tag", help="della: job name and results directory")
        parser.add_argument("--time", help="della: Slurm --time (default: sized per stage)")
        parser.add_argument("--mem", help="della: Slurm --mem (default: sized per worker)")
        parser.add_argument(
            "--constraint",
            default=DELLA_CONSTRAINT,
            help=f"della: Slurm --constraint (default: {DELLA_CONSTRAINT}; '' for none)",
        )
        parser.add_argument(
            "--evaluation-workers",
            type=int,
            help="Test tasks evaluated at once per run (default: 1 for manifest, every "
            "task of the stage for della); more than 1 evaluates with per-task RNGs",
        )
        parser.add_argument(
            "--human-reset-success-probabilities",
            type=float,
            nargs="+",
            help="Sweep the human pour-back's true competence: every p x arm x seed, each "
            "under human-reset-p<p>/ (default: no sweep; the stage's runs unchanged)",
        )
        args = parser.parse_args()
        sweep = (
            tuple(args.human_reset_success_probabilities)
            if args.human_reset_success_probabilities
            else None
        )
        if args.mode == "check":
            rows = KitchenLaunch.rows(
                runs=KitchenLaunch.runs(
                    stage=args.stage,
                    results_root=Path("/unused"),
                    seeds=tuple(args.seeds),
                    human_reset_success_probabilities=sweep,
                )
            )
            for row in rows:
                print(f"ok {row['arm']} seed {row['seed']}: {STAGES[args.stage][0]} ticks")
            if args.native:
                print(json.dumps(KitchenLaunch.native_check(), indent=2))
            return 0
        if args.mode == "manifest":
            if args.out is None or args.results_root is None:
                parser.error("manifest needs --out and --results-root")
            KitchenLaunch.write_manifest(
                path=args.out,
                stage=args.stage,
                results_root=args.results_root,
                evaluation_workers=args.evaluation_workers or 1,
                human_reset_success_probabilities=sweep,
            )
            print(f"Planned the runs in {args.out}; nothing submitted")
            return 0
        if args.mode == "della":
            if args.out is None or args.tag is None:
                parser.error("della needs --out DIR and --tag TAG")
            sbatch = KitchenLaunch.della(
                out=args.out,
                tag=args.tag,
                stage=args.stage,
                seeds=tuple(args.seeds),
                time=args.time,
                mem=args.mem,
                evaluation_workers=args.evaluation_workers,
                constraint=args.constraint,
                human_reset_success_probabilities=sweep,
            )
            print(f"Planned; nothing copied or submitted. sbatch {sbatch} <Della manifest>")
            return 0
        if args.manifest is None or not args.go:
            parser.error("run needs --manifest and --go")
        return KitchenLaunch.run(
            manifest=args.manifest, max_workers=args.max_workers, task=args.task
        )


if __name__ == "__main__":
    raise SystemExit(KitchenLaunch.main())
