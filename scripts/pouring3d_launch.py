"""Freeze paired pouring pilots; planning never submits or starts experiments.

Use `check`, `manifest --out FILE --results-root DIR`, then `run --manifest FILE --go`.
Run through scripts/with_env.sh in the frozen checkout. See docs/experiments/pouring3d-pilots.md.
"""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
from pathlib import Path

from hitl_pmp.cli import Cli
from hitl_pmp.step_protocol import StepProtocolValidation
from scripts.run_sweep import SweepRun, SweepRunner

REPO = Path(__file__).resolve().parents[1]
COSTS = REPO / "configs/pouring3d_pilot_costs.json"
STAGES = {"pilot": 6400, "long-pilot": 25600}
ARMS = ("ees", "curtis", "expectimax")
SEEDS = (0, 2)
DEPENDENCIES = {
    "kinder": "reference/kindergarden",
    "kinder_models": "reference/kinder-baselines",
}


class PouringLaunch:
    """One fixed comparison, shared by manifest validation and execution."""

    @staticmethod
    def flags(*, arm: str, stage: str, seed: int) -> list[str]:
        if arm not in ARMS:
            raise ValueError("Only EES, Curtis and expectimax are admitted; always-ask is pending")
        common = [
            "--payload-count", "3", "--requested-counts", "1", "2", "3",
            "--surface-mu", "0.1", "--canonical-seed", "0",
            "--pour-tilt-bounds", "87", "95",
            "--pour-amplitude-bounds", "0", "0.025",
            "--pour-duration-bounds", "0", "30",
            "--test-env-seed-offset", "10000",
            "--practice-reset-policy", "never",
            "--max-steps-per-interaction", "20",
            "--practice-step-budget", str(STAGES[stage]),
            "--measurement-interval-steps", "3200", "--initial-measurement",
            "--num-test-tasks", "9", "--evaluation-control-steps", "2500",
            "--skill-step-limit", "2500", "--stop-after-perfect-evaluations", "0",
            "--practice-cost-config", str(COSTS), "--human-reset-practice-cost", "5",
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
        if arm == "ees":
            return [
                *common,
                "--ees-score-human-skills",
            ]  # fmt: skip
        planner = [
            *common, "--pomdp-competence-model", "local_trend",
            "--pomdp-linear-cost-lambda", "0.0003", "--pomdp-search-depth", "6",
            "--pomdp-inference-engine", "grid", "--pomdp-num-particles", "1024",
            "--pomdp-observation-probability-weight", "0.01" if arm == "curtis" else "0",
            "--pomdp-solver", "determinized_astar" if arm == "curtis" else "expectimax",
        ]  # fmt: skip
        if arm == "curtis":
            planner += [
                "--pomdp-determinized-variant", "curtis", "--pomdp-num-rollouts", "all",
                "--pomdp-draw-scope", "node", "--pomdp-max-search-iterations", "100",
            ]  # fmt: skip
        return planner

    @staticmethod
    def runs(*, stage: str, results_root: Path) -> list[SweepRun]:
        return [
            run
            for arm in ARMS
            for seed in SEEDS
            for run in SweepRunner.plan(
                env="pouring3d",
                methods=["ees" if arm == "ees" else "pomdp"],
                seeds=[seed],
                results_root=results_root / arm,
                shared_args=PouringLaunch.flags(arm=arm, stage=stage, seed=seed),
                method_args={},
            )
        ]

    @staticmethod
    def rows(*, runs: list[SweepRun]) -> list[dict]:
        rows = []
        for run in runs:
            args = Cli.parse_args(argv=run.command[3:])
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
            unknown = PouringLaunch.git(
                repo=root, args=["ls-files", "--others", "--exclude-standard", "--", *paths]
            )
            if unknown:
                raise RuntimeError(f"Uncommitted runtime files in {root}: {unknown}")
        dependencies = {}
        for package, relative in DEPENDENCIES.items():
            expected = PouringLaunch.git(args=["ls-tree", "HEAD", relative]).split()[2]
            actual = PouringLaunch.git(repo=REPO / relative, args=["rev-parse", "HEAD"])
            if actual != expected:
                raise RuntimeError(f"{package} revision {actual} != pinned {expected}")
            dependencies[relative] = actual
        if imports:
            for package, relative in {"hitl_pmp": "src", **DEPENDENCIES}.items():
                source = importlib.import_module(package).__file__
                if source is None or not Path(source).resolve().is_relative_to(REPO / relative):
                    raise RuntimeError(f"{package} imports outside this checkout: {source}")
        return {"commit": PouringLaunch.git(args=["rev-parse", "HEAD"]), **dependencies}

    @staticmethod
    def write_manifest(*, path: Path, stage: str, results_root: Path) -> None:
        identity = PouringLaunch.identity()
        root = results_root.resolve()
        rows = PouringLaunch.rows(runs=PouringLaunch.runs(stage=stage, results_root=root))
        data = {
            "status": "planned",
            "identity": identity,
            "stage": stage,
            "results_root": str(root),
            "working_directory": str(REPO),
            "cost_sha256": hashlib.sha256(COSTS.read_bytes()).hexdigest(),
            "cost_config": json.loads(COSTS.read_text()),
            "runs": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(json.dumps(data, indent=2) + "\n")

    @staticmethod
    def native_check() -> dict:
        from hitl_pmp.environments.pouring3d.practice.cli import FrozenPourCli

        args = Cli.parse_args(
            argv=PouringLaunch.runs(stage="pilot", results_root=Path("/unused"))[0].command[3:]
        )
        problem = FrozenPourCli.build_problem(args=args)
        try:
            problem.hard_reset()
            controller = problem.env.controller()
            provider = FrozenPourCli.context(args=args, problem=problem).skill_provider
            skills = sorted(skill.name for skill in provider.skills())
            pours = [name for name in skills if name.startswith("Pour")]
            if pours != ["Pour1", "Pour2", "Pour3"]:
                raise RuntimeError(f"Unexpected learned skills: {pours}")
            locations = controller.scene.cube_locations()
            if locations != ["pitcher"] * 3 or not controller.ready():
                raise RuntimeError(f"Native initial scene is not ready: {locations}")
            return {"skills": skills, "initial_locations": locations, "ready": True}
        finally:
            problem.env.close()

    @staticmethod
    def run(*, manifest: Path, max_workers: int, task: int | None = None) -> int:
        data = json.loads(manifest.read_text())
        if PouringLaunch.identity(imports=True) != data["identity"]:
            raise RuntimeError("Launch identity differs from the frozen manifest")
        if hashlib.sha256(COSTS.read_bytes()).hexdigest() != data["cost_sha256"]:
            raise RuntimeError("Launch cost file differs from the frozen manifest")
        runs = PouringLaunch.runs(stage=data["stage"], results_root=Path(data["results_root"]))
        if PouringLaunch.rows(runs=runs) != data["runs"]:
            raise RuntimeError("Launch commands or resolved defaults differ from the manifest")
        rows = data["runs"]
        if task is not None:
            if not 1 <= task <= len(runs):
                raise ValueError("Task index must be between 1 and 6")
            runs, rows = [runs[task - 1]], [rows[task - 1]]
        for run in runs:
            if run.output_dir.exists():
                raise FileExistsError(f"Refusing to overwrite prior run: {run.output_dir}")
        physics = PouringLaunch.native_check()
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
        parser.add_argument("mode", choices=("check", "manifest", "run"))
        parser.add_argument("--stage", choices=tuple(STAGES), default="pilot")
        parser.add_argument("--out", type=Path)
        parser.add_argument("--results-root", type=Path)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--task", type=int, help="Run one manifest row (1-based array index)")
        parser.add_argument("--native", action="store_true")
        parser.add_argument("--max-workers", type=int, choices=(1, 2, 3), default=1)
        parser.add_argument("--go", action="store_true")
        args = parser.parse_args()
        if args.mode == "check":
            rows = PouringLaunch.rows(
                runs=PouringLaunch.runs(stage=args.stage, results_root=Path("/unused"))
            )
            for row in rows:
                print(f"ok {row['arm']} seed {row['seed']}: {STAGES[args.stage]} ticks")
            if args.native:
                print(json.dumps(PouringLaunch.native_check(), indent=2))
            return 0
        if args.mode == "manifest":
            if args.out is None or args.results_root is None:
                parser.error("manifest needs --out and --results-root")
            PouringLaunch.write_manifest(
                path=args.out, stage=args.stage, results_root=args.results_root
            )
            print(f"Planned six runs in {args.out}; nothing submitted")
            return 0
        if args.manifest is None or not args.go:
            parser.error("run needs --manifest and --go")
        return PouringLaunch.run(
            manifest=args.manifest, max_workers=args.max_workers, task=args.task
        )


if __name__ == "__main__":
    raise SystemExit(PouringLaunch.main())
