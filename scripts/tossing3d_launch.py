"""The final Tossing3D comparison: every arm's argv, in one place.

    python -m scripts.tossing3d_launch print  [--arms A ...] [--size dry]
    python -m scripts.tossing3d_launch check  [--arms A ...]
    python -m scripts.tossing3d_launch della  --tag TAG --out DIR [--sha SHA] [--size dry]

`print` writes each arm's `hitl_pmp.cli` argv (seed 0); `check` parses every arm with the
real CLI; `della` writes one Slurm array per time group (manifest + sbatch, the shelf
launcher's format) and submits nothing.

The plan (Josh, 2026-10-19 to 2026-10-22):
- Tossing3D-o1, barrier layout for practice and evaluation, the step protocol with
  160,000 counted practice steps and an evaluation every 3,200, 10 held-out tasks.
- Six arms: EES, ours with expectimax, ours with Curtis determinized A* at
  observation weight 0.01, and an always-ask variant of each
  (`--always-ask-reset`). Seeds 0-5: six, so a paired exact sign test can reach
  p = 0.031.
- The planner arms are `--preset tossing3d-canonical` (Model B, expectimax depth 6,
  observation weight 0, lambda 3e-4, 0.01 per control tick, human reset 5), with
  `--pomdp-solver determinized_astar` for A*.
- EES takes EXP-22c's flags (the preset is the pomdp method's only), less the
  cycle-mode ones the step protocol replaces, and prices a human reset at 5 in its own
  plan-cost units. It runs without skip_perfect, the shelf's labelled deviation (a skill
  can measure 1.0 after a few samples and be wrongly treated as mastered).
- Each arm's cost file comes from `Tossing3DArmCosts` (#532): EES vs planner prices,
  and the sampled-start reset priced for always-ask arms.
- Every arm stops after three perfect (10/10) evaluations in a row (Josh, 2026-10-22);
  otherwise it spends the whole budget.
- Video: the step protocol writes `tossing3d_state_log.jsonl` for every run, and videos
  are reconstructed from it (hence `--defer-rendering`), so no flag is needed here.

Della: code in `/scratch/gpfs/TSILVER/jr2860/code/<sha>`, results in
`/scratch/gpfs/TSILVER/jr2860/results/<tag>/<arm>/<method>/<seed>`; submit with
`sbatch <out>/array-<group>.sbatch <out>/manifest-<group>.tsv`, once per group.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from scripts.sweepoffshelf3d_launch import Budget, DellaGroup, ShelfLaunch

REPO = Path(__file__).resolve().parents[1]
ENV = "tossing3d"
ALWAYS_ASK_FLAG = "--always-ask-reset"
SEEDS = 6
DELLA = "/scratch/gpfs/TSILVER/jr2860"
DEFAULT_ARMS = (
    "ees",
    "expectimax",
    "determinized-curtis",
    "ees-always-ask",
    "expectimax-always-ask",
    "determinized-curtis-always-ask",
)

# Josh, 2026-10-22: twice the Della dry run's projection per group (EXP-22c's full runs on
# Della took at most 2.2 h for EES and 3.2 h for Model-B expectimax). A sim run that
# hits its limit cannot resume, so it keeps only the evaluations written so far.
DELLA_GROUPS = {
    "ees": DellaGroup(time="08:00:00"),
    "determinized": DellaGroup(time="16:00:00"),
    "expectimax": DellaGroup(time="48:00:00"),
}

FULL = Budget(
    practice_steps=160000, evaluation_interval=3200, evaluation_control_steps=500, tasks=10
)
DRY = Budget(practice_steps=3200, evaluation_interval=3200, evaluation_control_steps=200, tasks=1)
PILOT = Budget(
    practice_steps=6400, evaluation_interval=3200, evaluation_control_steps=500, tasks=10
)
LONG_PILOT = Budget(
    practice_steps=25600, evaluation_interval=3200, evaluation_control_steps=500, tasks=10
)
SIZES = {"full": FULL, "dry": DRY, "pilot": PILOT, "long-pilot": LONG_PILOT}

# The scene and practice settings every arm shares (the preset sets them for the planner
# arms too; repeating them here keeps the two methods on one scene).
SCENE = [
    "--bin-mass-kg", "1.0",
    "--cube-mass-kg", "0.1",
    "--layout", "barrier",
    "--evaluation-layout", "barrier",
    "--max-steps-per-interaction", "20",
    "--practice-reset-policy", "never",
    "--defer-rendering",
    "--test-env-seed-offset", "10000",
]  # fmt: skip
# EXP-22c's EES flags under the step protocol (cycle-mode-only flags dropped).
EES = [
    "--no-ees-reset-gate",
    "--exploration-epsilon", "0.5",
    "--goal-pursuit-horizon", "0",
    "--goal-pursuit-interval", "1",
    "--competence-recency-size", "5",
    "--competence-window-size", "5",
    "--planning-timeout", "10.0",
    "--sampler-max-train-iters", "10000",
    "--record-sampler-draws",
    "--no-reproduce-predicators-double-observe",
    "--reproduce-predicators-explore-target-only",
    "--reproduce-predicators-practice-target-history",
    "--reproduce-predicators-random-when-stranded",
    "--reproduce-predicators-seen-task-order",
    "--no-reproduce-predicators-skip-perfect",
]  # fmt: skip
PLANNER = ["--preset", "tossing3d-canonical", "--pomdp-linear-cost-lambda", "0.0003"]


class Tossing3DArm(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    method: str
    flags: list[str]
    always_ask: bool


class Tossing3DLaunch:
    """A static-method container, never instantiated."""

    @staticmethod
    def arms() -> dict[str, Tossing3DArm]:
        canonical = [
            Tossing3DArm(name="ees", method="ees", flags=EES, always_ask=False),
            Tossing3DArm(
                name="expectimax",
                method="pomdp",
                flags=[*PLANNER, "--pomdp-solver", "expectimax"],
                always_ask=False,
            ),
            Tossing3DArm(
                name="determinized",
                method="pomdp",
                flags=[*PLANNER, "--pomdp-solver", "determinized_astar"],
                always_ask=False,
            ),
            # Josh, 2026-10-23: determinized also as Curtis et al.'s PO_DAStar (#577):
            # every outcome its own child at -lambda log p, lambda = 0.01, the value that
            # passed the 3,200-step gate (0.1 STOPs at the first decision).
            Tossing3DArm(
                name="determinized-curtis",
                method="pomdp",
                flags=[
                    *PLANNER,
                    "--pomdp-solver",
                    "determinized_astar",
                    "--pomdp-determinized-variant",
                    "curtis",
                    "--pomdp-num-rollouts",
                    "all",
                    "--pomdp-observation-probability-weight",
                    "0.01",
                ],  # fmt: skip
                always_ask=False,
            ),
        ]
        always_ask = [
            Tossing3DArm(
                name=f"{arm.name}-always-ask",
                method=arm.method,
                flags=[*arm.flags, ALWAYS_ASK_FLAG],
                always_ask=True,
            )
            for arm in canonical
        ]
        return {arm.name: arm for arm in [*canonical, *always_ask]}

    @staticmethod
    def method_args(*, arm: Tossing3DArm, size: Budget, repo: Path = REPO) -> list[str]:
        """Every `hitl_pmp.cli` flag except `--env/--method/--seed/--output-dir`;
        `repo` is the checkout the run executes from, which holds the cost files."""
        costs = Tossing3DArmCosts.cost_config(method=arm.method, always_ask=arm.always_ask)
        costs = repo / costs.relative_to(REPO.resolve())
        return [
            *arm.flags,
            *SCENE,
            "--practice-step-budget", str(size.practice_steps),
            "--measurement-interval-steps", str(size.evaluation_interval),
            "--evaluation-control-steps", str(size.evaluation_control_steps),
            "--num-test-tasks", str(size.tasks),
            "--stop-after-perfect-evaluations", "0" if size in (PILOT, LONG_PILOT) else "3",
            "--practice-cost-config", str(costs),
        ]  # fmt: skip

    @staticmethod
    def cli_argv(
        *, arm: Tossing3DArm, size: Budget, seed: int, output_dir: Path, repo: Path = REPO
    ) -> list[str]:
        return [
            "--env", ENV,
            "--method", arm.method,
            "--seed", str(seed),
            "--output-dir", str(output_dir),
            *Tossing3DLaunch.method_args(arm=arm, size=size, repo=repo),
        ]  # fmt: skip

    @staticmethod
    def della_group(*, arm: Tossing3DArm) -> str:
        if arm.method == "ees":
            return "ees"
        return "determinized" if arm.name.startswith("determinized") else "expectimax"

    @staticmethod
    def della_manifest(
        *,
        arms: list[Tossing3DArm],
        size: Budget,
        sha: str,
        tag: str,
        seeds: tuple[int, ...] = tuple(range(SEEDS)),
    ) -> list[str]:
        """One tab-separated line per arm x seed, arm-major, in array-task order."""
        code = Path(f"{DELLA}/code/{sha}")
        lines = []
        for arm in arms:
            for seed in seeds:
                output_dir = Path(f"{DELLA}/results/{tag}/{arm.name}/{arm.method}/{seed}")
                argv = Tossing3DLaunch.cli_argv(
                    arm=arm, size=size, seed=seed, output_dir=output_dir, repo=code
                )
                fields = [f"{tag}-{arm.name}-s{seed}", sha, arm.name, str(seed)]
                lines.append("\t".join([*fields, str(output_dir), shlex.join(argv)]))
        return lines

    @staticmethod
    def parse(*, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("mode", choices=["print", "check", "della", "pilot", "worker"])
        parser.add_argument("--arms", nargs="+", help="Default: every arm of the plan.")
        parser.add_argument("--seeds", nargs="+", type=int)
        parser.add_argument("--no-bin-pouring", action="store_true")
        parser.add_argument("--results-root", type=Path)
        parser.add_argument("--manifest", type=Path)
        parser.add_argument("--task", type=int)
        parser.add_argument("--size", choices=sorted(SIZES), default="full")
        parser.add_argument("--tag", help="della: the results directory and job name.")
        parser.add_argument("--sha", help="della: the commit to run. Default: HEAD.")
        parser.add_argument("--out", type=Path, help="della: where to write the files.")
        return parser.parse_args(argv)


def main(*, argv: list[str] | None = None) -> int:
    args = Tossing3DLaunch.parse(argv=sys.argv[1:] if argv is None else argv)
    if args.mode == "worker":
        return run_worker(manifest=args.manifest, index=args.task, expected_sha=args.sha)
    arms = Tossing3DLaunch.arms()
    if args.mode == "pilot":
        args.size = "pilot"
    names = args.arms if args.arms is not None else list(DEFAULT_ARMS)
    unknown = sorted(set(names) - set(arms))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(arms)}")
    selected = [arms[name] for name in names]
    if args.no_bin_pouring:
        selected = [
            arm.model_copy(update={"flags": [*arm.flags, "--no-bin-pouring"]}) for arm in selected
        ]
    seeds = (
        tuple(args.seeds)
        if args.seeds is not None
        else ((0, 2) if args.size in ("pilot", "long-pilot") else tuple(range(SEEDS)))
    )
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise SystemExit("seeds must be distinct non-negative integers")
    size = SIZES[args.size]
    if args.mode == "pilot":
        return run_pilots(arms=selected, seeds=seeds, results_root=args.results_root)
    if args.mode == "print":
        for arm in selected:
            argv0 = Tossing3DLaunch.cli_argv(
                arm=arm, size=size, seed=0, output_dir=Path(f"<root>/{arm.name}/{arm.method}/0")
            )
            print(f"# {arm.name}\n{shlex.join(argv0)}\n")
        return 0
    if args.mode == "check":
        from hitl_pmp.cli import Cli

        for arm in selected:
            parsed = Cli.parse_args(
                argv=Tossing3DLaunch.cli_argv(
                    arm=arm, size=size, seed=0, output_dir=Path("/unused")
                )
            )
            print(f"ok  {arm.name}: --method {parsed.method}, {parsed.practice_step_budget} steps")
        return 0
    if args.tag is None or args.out is None:
        raise SystemExit("della needs --tag and --out")
    sha = args.sha or ShelfLaunch.head_commit()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise SystemExit("--sha must name a full frozen commit")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.tag):
        raise SystemExit("--tag must contain only letters, digits, underscores and hyphens")
    args.out.mkdir(parents=True, exist_ok=True)
    for group, limits in DELLA_GROUPS.items():
        if size in (PILOT, LONG_PILOT):
            limits = DellaGroup(time="02:00:00")
        members = [arm for arm in selected if Tossing3DLaunch.della_group(arm=arm) == group]
        if not members:
            continue
        lines = Tossing3DLaunch.della_manifest(
            arms=members, size=size, sha=sha, tag=args.tag, seeds=seeds
        )
        manifest = args.out / f"manifest-{group}.tsv"
        sbatch = args.out / f"array-{group}.sbatch"
        manifest.write_text("\n".join(lines) + "\n")
        sbatch.write_text(
            della_sbatch(
                tag=args.tag,
                group=group,
                limits=limits,
                tasks=len(lines),
                sha=sha,
            )
        )
        print(f"{group}: {len(lines)} tasks, --time={limits.time}: sbatch {sbatch} {manifest}")
    return 0


def della_sbatch(*, tag: str, group: str, limits: DellaGroup, tasks: int, sha: str) -> str:
    """Write a worker invocation; manifest arguments are parsed as data, never eval'd."""
    return f"""#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time={limits.time}
#SBATCH --array=1-{tasks}
#SBATCH --job-name={tag}-{group}
#SBATCH --output={DELLA}/slurm-logs/%x-%A_%a.out
set -euo pipefail
CODE={DELLA}/code/{sha}
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV={DELLA}/envs/hitl-pmp
export FD_EXEC_PATH={DELLA}/downward
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export HITL_PMP_TMPDIR={DELLA}/tmp/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p "$HITL_PMP_TMPDIR"
export TMP="$HITL_PMP_TMPDIR" TMPDIR="$HITL_PMP_TMPDIR" TEMP="$HITL_PMP_TMPDIR"
export XDG_RUNTIME_DIR="$HITL_PMP_TMPDIR/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
exec "$CODE/scripts/with_env.sh" python -m scripts.tossing3d_launch worker \\
  --sha {sha} --manifest "$1" --task "$SLURM_ARRAY_TASK_ID"
"""


def preflight(*, argv: list[str], output: Path, frozen_sha: str | None = None) -> None:
    """Check the imported code and compiled physics, and save the effective settings."""
    import importlib

    import numpy as np

    from hitl_pmp.cli import Cli
    from hitl_pmp.config_snapshot import ConfigSnapshot
    from hitl_pmp.environments.tossing3d.cli import Tossing3DCli
    from hitl_pmp.environments.tossing3d.skill_provider import Tossing3DSkillProvider
    from hitl_pmp.step_protocol import StepProtocolValidation

    args = Cli.parse_args(argv=argv)
    StepProtocolValidation.check(args=args)
    identity = ConfigSnapshot.collect(args=args, fd_exec_path=os.environ.get("FD_EXEC_PATH"))
    if frozen_sha is not None:
        if identity.git_commit != frozen_sha:
            raise RuntimeError(f"code revision {identity.git_commit} != frozen {frozen_sha}")
        subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--ignore-submodules=untracked"],
            check=True,
        )
        unknown = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO),
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "scripts",
                "configs",
            ],
            text=True,
        ).strip()
        if unknown:
            raise RuntimeError(f"uncommitted source files in the frozen checkout: {unknown}")
        for package, relative in (
            ("kinder", "reference/kindergarden"),
            ("kinder_models", "reference/kinder-baselines"),
        ):
            module_file = importlib.import_module(package).__file__
            if module_file is None:
                raise RuntimeError(f"{package} has no identifiable source file")
            imported = Path(module_file).resolve()
            if not imported.is_relative_to(REPO / relative):
                raise RuntimeError(f"{package} imports outside the frozen checkout: {imported}")
            expected = subprocess.check_output(
                ["git", "-C", str(REPO), "ls-tree", "HEAD", relative], text=True
            ).split()[2]
            actual = subprocess.check_output(
                ["git", "-C", str(REPO / relative), "rev-parse", "HEAD"], text=True
            ).strip()
            if expected != actual:
                raise RuntimeError(f"{package} is at {actual}, expected {expected}")
            subprocess.run(
                ["git", "-C", str(REPO / relative), "diff", "--quiet", "HEAD"], check=True
            )
    problem = Tossing3DCli.build_practice_problem(args=args)
    try:
        problem.hard_reset()
        env = problem.env
        backend = env.kinder_backend()
        model = backend._object_centric()._robot_env.sim.model.mj_model  # noqa: SLF001
        masses = {
            name: float(model.body_subtreemass[model.body(name).id]) for name in ("bin_0", "cube_0")
        }
        if not np.isclose(masses["bin_0"], args.bin_mass_kg) or not np.isclose(
            masses["cube_0"], args.cube_mass_kg
        ):
            raise RuntimeError(f"compiled masses differ from the run arguments: {masses}")
        if not model.body_jntnum[model.body("bin_0").id]:
            raise RuntimeError("the bin must remain movable, including in the ablation")
        offered = "PourCubeOutOfBin" in {
            skill.name for skill in Tossing3DSkillProvider(env=env).skills()
        }
        if offered != args.enable_bin_pouring:
            raise RuntimeError("pour availability differs from the run arguments")
        output.mkdir(parents=True, exist_ok=True)
        (output / "physics.json").write_text(
            json.dumps(
                {
                    "argv": argv,
                    "command": shlex.join([sys.executable, "-m", "hitl_pmp.cli", *argv]),
                    "working_directory": str(REPO),
                    "practice_cost_config": {
                        "path": str(args.practice_cost_config),
                        "sha256": hashlib.sha256(
                            Path(args.practice_cost_config).read_bytes()
                        ).hexdigest(),
                        "content": json.loads(Path(args.practice_cost_config).read_text()),
                    },
                    "masses_kg": masses,
                    "movable_bin": True,
                    "bin_pouring": offered,
                    "code": identity.model_dump(mode="json"),
                    "tmp": {name: os.environ.get(name) for name in ("TMP", "TMPDIR", "TEMP")},
                    "solver_source_commit": "2cd74021bd6e37b5fe87ef95cdcb6ca2412e4189",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        problem.env.close()


def run_worker(*, manifest: Path | None, index: int | None, expected_sha: str | None) -> int:
    if manifest is None or index is None or index < 1 or expected_sha is None:
        raise SystemExit("worker needs --manifest, positive --task, and --sha")
    lines = manifest.read_text().splitlines()
    if index > len(lines):
        raise SystemExit(f"no manifest task {index}")
    name, sha, arm, seed, directory, command = lines[index - 1].split("\t")
    if sha != expected_sha:
        raise RuntimeError(f"manifest revision {sha} != worker revision {expected_sha}")
    argv = shlex.split(command)
    output = Path(directory)
    preflight(argv=argv, output=output, frozen_sha=sha)
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.json").write_text(
        json.dumps(
            {
                "name": name,
                "arm": arm,
                "seed": int(seed),
                "source_commit": sha,
                "manifest": str(manifest.resolve()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "manifest_row": lines[index - 1],
                "task": index,
                "argv": argv,
                "command": shlex.join([sys.executable, "-m", "hitl_pmp.cli", *argv]),
                "working_directory": str(REPO),
                "physics_record": "physics.json",
                "slurm": {
                    key: os.environ.get(key)
                    for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"starting {name}: arm={arm} seed={seed} sha={sha}", flush=True)
    return subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/scratch_watchdog.py"),
            "--path",
            "/scratch/gpfs/TSILVER",
            "--output-dir",
            str(output),
            "--",
            sys.executable,
            "-m",
            "hitl_pmp.cli",
            *argv,
        ],
        cwd=REPO,
        check=False,
    ).returncode


def run_pilots(
    *, arms: list[Tossing3DArm], seeds: tuple[int, ...], results_root: Path | None
) -> int:
    """Sequential pilots; invoke this mode under run-heavy on agni."""
    if results_root is None:
        raise SystemExit("pilot needs --results-root")
    outcomes = []
    for arm in arms:
        for seed in seeds:
            output = results_root.resolve() / arm.name / arm.method / str(seed)
            argv = Tossing3DLaunch.cli_argv(arm=arm, size=PILOT, seed=seed, output_dir=output)
            preflight(argv=argv, output=output)
            started = time.monotonic()
            print(f"pilot starting {arm.name} seed={seed}", flush=True)
            with (output / "cli.log").open("w") as log:
                completed = subprocess.run(
                    [sys.executable, "-m", "hitl_pmp.cli", *argv],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            outcome = {
                "arm": arm.name,
                "seed": seed,
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "output": str(output),
            }
            outcomes.append(outcome)
            (results_root / "pilot-outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
            print(json.dumps(outcome), flush=True)
            if completed.returncode:
                return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
