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
- Eight arms: EES, ours with expectimax, ours with determinized A* (as is, and with
  `--pomdp-draw-scope skill`), and an always-ask variant of each (`--always-ask-reset`).
  Seeds 0-5: six, so a paired exact sign test can reach p = 0.031.
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
import shlex
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hitl_pmp.presets.arm_costs import Tossing3DArmCosts
from scripts.sweepoffshelf3d_launch import Budget, DellaGroup, ShelfLaunch

REPO = Path(__file__).resolve().parents[1]
ENV = "tossing3d"
ALWAYS_ASK_FLAG = "--always-ask-reset"
SEEDS = 6
DELLA = "/scratch/gpfs/TSILVER/jr2860"

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
SIZES = {"full": FULL, "dry": DRY}

# The scene and practice settings every arm shares (the preset sets them for the planner
# arms too; repeating them here keeps the two methods on one scene).
SCENE = [
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
PLANNER = ["--preset", "tossing3d-canonical"]


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
            # Josh, 2026-10-23: determinized also with one sampled outcome per acting
            # belief, so a cheap reset cannot re-roll a pick (#574).
            Tossing3DArm(
                name="determinized-skill-draws",
                method="pomdp",
                flags=[
                    *PLANNER,
                    "--pomdp-solver",
                    "determinized_astar",
                    "--pomdp-draw-scope",
                    "skill",
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
            "--stop-after-perfect-evaluations", "3",
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
    def della_manifest(*, arms: list[Tossing3DArm], size: Budget, sha: str, tag: str) -> list[str]:
        """One tab-separated line per arm x seed, arm-major, in array-task order."""
        code = Path(f"{DELLA}/code/{sha}")
        lines = []
        for arm in arms:
            for seed in range(SEEDS):
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
        parser.add_argument("mode", choices=["print", "check", "della"])
        parser.add_argument("--arms", nargs="+", help="Default: every arm of the plan.")
        parser.add_argument("--size", choices=sorted(SIZES), default="full")
        parser.add_argument("--tag", help="della: the results directory and job name.")
        parser.add_argument("--sha", help="della: the commit to run. Default: HEAD.")
        parser.add_argument("--out", type=Path, help="della: where to write the files.")
        return parser.parse_args(argv)


def main(*, argv: list[str] | None = None) -> int:
    args = Tossing3DLaunch.parse(argv=sys.argv[1:] if argv is None else argv)
    arms = Tossing3DLaunch.arms()
    names = args.arms if args.arms is not None else list(arms)
    unknown = sorted(set(names) - set(arms))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(arms)}")
    selected = [arms[name] for name in names]
    size = SIZES[args.size]
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
    args.out.mkdir(parents=True, exist_ok=True)
    for group, limits in DELLA_GROUPS.items():
        members = [arm for arm in selected if Tossing3DLaunch.della_group(arm=arm) == group]
        if not members:
            continue
        lines = Tossing3DLaunch.della_manifest(arms=members, size=size, sha=sha, tag=args.tag)
        manifest = args.out / f"manifest-{group}.tsv"
        sbatch = args.out / f"array-{group}.sbatch"
        manifest.write_text("\n".join(lines) + "\n")
        sbatch.write_text(
            ShelfLaunch.della_sbatch(tag=args.tag, group=group, limits=limits, tasks=len(lines))
        )
        print(f"{group}: {len(lines)} tasks, --time={limits.time}: sbatch {sbatch} {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
