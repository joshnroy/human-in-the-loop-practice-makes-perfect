"""The SweepOffShelf3D practice experiment: every arm's argv, in one place.

    python -m scripts.sweepoffshelf3d_launch print  [--arms A ...]
    python -m scripts.sweepoffshelf3d_launch check  [--arms A ...]
    python -m scripts.sweepoffshelf3d_launch launch --arms A ... --results-root DIR --go

`print` writes each arm's `hitl_pmp.cli` argv (seed 0); `check` parses every arm with
the real CLI; `launch` runs `scripts/run_sweep.py` once per arm, one arm after another,
into `<results-root>/<arm>/<method>/<seed>/`. Run it through `scripts/with_env.sh` from
the checkout under test, so `hitl_pmp` and KINDER both import from it.

The plan (Josh, 2026-10-14 and 2026-10-20):
- arms: EES, ours with determinized A*, ours with expectimax at depth 6 and at depth 7
  (`--expectimax-depths`), and an always-ask variant of each. "Ours" is
  `--method pomdp` with Model B (`local_trend`).
- p = 1 (`--reset-cubes-success-probability 1`), seeds 0-5 (six, because a paired exact
  sign-flip test over five pairs cannot reach p < 0.05).
- 85,000 counted practice steps, an evaluation every 15,000, 3,000 control steps per
  held-out task, 10 tasks, stop after 3 perfect evaluations, sessions of <= 20 actions.
- `--pomdp-observation-probability-weight 0.0`: the method's default is 0.1, and at 0.1
  the planner STOPs at the root. EES uses a Beta(1, 3) prior, a labelled deviation from
  the paper's Beta(10, 1). Step accounting is `configs/sweepoffshelf3d_practice_costs.json`
  (1 per control step, each human reset costs 1 and takes 1 step, lambda 3e-6). The lab2
  scene is the environment's default; no arm passes `--no-scene-bg`.

The always-ask arms add `--always-ask-reset` (PR #477) and use #477's cost file, which
also prices `reset_to_sampled_start`. Every mode refuses them on a checkout whose CLI
does not accept that flag. The skill benchmark launch gate (`hitl_pmp/launch_gate.py`)
is never skipped unless `--skip-skill-benchmark-gate` is passed here explicitly.

Before a launch:
1. A clean worktree at the launch commit with its own `reference/` populated
   (`git submodule update --init reference/kindergarden reference/kinder-baselines`) and
   the MimicLabs meshes and textures copied in; `launch` refuses if `hitl_pmp`, `kinder`
   or `kinder_models` imports from outside it.
2. A skill benchmark report for that commit, or Josh's explicit gate override.
3. Disk: a POMDP run's `pomdp_decisions.jsonl` is tens of GB. Check free space first
   (`df -h` locally; `checkquota` on a cluster scratch filesystem).
4. On agni, run each arm as its own systemd service (a `--scope` dies with its shell)
   inside `agents.slice`, with `MemorySwapMax=0`; budget about 3 GiB per run.

After a run finishes, compress its decision logs:

    find <results-root>/<arm> -name '*_decisions.jsonl' -exec zstd -T0 --rm {} +

Nothing on main reads `.jsonl.zst` yet, so decompress (`zstd -d`) before analysis.

On a cluster, `print` gives each arm's per-seed argv; run it with the job's own
`scripts/with_env.sh` and the same `<arm>/<method>/<seed>` output layout.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO = Path(__file__).resolve().parents[1]
COSTS = REPO / "configs" / "sweepoffshelf3d_practice_costs.json"
ALWAYS_ASK_COSTS = REPO / "configs" / "sweepoffshelf3d_practice_costs_sampled_start.json"
ENV = "sweepoffshelf3d"
ALWAYS_ASK_FLAG = "--always-ask-reset"
SEEDS = 6
P = "1"


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    practice_steps: int
    evaluation_interval: int
    evaluation_control_steps: int
    tasks: int


FULL = Budget(
    practice_steps=85000, evaluation_interval=15000, evaluation_control_steps=3000, tasks=10
)
DRY = Budget(practice_steps=1500, evaluation_interval=1500, evaluation_control_steps=300, tasks=1)
SIZES = {"full": FULL, "dry": DRY}

# Every flag the 2026-10-13 shelf runs set, kept by the 2026-10-14 stage-5 rerun.
OCT13 = [
    "--no-ees-reset-gate",
    "--goal-pursuit-horizon", "0",
    "--exploration-epsilon", "0.5",
    "--planning-timeout", "10.0",
    "--sampler-max-train-iters", "10000",
    "--test-env-seed-offset", "10000",
    "--record-episode-traces",
    "--record-sampler-draws",
    "--no-reproduce-predicators-double-observe",
    "--reproduce-predicators-practice-target-history",
]  # fmt: skip
EES = [
    "--ees-prior-alpha", "1",
    "--ees-prior-beta", "3",
    "--ees-score-human-skills",
    "--reproduce-predicators-explore-target-only",
    "--reproduce-predicators-seen-task-order",
    "--reproduce-predicators-skip-perfect",
    "--reproduce-predicators-random-when-stranded",
]  # fmt: skip
PLANNER = [
    "--pomdp-competence-model", "local_trend",
    "--pomdp-inference-engine", "grid",
    "--pomdp-grid-competence-bins", "25",
    "--pomdp-grid-learning-rate-bins", "16",
    "--pomdp-competence-evidence", "non_epsilon",
    "--pomdp-observation-probability-weight", "0.0",
    "--pomdp-max-search-iterations", "100",
    "--pomdp-linear-cost-lambda", "3e-06",
    "--pomdp-competence-process-noise-std", "0.03",
    "--pomdp-learning-rate-decay", "0.9",
    "--pomdp-learning-rate-max", "0.15",
    "--pomdp-learning-rate-process-noise-std", "0.005",
    "--pomdp-learning-time-scale", "1.0",
    "--pomdp-num-particles", "1024",
    "--pomdp-num-samples", "100",
    "--no-pomdp-log-full-search-tree",
    "--no-reproduce-predicators-explore-target-only",
]  # fmt: skip


class ShelfArm(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    method: str
    flags: list[str]
    always_ask: bool


class ShelfLaunch:
    """A static-method container, never instantiated."""

    @staticmethod
    def arms(*, expectimax_depths: list[int]) -> dict[str, ShelfArm]:
        canonical = [
            ShelfArm(name="ees", method="ees", flags=EES, always_ask=False),
            ShelfArm(
                name="cccc-determinized",
                method="pomdp",
                flags=["--pomdp-solver", "determinized_astar", *PLANNER],
                always_ask=False,
            ),
            *(
                ShelfArm(
                    name=f"cccc-expectimax-d{depth}",
                    method="pomdp",
                    flags=[
                        "--pomdp-solver",
                        "expectimax",
                        "--pomdp-search-depth",
                        str(depth),
                        *PLANNER,
                    ],  # fmt: skip
                    always_ask=False,
                )
                for depth in expectimax_depths
            ),
        ]
        always_ask = [
            ShelfArm(
                name=f"{arm.name}-always-ask",
                method=arm.method,
                flags=[*arm.flags, ALWAYS_ASK_FLAG],
                always_ask=True,
            )
            for arm in canonical
        ]
        return {arm.name: arm for arm in [*canonical, *always_ask]}

    @staticmethod
    def method_args(*, arm: ShelfArm, size: Budget) -> list[str]:
        """Every `hitl_pmp.cli` flag except `--env/--method/--seed/--output-dir`."""
        costs = ALWAYS_ASK_COSTS if arm.always_ask else COSTS
        return [
            *arm.flags,
            "--practice-step-budget", str(size.practice_steps),
            "--measurement-interval-steps", str(size.evaluation_interval),
            "--evaluation-control-steps", str(size.evaluation_control_steps),
            "--num-test-tasks", str(size.tasks),
            "--stop-after-perfect-evaluations", "3",
            "--max-steps-per-interaction", "20",
            "--competence-window-size", "5",
            "--competence-recency-size", "5",
            "--practice-reset-policy", "never",
            "--defer-rendering",
            "--human-reset-practice-cost", "1.0",
            "--practice-cost-config", str(costs),
            "--reset-cubes-success-probability", P,
            *OCT13,
        ]  # fmt: skip

    @staticmethod
    def cli_argv(*, arm: ShelfArm, size: Budget, seed: int, output_dir: Path) -> list[str]:
        """One run's argv after `python -m hitl_pmp.cli`, as `run_sweep` builds it."""
        return [
            "--env", ENV,
            "--method", arm.method,
            "--seed", str(seed),
            "--output-dir", str(output_dir),
            *ShelfLaunch.method_args(arm=arm, size=size),
        ]  # fmt: skip

    @staticmethod
    def sweep_command(
        *, arm: ShelfArm, size: Budget, results_root: Path, max_workers: int, skip_gate: bool
    ) -> list[str]:
        command = [
            sys.executable, "-m", "scripts.run_sweep",
            "--env", ENV,
            "--methods", arm.method,
            "--num-seeds", str(SEEDS),
            "--results-root", str(results_root / arm.name),
            "--max-workers", str(max_workers),
            "--shared-args", shlex.join(ShelfLaunch.method_args(arm=arm, size=size)),
        ]  # fmt: skip
        if skip_gate:
            command.append("--skip-skill-benchmark-gate")
        return command

    @staticmethod
    def always_ask_supported(*, method: str) -> bool:
        """Whether this checkout's CLI accepts `--always-ask-reset` for `method`."""
        from hitl_pmp.cli import ENVIRONMENTS, METHODS, Cli

        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        Cli.add_global_arguments(parser=parser)
        ENVIRONMENTS[ENV].add_arguments(parser=parser)
        METHODS[method].add_arguments(parser=parser)
        _, unknown = parser.parse_known_args(["--env", ENV, "--method", method, ALWAYS_ASK_FLAG])
        return ALWAYS_ASK_FLAG not in unknown

    @staticmethod
    def require_runnable(*, arm: ShelfArm) -> None:
        if arm.always_ask and not ShelfLaunch.always_ask_supported(method=arm.method):
            raise SystemExit(
                f"refusing arm {arm.name}: it needs {ALWAYS_ASK_FLAG} (PR #477), which this "
                f"checkout's hitl_pmp.cli does not accept for --method {arm.method}. "
                "Select only the canonical arms with --arms, or launch from a checkout "
                "that has the flag."
            )

    @staticmethod
    def outside(*, repo: Path, files: dict[str, Path]) -> list[str]:
        """Each module whose file does not lie under `repo`."""
        return [
            f"{name}: {file}"
            for name, file in files.items()
            if not file.resolve().is_relative_to(repo.resolve())
        ]

    @staticmethod
    def require_isolated() -> None:
        """A sweep must not import a dependency out of another checkout (CLAUDE.md)."""
        import kinder
        import kinder_models

        import hitl_pmp

        modules = {"hitl_pmp": hitl_pmp, "kinder": kinder, "kinder_models": kinder_models}
        stray = ShelfLaunch.outside(
            repo=REPO, files={name: Path(str(m.__file__)) for name, m in modules.items()}
        )
        if stray:
            raise SystemExit(f"refusing to launch: imports from outside {REPO}: {stray}")

    @staticmethod
    def parse(*, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("mode", choices=["print", "check", "launch"])
        parser.add_argument("--arms", nargs="+", help="Default: every arm of the plan.")
        parser.add_argument("--expectimax-depths", nargs="+", type=int, default=[6, 7])
        parser.add_argument("--size", choices=sorted(SIZES), default="full")
        parser.add_argument("--results-root", type=Path)
        parser.add_argument("--max-workers", type=int, default=3, help="Runs in flight per arm.")
        parser.add_argument(
            "--skip-skill-benchmark-gate",
            action="store_true",
            help="Forward the launch gate override to run_sweep. Josh's explicit call only.",
        )
        parser.add_argument("--go", action="store_true", help="Required by launch.")
        return parser.parse_args(argv)


def main(*, argv: list[str] | None = None) -> int:
    args = ShelfLaunch.parse(argv=sys.argv[1:] if argv is None else argv)
    arms = ShelfLaunch.arms(expectimax_depths=args.expectimax_depths)
    names = args.arms if args.arms is not None else list(arms)
    unknown = sorted(set(names) - set(arms))
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {list(arms)}")
    selected = [arms[name] for name in names]
    for arm in selected:
        ShelfLaunch.require_runnable(arm=arm)
    size = SIZES[args.size]

    if args.mode == "print":
        for arm in selected:
            argv0 = ShelfLaunch.cli_argv(
                arm=arm, size=size, seed=0, output_dir=Path(f"<root>/{arm.name}/{arm.method}/0")
            )
            print(f"# {arm.name}\n{shlex.join(argv0)}\n")
        return 0

    if args.mode == "check":
        from hitl_pmp.cli import Cli

        for arm in selected:
            parsed = Cli.parse_args(
                argv=ShelfLaunch.cli_argv(arm=arm, size=size, seed=0, output_dir=Path("/unused"))
            )
            print(f"ok  {arm.name}: --method {parsed.method}, {parsed.practice_step_budget} steps")
        return 0

    if not args.go:
        raise SystemExit("launch starts the experiment: pass --go only on Josh's go")
    if args.results_root is None:
        raise SystemExit("launch needs --results-root")
    ShelfLaunch.require_isolated()
    failed = 0
    for arm in selected:
        command = ShelfLaunch.sweep_command(
            arm=arm,
            size=size,
            results_root=args.results_root,
            max_workers=args.max_workers,
            skip_gate=args.skip_skill_benchmark_gate,
        )
        print("+", shlex.join(command), flush=True)
        failed += subprocess.call(command, cwd=REPO) != 0
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
