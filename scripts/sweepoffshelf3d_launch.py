"""The SweepOffShelf3D practice experiment: every arm's argv, in one place.

    python -m scripts.sweepoffshelf3d_launch print  [--arms A ...]
    python -m scripts.sweepoffshelf3d_launch check  [--arms A ...]
    python -m scripts.sweepoffshelf3d_launch launch --arms A ... --results-root DIR --go
    python -m scripts.sweepoffshelf3d_launch della  --tag TAG --out DIR [--sha SHA]

`print` writes each arm's `hitl_pmp.cli` argv (seed 0); `check` parses every arm with
the real CLI; `launch` runs `scripts/run_sweep.py` once per arm, one arm after another,
into `<results-root>/<arm>/<method>/<seed>/`. Run it through `scripts/with_env.sh` from
the checkout under test, so `hitl_pmp` and KINDER both import from it.

The plan (Josh, 2026-10-14 and 2026-10-20):
- arms: EES, ours with determinized A*, ours with expectimax at depth 6, and an
  always-ask variant of each: six arms. Depth 7 was dropped (Josh, 2026-10-20: "let's
  drop d7") and stays selectable with `--expectimax-depths 6 7`. "Ours" is
  `--method pomdp` with Model B (`local_trend`).
- p = 1 (`--reset-cubes-success-probability 1`), seeds 0-5 (six, because a paired exact
  sign-flip test over five pairs cannot reach p < 0.05).
- 85,000 counted practice steps, an evaluation every 15,000, 3,000 control steps per
  held-out task, 10 tasks, stop after 3 perfect evaluations, sessions of <= 20 actions.
- `--pomdp-observation-probability-weight 0.0`: the method's default is 0.1, and at 0.1
  the planner STOPs at the root. EES uses a Beta(1, 3) prior, a labelled deviation from
  the paper's Beta(10, 1). EES also runs WITHOUT skip_perfect, a second labelled deviation:
  predicators' default (`active_sampler_explorer_skip_perfect = True`) scores a skill
  whose measured success rate is 1.0 at -inf, and a skill can measure 1.0 after only a
  few samples, so it would be wrongly treated as mastered (Josh, 2026-10-21).
  Step accounting: 1 per control step, each human reset takes 1 counted step of the
  budget, lambda 3e-6. The lab2 scene is the environment's default; no arm passes
  `--no-scene-bg`.
- The expectimax arms stop each search after `EXPECTIMAX_MAX_SEARCH_NODES` nodes, about
  60 s on Della, and act on the best root choice already evaluated exactly (Josh,
  2026-10-21). Prices, costs and the method are unchanged.

Prices (Josh, 2026-10-21), one rule in two units, because the two methods' costs are in
different units:
- The planner arms (determinized, expectimax, and their always-ask variants) price a
  robot skill at a FIXED nominal number of control steps per skill,
  `configs/sweepoffshelf3d_skill_step_costs.json` (`--pomdp-skill-step-costs`; measured
  from the skill benchmark by `scripts/skill_step_costs.py`, which records the commit and
  report sha256), and every human reset at `PLANNER_HUMAN_RESET_COST` = 500 control steps,
  Tossing3D's canonical price (`configs/sweepoffshelf3d_practice_costs_planner*.json`).
- The EES arms price every human reset at `EES_HUMAN_RESET_COST` = 5 in EES's own
  plan-cost units, where a robot skill costs -log(competence): EXP-22c's ees-lamcost5
  (`configs/sweepoffshelf3d_practice_costs_ees*.json`). EES reads the JSON human cost as
  its plan cost.
Only prices change: the practice budget and the x-axis are charged the real control
steps each skill took and one step per human reset, whatever its price. A different
reset price is a new pair of cost files; the tests pin each file to its constant.

The always-ask arms add `--always-ask-reset` (PR #477) and use the `_sampled_start`
cost file, which also prices `reset_to_sampled_start`. Every mode refuses them on a
checkout whose CLI does not accept that flag. The skill benchmark launch gate
(`hitl_pmp/launch_gate.py`) is never skipped unless `--skip-skill-benchmark-gate` is
passed here explicitly.

Before a launch:
1. A clean worktree at the launch commit with its own `reference/` populated
   (`git submodule update --init reference/kindergarden reference/kinder-baselines`) and
   the MimicLabs meshes and textures copied in; `launch` refuses if `hitl_pmp`, `kinder`
   or `kinder_models` imports from outside it.
2. A skill benchmark report for that commit, or Josh's explicit gate override.
3. Disk: a POMDP run writes `pomdp_decisions.jsonl.zst`, one zstd frame per record
   (PR #504); the analysis readers open it through `JsonlLog`. Check free space first
   (`df -h` locally; on Della `df` on `/scratch/gpfs/TSILVER` reports the fileset quota),
   and wrap long runs in `scripts/scratch_watchdog.py`.
4. On agni, run each arm as its own systemd service (a `--scope` dies with its shell)
   inside `agents.slice`, with `MemorySwapMax=0`; budget about 3 GiB per run.

Every other log is still written plain, `recording.jsonl` included. Every run records
full motion for its evaluations (`--record-motion evaluation`), and seed 0 of each arm
for practice too (`--record-motion all`), so `analysis/render_recording.py` draws
full-motion videos. `launch` shares one argv across seeds through `run_sweep`, so there
every seed records evaluation motion only.

On a cluster, `print` gives each arm's per-seed argv; run it with the job's own
`scripts/with_env.sh` and the same `<arm>/<method>/<seed>` output layout.

`della` writes one Slurm array per time group, since an array shares one `--time`:
`<out>/manifest-<group>.tsv` (one line per arm x seed: name, sha, arm, seed, output dir,
`hitl_pmp.cli` argv, tab-separated) and `<out>/array-<group>.sbatch`, 36 tasks in all for
the plan, and submits nothing. The groups and their limits are `DELLA_GROUPS`: `ees`
24 h, `determinized` 72 h (to be set from its Della dry run), `expectimax` 72 h;
`--della-time GROUP=HH:MM:SS` and `--della-qos GROUP=QOS` override them. No QOS is set
by default: Della's `cpu` partition assigns it from `--time`. Code is the worktree
`/scratch/gpfs/TSILVER/jr2860/code/<sha>` (`--sha` defaults to this checkout's HEAD);
results go to `/scratch/gpfs/TSILVER/jr2860/results/<tag>/<arm>/<method>/<seed>`. Each
task pre-flights `import zstandard` and the KINDER-import isolation check, then runs
`hitl_pmp.cli` under `scripts/scratch_watchdog.py --path /scratch/gpfs/TSILVER`
(`statvfs` there reports the fileset quota). The launch gate runs inside each task; it
needs the skill benchmark report for `<sha>` under the code worktree's
`benchmark-reports/` (or `$HITL_SKILL_BENCHMARK_DIR`), and is skipped only when
`--skip-skill-benchmark-gate` is passed here. Submit with
`sbatch <out>/array-<group>.sbatch <out>/manifest-<group>.tsv`, once per group.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

REPO = Path(__file__).resolve().parents[1]
# Each method's human-reset price, in its own units (see the module docstring).
PLANNER_HUMAN_RESET_COST = 500
EES_HUMAN_RESET_COST = 5
# Josh, 2026-10-21: bound expectimax's planning to about 60 s per decision, as a count so
# a seed still determines the run. 43 search nodes/s measured on a Della cpu node at the
# launch job shape (2 CPUs), from two depth-6 shelf decisions; see `--pomdp-max-search-nodes`.
EXPECTIMAX_MAX_SEARCH_NODES = 2600
PLANNER_COSTS = REPO / "configs" / "sweepoffshelf3d_practice_costs_planner.json"
PLANNER_ALWAYS_ASK_COSTS = (
    REPO / "configs" / "sweepoffshelf3d_practice_costs_planner_sampled_start.json"
)
EES_COSTS = REPO / "configs" / "sweepoffshelf3d_practice_costs_ees.json"
EES_ALWAYS_ASK_COSTS = REPO / "configs" / "sweepoffshelf3d_practice_costs_ees_sampled_start.json"
SKILL_STEP_COSTS = REPO / "configs" / "sweepoffshelf3d_skill_step_costs.json"
ENV = "sweepoffshelf3d"
ALWAYS_ASK_FLAG = "--always-ask-reset"
SEEDS = 6
P = "1"
# Full-motion recording: every run's evaluations, and practice too for these seeds.
PRACTICE_MOTION_SEEDS = frozenset({0})
DELLA = "/scratch/gpfs/TSILVER/jr2860"
DELLA_QUOTA_PATH = "/scratch/gpfs/TSILVER"


class DellaGroup(BaseModel):
    """One Slurm array: the arms that share a time limit."""

    model_config = ConfigDict(frozen=True)

    time: str
    qos: str | None = None


# Josh, 2026-10-20. The determinized limit is a placeholder until its Della dry run.
DELLA_GROUPS = {
    "ees": DellaGroup(time="24:00:00"),
    "determinized": DellaGroup(time="72:00:00"),
    "expectimax": DellaGroup(time="72:00:00"),
}


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

# Every flag the 2026-10-13 shelf runs set, kept by the 2026-10-14 stage-5 rerun, less
# `--record-episode-traces`: the step protocol never wrote that file and now refuses it.
OCT13 = [
    "--no-ees-reset-gate",
    "--goal-pursuit-horizon", "0",
    "--exploration-epsilon", "0.5",
    "--planning-timeout", "10.0",
    "--sampler-max-train-iters", "10000",
    "--test-env-seed-offset", "10000",
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
                        "--pomdp-max-search-nodes",
                        str(EXPECTIMAX_MAX_SEARCH_NODES),
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
    def cost_config(*, arm: ShelfArm) -> Path:
        """The arm's step-accounting file: its method's reset price, and the full reset
        too for an always-ask arm."""
        if arm.method == "ees":
            return EES_ALWAYS_ASK_COSTS if arm.always_ask else EES_COSTS
        return PLANNER_ALWAYS_ASK_COSTS if arm.always_ask else PLANNER_COSTS

    @staticmethod
    def method_args(*, arm: ShelfArm, size: Budget, repo: Path = REPO) -> list[str]:
        """Every `hitl_pmp.cli` flag except `--env/--method/--seed/--output-dir`.

        `repo` is the checkout the run executes from, which holds the cost files."""
        costs = repo / ShelfLaunch.cost_config(arm=arm).relative_to(REPO)
        fixed = (
            ["--pomdp-skill-step-costs", str(repo / SKILL_STEP_COSTS.relative_to(REPO))]
            if arm.method == "pomdp"
            else []
        )
        return [
            *arm.flags,
            *fixed,
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
    def cli_argv(
        *, arm: ShelfArm, size: Budget, seed: int, output_dir: Path, repo: Path = REPO
    ) -> list[str]:
        """One run's argv after `python -m hitl_pmp.cli`, as `run_sweep` builds it."""
        return [
            "--env", ENV,
            "--method", arm.method,
            "--seed", str(seed),
            "--output-dir", str(output_dir),
            *ShelfLaunch.method_args(arm=arm, size=size, repo=repo),
            "--record-motion", ShelfLaunch.motion_scope(seed=seed),
        ]  # fmt: skip

    @staticmethod
    def motion_scope(*, seed: int) -> str:
        return "all" if seed in PRACTICE_MOTION_SEEDS else "evaluation"

    @staticmethod
    def della_manifest(
        *, arms: list[ShelfArm], size: Budget, sha: str, tag: str, skip_gate: bool
    ) -> list[str]:
        """One tab-separated line per arm x seed, arm-major, in array-task order."""
        code = Path(f"{DELLA}/code/{sha}")
        lines = []
        for arm in arms:
            for seed in range(SEEDS):
                output_dir = Path(f"{DELLA}/results/{tag}/{arm.name}/{arm.method}/{seed}")
                argv = ShelfLaunch.cli_argv(
                    arm=arm, size=size, seed=seed, output_dir=output_dir, repo=code
                )
                if skip_gate:
                    argv.append("--skip-skill-benchmark-gate")
                fields = [f"{tag}-{arm.name}-s{seed}", sha, arm.name, str(seed)]
                lines.append("\t".join([*fields, str(output_dir), shlex.join(argv)]))
        return lines

    @staticmethod
    def della_group(*, arm: ShelfArm) -> str:
        if arm.method == "ees":
            return "ees"
        return "determinized" if "determinized" in arm.name else "expectimax"

    @staticmethod
    def della_groups(*, overrides: list[str], field: str) -> dict[str, DellaGroup]:
        """`DELLA_GROUPS` with each `GROUP=VALUE` override applied to `field`."""
        groups = dict(DELLA_GROUPS)
        for override in overrides:
            group, sep, value = override.partition("=")
            if not sep or not value or group not in groups:
                raise SystemExit(
                    f"bad override {override!r}: want GROUP=VALUE, GROUP one of {list(groups)}"
                )
            groups[group] = groups[group].model_copy(update={field: value})
        return groups

    @staticmethod
    def della_sbatch(*, tag: str, group: str, limits: DellaGroup, tasks: int) -> str:
        qos = f"#SBATCH --qos={limits.qos}\n" if limits.qos else ""
        return f"""#!/bin/bash
# One SweepOffShelf3D run per array task, group {group}:
#   sbatch array-{group}.sbatch manifest-{group}.tsv
# Written by `python -m scripts.sweepoffshelf3d_launch della`; never edit by hand.
# Manifest line i = <name>\\t<sha>\\t<arm>\\t<seed>\\t<output dir>\\t<hitl_pmp.cli argv>
#SBATCH --account=tsilver
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time={limits.time}
{qos}#SBATCH --array=1-{tasks}
#SBATCH --job-name={tag}-{group}
#SBATCH --output={DELLA}/slurm-logs/%x-%A_%a.out
set -eo pipefail
S={DELLA}
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$1")
IFS=$'\\t' read -r NAME SHA ARM SEED OUT ARGV <<<"$LINE"
[ -n "$ARGV" ] || {{ echo "no manifest line ${{SLURM_ARRAY_TASK_ID}}" >&2; exit 2; }}
CODE="{DELLA}/code/$SHA"
[ -d "$CODE" ] || {{ echo "no code worktree $CODE" >&2; exit 2; }}
echo "task=$SLURM_ARRAY_TASK_ID name=$NAME arm=$ARM seed=$SEED node=$(hostname) start=$(date -Is)"
export CONDA_SH=/usr/licensed/anaconda3/2026.7.1/etc/profile.d/conda.sh
export HITL_PMP_CONDA_ENV="$S/envs/hitl-pmp"
export FD_EXEC_PATH="$S/downward"
export DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export XDG_RUNTIME_DIR="${{TMPDIR:-/tmp}}/xdg-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
mkdir -p "$XDG_RUNTIME_DIR"
export LD_PRELOAD="$HITL_PMP_CONDA_ENV/lib/libstdc++.so.6"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CODE"
"$CODE/scripts/with_env.sh" python -c "import zstandard" || {{
  echo "preflight: zstandard does not import in $HITL_PMP_CONDA_ENV; run: pip install zstandard" >&2
  exit 3
}}
"$CODE/scripts/with_env.sh" python -c \\
  "from scripts.sweepoffshelf3d_launch import ShelfLaunch; ShelfLaunch.require_isolated()"
eval "ARGS=($ARGV)"
mkdir -p "$OUT"
set +e
"$CODE/scripts/with_env.sh" python "$CODE/scripts/scratch_watchdog.py" \\
  --path {DELLA_QUOTA_PATH} --output-dir "$OUT" \\
  -- "$CODE/scripts/with_env.sh" python -m hitl_pmp.cli "${{ARGS[@]}}"
rc=$?
echo "end=$(date -Is) rc=$rc"
exit $rc
"""

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
            # One argv for every seed, so no seed records practice motion here.
            "--shared-args", shlex.join(
                [*ShelfLaunch.method_args(arm=arm, size=size), "--record-motion", "evaluation"]
            ),
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
    def head_commit() -> str:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    @staticmethod
    def parse(*, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("mode", choices=["print", "check", "launch", "della"])
        parser.add_argument("--arms", nargs="+", help="Default: every arm of the plan.")
        parser.add_argument("--expectimax-depths", nargs="+", type=int, default=[6])
        parser.add_argument("--size", choices=sorted(SIZES), default="full")
        parser.add_argument("--results-root", type=Path)
        parser.add_argument("--max-workers", type=int, default=3, help="Runs in flight per arm.")
        parser.add_argument(
            "--skip-skill-benchmark-gate",
            action="store_true",
            help="Forward the launch gate override to run_sweep. Josh's explicit call only.",
        )
        parser.add_argument("--go", action="store_true", help="Required by launch.")
        parser.add_argument("--tag", help="della: the results directory and job name.")
        parser.add_argument("--sha", help="della: the commit to run. Default: HEAD.")
        parser.add_argument("--out", type=Path, help="della: where to write the files.")
        parser.add_argument(
            "--della-time",
            action="append",
            default=[],
            metavar="GROUP=HH:MM:SS",
            help=f"della: a group's --time. Groups: {list(DELLA_GROUPS)}.",
        )
        parser.add_argument(
            "--della-qos",
            action="append",
            default=[],
            metavar="GROUP=QOS",
            help="della: a group's --qos. Default: none (the partition picks it).",
        )
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

    if args.mode == "della":
        if args.tag is None or args.out is None:
            raise SystemExit("della needs --tag and --out")
        sha = args.sha or ShelfLaunch.head_commit()
        groups = ShelfLaunch.della_groups(overrides=args.della_time, field="time")
        for group, limits in ShelfLaunch.della_groups(
            overrides=args.della_qos, field="qos"
        ).items():
            groups[group] = groups[group].model_copy(update={"qos": limits.qos})
        args.out.mkdir(parents=True, exist_ok=True)
        for group, limits in groups.items():
            members = [arm for arm in selected if ShelfLaunch.della_group(arm=arm) == group]
            if not members:
                continue
            lines = ShelfLaunch.della_manifest(
                arms=members,
                size=size,
                sha=sha,
                tag=args.tag,
                skip_gate=args.skip_skill_benchmark_gate,
            )
            manifest = args.out / f"manifest-{group}.tsv"
            sbatch = args.out / f"array-{group}.sbatch"
            manifest.write_text("\n".join(lines) + "\n")
            sbatch.write_text(
                ShelfLaunch.della_sbatch(tag=args.tag, group=group, limits=limits, tasks=len(lines))
            )
            print(f"{group}: {len(lines)} tasks, --time={limits.time}: sbatch {sbatch} {manifest}")
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
