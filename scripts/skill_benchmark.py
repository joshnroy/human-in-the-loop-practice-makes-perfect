"""The SweepOffShelf3D skill benchmark: generate its states, run it, report it.

    # once per scene (and whenever a family is added); appends, never rewrites
    scripts/with_env.sh python -m scripts.skill_benchmark generate \
        [--states benchmarks/sweepoffshelf3d/states.jsonl] [--harvest <run dirs>...]

    # an experiment-scale run: on agni, under run-heavy. --out defaults to where the
    # launch gate (hitl_pmp/launch_gate.py) reads this commit's report.
    run-heavy scripts/with_env.sh python -m scripts.skill_benchmark run \
        [--out <dir>] [--workers 4] [--no-videos]

    scripts/with_env.sh python -m scripts.skill_benchmark report --out <dir>

`run` resumes: cases already in `<out>/results-*.jsonl` are skipped. It then writes
`<out>/report.json` (what the launch gate reads) and `<out>/report.html`.
"""

import argparse
import sys
import time
from pathlib import Path

from analysis.sweepoffshelf3d_skill_benchmark import SkillBenchmarkPage
from hitl_pmp.environments.sweepoffshelf3d.environment import SweepOffShelf3DEnvironment
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.harvest import HarvestedStates
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.report import DOMAIN, BenchmarkReports
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.runner import ShardWorker
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import (
    StateFamilies,
    StateSet,
    StateSpec,
)
from hitl_pmp.launch_gate import LaunchGate, LaunchIdentity

DEFAULT_STATES = Path(__file__).resolve().parents[1] / "benchmarks/sweepoffshelf3d/states.jsonl"
DEFAULT_WORKERS = 4


class SkillBenchmarkCli:
    """A static-method container, never instantiated."""

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        sub = parser.add_subparsers(dest="command", required=True)
        generate = sub.add_parser("generate")
        generate.add_argument("--states", type=Path, default=DEFAULT_STATES)
        generate.add_argument("--harvest", type=Path, nargs="*", default=[])
        generate.add_argument("--no-families", action="store_true")
        generate.add_argument("--only", default=None, help="comma-separated state-id prefixes")
        run = sub.add_parser("run")
        run.add_argument("--states", type=Path, default=DEFAULT_STATES)
        run.add_argument(
            "--out",
            type=Path,
            default=None,
            help="Default: where the launch gate reads this commit's report.",
        )
        run.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
        run.add_argument("--no-videos", action="store_true")
        report = sub.add_parser("report")
        report.add_argument("--states", type=Path, default=DEFAULT_STATES)
        report.add_argument("--out", type=Path, default=None)
        args = parser.parse_args(argv)
        if getattr(args, "out", "unset") is None:
            args.out = SkillBenchmarkCli.default_out()
        if args.command == "generate":
            return SkillBenchmarkCli.generate(args=args)
        if args.command == "run":
            started = time.monotonic()
            codes = ShardWorker.launch(
                states_path=args.states,
                out=args.out,
                shards=args.workers,
                videos=not args.no_videos,
            )
            print(f"workers exited {codes}")
            return SkillBenchmarkCli.report(args=args, elapsed=time.monotonic() - started)
        return SkillBenchmarkCli.report(args=args, elapsed=0.0)

    @staticmethod
    def default_out() -> Path:
        """`benchmark-reports/sweepoffshelf3d/<commit>/`: the launch gate's location."""
        commit = LaunchIdentity.current().git_commit
        return LaunchGate.report_path(
            root=LaunchGate.default_root(), domain=DOMAIN, commit=commit
        ).parent

    @staticmethod
    def generate(*, args: argparse.Namespace) -> int:
        env = SweepOffShelf3DEnvironment(scene_bg=True)
        specs: list[StateSpec] = []
        try:
            if not args.no_families:
                specs += list(StateFamilies.specs(env=env))
            if args.harvest:
                specs += HarvestedStates.specs(
                    harvests=HarvestedStates.candidates(paths=args.harvest)
                )
            if args.only:
                prefixes = tuple(args.only.split(","))
                specs = [spec for spec in specs if spec.state_id.startswith(prefixes)]
            known = {state.state_id for state in StateSet.load(path=args.states)}
            built = []
            for spec in specs:
                if spec.state_id in known:
                    continue
                state, note = StateFamilies.build(env=env, spec=spec)
                print(note, flush=True)
                if state is not None:
                    built.append(state)
                    StateSet.append(path=args.states, states=[state])
            print(f"added {len(built)}/{len(specs)} state(s) to {args.states}")
        finally:
            env.close()
        return 0

    @staticmethod
    def report(*, args: argparse.Namespace, elapsed: float) -> int:
        report = BenchmarkReports.build(
            out=args.out, states_path=args.states, elapsed_seconds=elapsed
        )
        page = SkillBenchmarkPage.render(out=args.out)
        for verdict in report.skills:
            mark = "pass" if verdict.passed else "FAIL"
            print(f"{verdict.skill:<28}{verdict.successes}/{verdict.total} {mark} {verdict.labels}")
        print(f"{'PASSED' if report.passed else 'FAILED'}: {page}")
        return 0


if __name__ == "__main__":
    sys.exit(SkillBenchmarkCli.main())
