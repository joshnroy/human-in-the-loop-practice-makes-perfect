"""The Tossing3D pick benchmark: generate its states, run a pick implementation on them,
report it.

    scripts/with_env.sh python -m scripts.tossing3d_pick_benchmark generate \
        [--states benchmarks/tossing3d/pick_states.jsonl] [--attribution <attribution.json>]

    systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
        scripts/with_env.sh python -m scripts.tossing3d_pick_benchmark run --out <dir> \
        [--workers 4] [--pick-impl module:Class] [--no-videos]

`--pick-impl` names a class with `name`, `plan_failure(*, env)` and `pick(*, env)` (see
`pick_benchmark/runner.py`'s `PickImplementation`); the default is the current
controller. `run` resumes from `<out>/results-*.jsonl`, then writes `report.json` and
`report.html`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from analysis.tossing3d_pick_benchmark import PickBenchmarkPage
from hitl_pmp.environments.tossing3d.environment import Tossing3DEnvironment
from hitl_pmp.environments.tossing3d.pick_benchmark.harvest import PickHarvest
from hitl_pmp.environments.tossing3d.pick_benchmark.report import PickReports
from hitl_pmp.environments.tossing3d.pick_benchmark.runner import (
    DEFAULT_IMPLEMENTATION,
    Implementations,
    PickShardWorker,
)
from hitl_pmp.environments.tossing3d.pick_benchmark.states import PickFamilies, PickStateSet

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STATES = REPO / "benchmarks/tossing3d/pick_states.jsonl"
DEFAULT_RUN_PREFIX = "exp22c/ModelB-lam3e-6/"


class PickBenchmarkCli:
    """A static-method container, never instantiated."""

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        sub = parser.add_subparsers(dest="command", required=True)
        generate = sub.add_parser("generate")
        generate.add_argument("--states", type=Path, default=DEFAULT_STATES)
        generate.add_argument("--attribution", type=Path, default=None)
        generate.add_argument("--run-prefix", default=DEFAULT_RUN_PREFIX)
        generate.add_argument("--only", default=None, help="comma-separated state-id prefixes")
        run = sub.add_parser("run")
        run.add_argument("--states", type=Path, default=DEFAULT_STATES)
        run.add_argument("--out", type=Path, required=True)
        run.add_argument("--workers", type=int, default=4)
        run.add_argument("--pick-impl", default=DEFAULT_IMPLEMENTATION)
        run.add_argument("--no-videos", action="store_true")
        report = sub.add_parser("report")
        report.add_argument("--states", type=Path, default=DEFAULT_STATES)
        report.add_argument("--out", type=Path, required=True)
        report.add_argument("--pick-impl", default=DEFAULT_IMPLEMENTATION)
        args = parser.parse_args(argv)
        if args.command == "generate":
            return PickBenchmarkCli.generate(args=args)
        started = time.monotonic()
        if args.command == "run":
            codes = PickShardWorker.launch(
                states_path=args.states,
                out=args.out,
                shards=args.workers,
                spec=args.pick_impl,
                videos=not args.no_videos,
            )
            print(f"workers exited {codes}")
        name = Implementations.load(spec=args.pick_impl).name
        result = PickReports.build(
            out=args.out,
            states_path=args.states,
            implementation=name,
            elapsed_seconds=time.monotonic() - started if args.command == "run" else 0.0,
        )
        page = PickBenchmarkPage.render(out=args.out)
        for verdict in result.skills:
            mark = "pass" if verdict.passed else "FAIL"
            print(f"{verdict.skill:<18}{verdict.successes}/{verdict.total} {mark} {verdict.labels}")
        print(f"{'PASSED' if result.passed else 'FAILED'}: {page}")
        return 0

    @staticmethod
    def generate(*, args: argparse.Namespace) -> int:
        harvests = []
        if args.attribution is not None:
            harvests = PickHarvest.refusals(
                attribution=json.loads(args.attribution.read_text()), run_prefix=args.run_prefix
            )
            print(f"harvested {len(harvests)} PickPlannable-false reset state(s)", flush=True)
        env = Tossing3DEnvironment(scene_bg=True)
        try:
            specs = list(PickFamilies.specs(env=env, harvests=harvests))
            if args.only:
                specs = [s for s in specs if s.state_id.startswith(tuple(args.only.split(",")))]
            known = {s.state_id for s in PickStateSet.load(path=args.states)}
            added = 0
            for spec in specs:
                if spec.state_id in known:
                    continue
                try:
                    state, note = PickFamilies.build(env=env, spec=spec)
                except Exception as exc:  # noqa: BLE001  (report and keep generating)
                    state, note = (
                        None,
                        f"{spec.state_id}: setup failed: {type(exc).__name__}: {exc}",
                    )
                print(note, flush=True)
                if state is not None:
                    added += len(PickStateSet.append(path=args.states, states=[state]))
            print(f"added {added}/{len(specs)} state(s) to {args.states}")
        finally:
            env.close()
        return 0


if __name__ == "__main__":
    sys.exit(PickBenchmarkCli.main())
