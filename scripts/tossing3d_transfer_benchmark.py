"""Standard Tossing3D EES budget: same-side practice, stock far-side evaluation.

Ten seeds, 100 cycles, 20 actions per cycle, ten fixed evaluation tasks per seed.
All EES learning settings retain their normal defaults. Output directories enable
continuous per-period practice videos through the standard runner.
"""

import argparse
import json
from pathlib import Path

from scripts.run_sweep import SweepRun, SweepRunner


class TransferBenchmark:
    @staticmethod
    def plan(*, results_root: Path) -> list[SweepRun]:
        return SweepRunner.plan(
            env="tossing3d",
            methods=["ees"],
            seeds=list(range(10)),
            results_root=results_root,
            shared_args=[
                "--layout",
                "same-side",
                "--evaluation-layout",
                "barrier",
                "--num-test-tasks",
                "10",
                "--practice-reset-policy",
                "never",
                "--record-sampler-draws",
                "--record-skill-competence",
            ],
            method_args={"ees": ["--num-cycles", "100", "--max-steps-per-interaction", "20"]},
        )

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--max-workers", type=int, default=4)
        parser.add_argument("--dry-run", action="store_true")
        args = parser.parse_args()
        if args.max_workers < 1:
            parser.error("--max-workers must be positive")
        runs = TransferBenchmark.plan(results_root=args.results_root)
        if args.dry_run:
            print(json.dumps([run.model_dump(mode="json") for run in runs], indent=2))
            return
        if any(run.output_dir.exists() for run in runs):
            parser.error("Output seed directories already exist; choose a fresh results root")
        outcomes = SweepRunner.execute(runs=runs, max_workers=args.max_workers)
        failures = [outcome for outcome in outcomes if not outcome.succeeded]
        print(f"{len(outcomes) - len(failures)}/{len(outcomes)} seeds completed", flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    TransferBenchmark.main()
