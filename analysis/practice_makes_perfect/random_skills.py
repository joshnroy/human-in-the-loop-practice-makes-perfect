"""Post-run analysis for the Random Skills baseline (and any other --method run
alongside it for comparison, e.g. skill-oracle): reads stats.json files written
by hitl_pmp.method_runner.MethodRunner.run under --output-dir, aggregates test
success rate per (method, grid_size) across seeds, and reports a table plus an
optional success-rate-vs-grid_size plot. Never runs a simulation or drives a
Problem/Method itself -- see CLAUDE.md's analysis/ convention: this only reads
--output-dir output already produced by
`python -m hitl_pmp.cli --env lightswitch --method <name> --grid-size <n> --seed
<s> --output-dir <results-root>/<method>/<grid_size>/<seed>`.

Expects --results-root DIR laid out as DIR/<method>/<grid_size>/<seed>/stats.json
-- one subdirectory per method, then one per grid_size value, then one per seed,
matching what invoking the CLI once per (method, grid_size, seed) naturally
produces when --output-dir is pointed at DIR/<method>/<grid_size>/<seed>.
"""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402


class RandomSkillsAnalysis:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_success_rate(*, stats_path: Path) -> float:
        metrics = Metrics.model_validate_json(stats_path.read_text())
        return metrics.percentage_success_overall_test()

    @staticmethod
    def load_success_rates_by_grid_size(*, method_dir: Path) -> dict[int, list[float]]:
        by_grid_size: dict[int, list[float]] = {}
        for grid_size_dir in sorted(method_dir.iterdir(), key=lambda p: int(p.name)):
            if not grid_size_dir.is_dir():
                continue
            rates = [
                RandomSkillsAnalysis.load_success_rate(stats_path=stats_path)
                for stats_path in sorted(grid_size_dir.glob("*/stats.json"))
            ]
            by_grid_size[int(grid_size_dir.name)] = rates
        return by_grid_size

    @staticmethod
    def summarize(*, results_root: Path) -> dict[str, dict[int, list[float]]]:
        return {
            method_dir.name: RandomSkillsAnalysis.load_success_rates_by_grid_size(
                method_dir=method_dir
            )
            for method_dir in sorted(results_root.iterdir())
            if method_dir.is_dir()
        }

    @staticmethod
    def print_table(*, summary: dict[str, dict[int, list[float]]]) -> None:
        header = f"{'method':<16}{'grid_size':>10}{'seeds':>8}{'mean':>10}{'stdev':>10}"
        print(header)
        for method, by_grid_size in summary.items():
            for grid_size, rates in by_grid_size.items():
                mean = statistics.mean(rates) if rates else float("nan")
                stdev = statistics.stdev(rates) if len(rates) > 1 else 0.0
                print(f"{method:<16}{grid_size:>10}{len(rates):>8}{mean:>10.1%}{stdev:>10.1%}")

    @staticmethod
    def plot_success_rate_vs_grid_size(
        *, summary: dict[str, dict[int, list[float]]], output_path: Path
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            for method, by_grid_size in summary.items():
                grid_sizes = sorted(by_grid_size)
                means = [
                    statistics.mean(by_grid_size[g]) if by_grid_size[g] else 0.0 for g in grid_sizes
                ]
                ax.plot(grid_sizes, means, marker="o", label=method)
            ax.set_xlabel("grid_size")
            ax.set_ylabel("Test success rate")
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)
        finally:
            plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional success-rate-vs-grid_size PNG path."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = RandomSkillsAnalysis.summarize(results_root=args.results_root)
    RandomSkillsAnalysis.print_table(summary=summary)
    if args.output is not None:
        RandomSkillsAnalysis.plot_success_rate_vs_grid_size(
            summary=summary, output_path=args.output
        )
        print(f"\nWrote plot to {args.output}")


if __name__ == "__main__":
    main()
