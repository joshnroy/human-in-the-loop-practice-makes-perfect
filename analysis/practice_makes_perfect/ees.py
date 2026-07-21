"""Post-run analysis for the EES ("Practice Makes Perfect") reproduction: reads
stats.json files written by hitl_pmp.method_runner.MethodRunner.run under
--output-dir and produces the paper's own Figure 4 view -- percentage of
evaluation tasks solved vs. number of online transitions collected, one line per
approach, solid line = mean across seeds and shading = standard error.

Never runs a simulation or drives a Problem/Method itself -- see CLAUDE.md's
analysis/ convention: this only reads --output-dir output already produced by
`python -m hitl_pmp.cli --env lightswitch --method <name> ... --output-dir
<results-root>/<method>/<seed>`.

Expects --results-root DIR laid out as DIR/<method>/<seed>/stats.json. A method
with a single checkpoint (e.g. --method skill-oracle, which never practices) is
drawn as a flat dashed reference line across the whole x-range rather than a
single point, since it is a constant upper/lower bound rather than a curve.
"""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering -- no GUI backend needed/available in CI

import matplotlib.pyplot as plt  # noqa: E402

from hitl_pmp.core.metrics.metrics import Metrics  # noqa: E402


class EesAnalysis:
    """A static-method container, never instantiated, same as every other
    business-logic class in this project."""

    @staticmethod
    def load_curves(*, method_dir: Path) -> list[list[tuple[int, float]]]:
        """One (transitions, fraction_solved) curve per seed. Computed by
        `Metrics` itself (task_training_curve) rather than recomputed here, so
        there is exactly one implementation of that arithmetic."""
        curves: list[list[tuple[int, float]]] = []
        for stats_path in sorted(method_dir.glob("*/stats.json")):
            metrics = Metrics.model_validate_json(stats_path.read_text())
            curves.append(metrics.task_training_curve())
        return curves

    @staticmethod
    def summarize(*, results_root: Path) -> dict[str, list[list[tuple[int, float]]]]:
        return {
            method_dir.name: EesAnalysis.load_curves(method_dir=method_dir)
            for method_dir in sorted(results_root.iterdir())
            if method_dir.is_dir()
        }

    @staticmethod
    def mean_and_stderr(
        *, curves: list[list[tuple[int, float]]]
    ) -> tuple[list[int], list[float], list[float]]:
        """Aligned by checkpoint index (every seed runs the same protocol, so the
        transition counts coincide); truncated to the shortest curve so a
        partially-finished seed can't silently shorten or skew the others."""
        usable = [curve for curve in curves if curve]
        if not usable:
            return [], [], []
        length = min(len(curve) for curve in usable)
        transitions = [usable[0][index][0] for index in range(length)]
        means: list[float] = []
        stderrs: list[float] = []
        for index in range(length):
            values = [curve[index][1] for curve in usable]
            means.append(statistics.mean(values))
            stderrs.append(
                statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
            )
        return transitions, means, stderrs

    @staticmethod
    def print_table(*, summary: dict[str, list[list[tuple[int, float]]]]) -> None:
        print(f"{'method':<16}{'transitions':>12}{'seeds':>7}{'mean':>9}{'stderr':>9}")
        for method, curves in summary.items():
            transitions, means, stderrs = EesAnalysis.mean_and_stderr(curves=curves)
            for index, checkpoint in enumerate(transitions):
                print(
                    f"{method:<16}{checkpoint:>12}{len(curves):>7}"
                    f"{means[index]:>9.1%}{stderrs[index]:>9.1%}"
                )

    @staticmethod
    def plot(*, summary: dict[str, list[list[tuple[int, float]]]], output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            max_transitions = max(
                (
                    EesAnalysis.mean_and_stderr(curves=curves)[0][-1]
                    for curves in summary.values()
                    if EesAnalysis.mean_and_stderr(curves=curves)[0]
                ),
                default=0,
            )
            for method, curves in sorted(summary.items()):
                transitions, means, stderrs = EesAnalysis.mean_and_stderr(curves=curves)
                if not transitions:
                    continue
                if len(transitions) == 1:
                    # A method with no practice phase is a constant reference
                    # level, not a curve -- draw it spanning the whole x-range.
                    ax.axhline(means[0], linestyle="--", alpha=0.7, label=f"{method} (no practice)")
                    continue
                ax.plot(transitions, means, marker="o", label=method)
                ax.fill_between(
                    transitions,
                    [mean - err for mean, err in zip(means, stderrs, strict=True)],
                    [mean + err for mean, err in zip(means, stderrs, strict=True)],
                    alpha=0.2,
                )
            ax.set_xlabel("Number of online transitions")
            ax.set_ylabel("Fraction of evaluation tasks solved")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, max_transitions if max_transitions else None)
            ax.set_title("Light Switch (25 cells): EES vs. baselines")
            ax.legend()
            fig.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
        finally:
            plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional learning-curve PNG path."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = EesAnalysis.summarize(results_root=args.results_root)
    EesAnalysis.print_table(summary=summary)
    if args.output is not None:
        EesAnalysis.plot(summary=summary, output_path=args.output)
        print(f"\nWrote plot to {args.output}")


if __name__ == "__main__":
    main()
