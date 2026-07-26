"""Post-run analysis for the Ball-Ring EES reproduction: overlays our EES curves
against the predicators reference and the paper's Figure 4, as the paper's own view
(% evaluation tasks solved vs. number of online transitions).

Reads only already-produced outputs (CLAUDE.md's analysis/ convention -- never runs
a simulation): our runs from --ours-1k-root / --ours-100k-root laid out as
DIR/<method>/<seed>/stats.json, and the predicators reference from --predicators-json
(a {seed: {transitions: frac_solved}} dump aggregated from predicators' native result
pickles). Figure 4's Ball-Ring EES curve is included as an eyeballed dashed guide.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Ball-Ring EES read off Figure 4 (middle panel), transitions -> % solved (+-5-10).
_FIGURE_4_EES = {0: 0, 250: 48, 500: 42, 1000: 44, 1500: 45, 2000: 80, 2500: 97}


class BallRingComparison:
    """A static-method container, never instantiated."""

    @staticmethod
    def mean_curve(*, root: Path, method: str = "ees") -> dict[int, tuple[float, float]]:
        """transitions -> (mean %, stderr %) across seeds, from stats.json files."""
        by_transitions: dict[int, list[float]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            for transitions, num_solved, num_total in json.loads(stats_path.read_text())[
                "evaluations"
            ]:
                by_transitions.setdefault(transitions, []).append(100.0 * num_solved / num_total)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def predicators_curve(*, json_path: Path) -> dict[int, tuple[float, float]]:
        per_seed: dict[str, dict[str, float]] = json.loads(json_path.read_text())
        by_transitions: dict[int, list[float]] = {}
        for seed_curve in per_seed.values():
            for transitions, frac in seed_curve.items():
                by_transitions.setdefault(int(transitions), []).append(100.0 * frac)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def _plot_curve(*, ax, curve: dict[int, tuple[float, float]], label: str, **kwargs) -> None:
        xs = sorted(curve)
        means = [curve[x][0] for x in xs]
        errs = [curve[x][1] for x in xs]
        (line,) = ax.plot(xs, means, label=label, **kwargs)
        ax.fill_between(
            xs,
            [m - e for m, e in zip(means, errs, strict=True)],
            [m + e for m, e in zip(means, errs, strict=True)],
            color=line.get_color(),
            alpha=0.15,
        )

    @staticmethod
    def render(
        *,
        ours_1k: Path,
        ours_100k: Path,
        predicators_json: Path,
        output: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        BallRingComparison._plot_curve(
            ax=ax,
            curve=BallRingComparison.predicators_curve(json_path=predicators_json),
            label="predicators (reference, 100k iters)",
            color="tab:green",
            linewidth=2,
        )
        BallRingComparison._plot_curve(
            ax=ax,
            curve=BallRingComparison.mean_curve(root=ours_100k),
            label="ours, EES (100k sampler iters)",
            color="tab:blue",
            linewidth=2,
        )
        BallRingComparison._plot_curve(
            ax=ax,
            curve=BallRingComparison.mean_curve(root=ours_1k),
            label="ours, EES (1k iters, default)",
            color="tab:orange",
            linewidth=2,
        )
        fig_xs = sorted(_FIGURE_4_EES)
        ax.plot(
            fig_xs,
            [_FIGURE_4_EES[x] for x in fig_xs],
            label="paper Figure 4 (eyeballed)",
            color="black",
            linestyle=":",
            linewidth=1.5,
        )
        ax.set_xlabel("Number of online transitions")
        ax.set_ylabel("% evaluation tasks solved")
        ax.set_title("Ball-Ring (Simulated): EES vs. predicators vs. paper")
        ax.set_ylim(-3, 103)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output, dpi=150)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-1k-root", type=Path, required=True)
    parser.add_argument("--ours-100k-root", type=Path, required=True)
    parser.add_argument("--predicators-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    BallRingComparison.render(
        ours_1k=args.ours_1k_root,
        ours_100k=args.ours_100k_root,
        predicators_json=args.predicators_json,
        output=args.output,
    )


if __name__ == "__main__":
    main()
