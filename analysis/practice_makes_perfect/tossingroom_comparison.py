"""Post-run analysis for the Tossing Room EES bring-up: EES learning curves at
several sampler-iteration budgets against the random-skills lower bound, in the
paper's own view (% evaluation tasks solved vs. number of online transitions).

Reads only already-produced output (CLAUDE.md's analysis/ convention -- never runs a
simulation): `<root>/<method>/<seed>/stats.json` as written by `--output-dir`, the
layout `scripts/run_sweep.py` produces.

The skill-oracle upper bound is drawn as a flat reference line rather than read from a
sweep, because it is already pinned by CI rather than measured here:
`tests/environments/tossingroom/test_integration.py` asserts 30/30 on the mixed goal
distribution. Re-deriving it with a sweep would spend compute to reproduce an assertion.

`--arm "label=path"` is repeatable so the sampler-iteration grid (1k / 10k / 100k)
plots as three EES curves on one axis. Every arm must have been run over the same
transition budget, since the x-axis is shared -- that is exactly what makes the
curves comparable, and it is why `PracticeCycleCli` gives EES and random-skills the
same two protocol flags.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Assigned in fixed order and never cycled, so an arm keeps its colour when another is
# added or dropped. Linestyle repeats the identity as a second channel, so the arms
# stay separable in greyscale and under colour-vision deficiency.
_ARM_STYLES: tuple[tuple[str, str], ...] = (
    ("tab:blue", "-"),
    ("tab:orange", "--"),
    ("tab:green", "-."),
    ("tab:purple", ":"),
)


class TossingRoomComparison:
    """A static-method container, never instantiated."""

    @staticmethod
    def per_seed_curves(*, root: Path, method: str) -> dict[str, dict[int, float]]:
        """seed -> {transitions: % solved}, one entry per stats.json under the root."""
        curves: dict[str, dict[int, float]] = {}
        for stats_path in sorted(root.glob(f"{method}/*/stats.json")):
            seed = stats_path.parent.name
            curves[seed] = {
                int(transitions): 100.0 * num_solved / num_total
                for transitions, num_solved, num_total in json.loads(stats_path.read_text())[
                    "evaluations"
                ]
            }
        return curves

    @staticmethod
    def mean_curve(*, root: Path, method: str) -> dict[int, tuple[float, float]]:
        """transitions -> (mean %, stderr %) across seeds."""
        by_transitions: dict[int, list[float]] = {}
        for curve in TossingRoomComparison.per_seed_curves(root=root, method=method).values():
            for transitions, percent in curve.items():
                by_transitions.setdefault(transitions, []).append(percent)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def summarize(*, root: Path, method: str) -> dict[str, float]:
        """The statistics this project reports alongside a mean -- variance and the
        worst seed, because a collapse-to-zero seed has repeatedly been the most
        informative signal here and a mean hides it entirely."""
        curves = TossingRoomComparison.per_seed_curves(root=root, method=method)
        if not curves:
            return {}
        first = [curve[min(curve)] for curve in curves.values()]
        final = [curve[max(curve)] for curve in curves.values()]
        downward = sum(
            1
            for curve in curves.values()
            for earlier, later in zip(sorted(curve), sorted(curve)[1:], strict=False)
            if curve[later] < curve[earlier]
        )
        return {
            "seeds": len(final),
            "first_mean": statistics.mean(first),
            "final_mean": statistics.mean(final),
            "final_sd": statistics.stdev(final) if len(final) > 1 else 0.0,
            "worst_seed": min(final),
            "seeds_at_zero": sum(1 for value in final if value == 0.0),
            "downward_steps": downward,
        }

    @staticmethod
    def _plot_curve(*, ax, curve: dict[int, tuple[float, float]], label: str, **kwargs) -> None:
        xs = sorted(curve)
        means = [curve[x][0] for x in xs]
        errs = [curve[x][1] for x in xs]
        (line,) = ax.plot(xs, means, label=label, linewidth=2, **kwargs)
        ax.fill_between(
            xs,
            [m - e for m, e in zip(means, errs, strict=True)],
            [m + e for m, e in zip(means, errs, strict=True)],
            color=line.get_color(),
            alpha=0.15,
            linewidth=0,
        )

    @staticmethod
    def render(
        *, arms: list[tuple[str, Path]], random_skills_root: Path | None, output: Path, title: str
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.axhline(
            100.0,
            color="grey",
            linestyle=(0, (2, 3)),
            linewidth=1.4,
            label="skill oracle (100%, pinned by CI)",
        )
        for index, (label, root) in enumerate(arms):
            color, linestyle = _ARM_STYLES[index % len(_ARM_STYLES)]
            TossingRoomComparison._plot_curve(
                ax=ax,
                curve=TossingRoomComparison.mean_curve(root=root, method="ees"),
                label=label,
                color=color,
                linestyle=linestyle,
            )
        if random_skills_root is not None:
            TossingRoomComparison._plot_curve(
                ax=ax,
                curve=TossingRoomComparison.mean_curve(
                    root=random_skills_root, method="random-skills"
                ),
                label="random skills",
                color="tab:red",
                linestyle=(0, (1, 1)),
            )
        ax.set_xlabel("Number of online transitions")
        ax.set_ylabel("% evaluation tasks solved")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(-3, 107)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(output, dpi=150)

    @staticmethod
    def print_table(*, arms: list[tuple[str, Path]], random_skills_root: Path | None) -> None:
        rows = [(label, root, "ees") for label, root in arms]
        if random_skills_root is not None:
            rows.append(("random skills", random_skills_root, "random-skills"))
        header = (
            f"{'arm':<28}{'seeds':>6}{'first':>8}{'final':>8}"
            f"{'sd':>7}{'worst':>7}{'zeros':>7}{'down':>6}"
        )
        print(header)
        print("-" * len(header))
        for label, root, method in rows:
            summary = TossingRoomComparison.summarize(root=root, method=method)
            if not summary:
                print(f"{label:<28}{'(no stats.json found)':>42}")
                continue
            print(
                f"{label:<28}{summary['seeds']:>6.0f}{summary['first_mean']:>8.1f}"
                f"{summary['final_mean']:>8.1f}{summary['final_sd']:>7.1f}"
                f"{summary['worst_seed']:>7.1f}{summary['seeds_at_zero']:>7.0f}"
                f"{summary['downward_steps']:>6.0f}"
            )


def _parse_arm(*, raw: str) -> tuple[str, Path]:
    label, separator, path = raw.partition("=")
    if not separator or not label:
        raise ValueError(f"--arm must look like label=path, got {raw!r}")
    return label, Path(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        required=True,
        help='Repeatable, "label=results-root". One EES curve per arm.',
    )
    parser.add_argument("--random-skills-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Tossing Room: EES vs. the random-skills lower bound")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    arms = [_parse_arm(raw=raw) for raw in args.arm]
    TossingRoomComparison.print_table(arms=arms, random_skills_root=args.random_skills_root)
    TossingRoomComparison.render(
        arms=arms,
        random_skills_root=args.random_skills_root,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
