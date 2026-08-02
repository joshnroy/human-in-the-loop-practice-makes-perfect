"""Post-run analysis for the `ignore_effects` fix on Ball-Ring: the learning curve
before and after, against both baselines, the predicators reference, and the paper's
own Figure 4 curve.

Reads only already-produced outputs (CLAUDE.md's analysis/ convention -- never runs a
simulation): our arms from DIR/<method>/<seed>/stats.json, and the predicators
reference from a {seed: {transitions: frac_solved}} JSON aggregated out of predicators'
native result pickles.

The point of the figure is the *shape*, not just the endpoint. Before the fix every
Ball-Ring evaluation plan was structurally unexecutable, so whether a task was solved
came down to Fast Downward's arbitrary tie-breaking among exactly cost-tied plan
orderings -- a per-task lottery. That is why the "before" band is wide and jagged while
the "after" band is narrow, and why the honest headline is the variance collapse
(sd 24.5 -> 4.2) as much as the mean improving.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Ball-Ring EES read off the paper's Figure 4 (middle panel), transitions -> % solved
# (+-5-10). Traced from the green "EES (Ours)" mean line; see ballring_comparison.py.
_FIGURE_4_EES = {
    0: 0,
    150: 32,
    300: 48,
    500: 44,
    700: 41,
    900: 43,
    1100: 46,
    1300: 57,
    1500: 64,
    1700: 70,
    1900: 76,
    2100: 87,
    2300: 93,
    2500: 97,
}


class BallRingIgnoreEffects:
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
        return BallRingIgnoreEffects._summarize(by_transitions=by_transitions)

    @staticmethod
    def predicators_curve(*, json_path: Path) -> dict[int, tuple[float, float]]:
        per_seed: dict[str, dict[str, float]] = json.loads(json_path.read_text())
        by_transitions: dict[int, list[float]] = {}
        for seed_curve in per_seed.values():
            for transitions, frac in seed_curve.items():
                by_transitions.setdefault(int(transitions), []).append(100.0 * frac)
        return BallRingIgnoreEffects._summarize(by_transitions=by_transitions)

    @staticmethod
    def _summarize(*, by_transitions: dict[int, list[float]]) -> dict[int, tuple[float, float]]:
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def _plot(*, ax, curve: dict[int, tuple[float, float]], label: str, **kwargs) -> None:
        if not curve:
            return
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
        before: Path,
        after: Path,
        baselines: Path,
        predicators_json: Path,
        output: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        # The privileged oracle is a non-learning baseline, so it is run with
        # num_cycles=0 and has exactly ONE evaluation point -- plotting it as a curve
        # would draw an invisible single marker. Draw it as the flat reference line it
        # actually is.
        oracle = BallRingIgnoreEffects.mean_curve(root=baselines, method="skill-oracle")
        if oracle:
            level = oracle[max(oracle)][0]
            ax.axhline(
                level,
                color="grey",
                linestyle="--",
                linewidth=1.5,
                label=f"Skill Oracle, privileged upper bound ({level:.0f}%)",
            )
        BallRingIgnoreEffects._plot(
            ax=ax,
            curve=BallRingIgnoreEffects.predicators_curve(json_path=predicators_json),
            label="predicators (reference)",
            color="tab:green",
            linewidth=2,
        )
        BallRingIgnoreEffects._plot(
            ax=ax,
            curve=BallRingIgnoreEffects.mean_curve(root=after),
            label="ours, AFTER ignore_effects",
            color="tab:blue",
            linewidth=2.5,
        )
        BallRingIgnoreEffects._plot(
            ax=ax,
            curve=BallRingIgnoreEffects.mean_curve(root=before),
            label="ours, BEFORE (plans unexecutable)",
            color="tab:red",
            linewidth=2,
        )
        BallRingIgnoreEffects._plot(
            ax=ax,
            curve=BallRingIgnoreEffects.mean_curve(root=baselines, method="random-skills"),
            label="Random Skills (lower bound)",
            color="black",
            linestyle="-.",
            linewidth=1.2,
        )
        fig_xs = sorted(_FIGURE_4_EES)
        ax.plot(
            fig_xs,
            [_FIGURE_4_EES[x] for x in fig_xs],
            label="paper Figure 4, EES (eyeballed)",
            color="black",
            linestyle=":",
            linewidth=1.5,
        )
        ax.set_xlabel("Number of online transitions")
        ax.set_ylabel("% evaluation tasks solved")
        ax.set_title("Ball-Ring (Simulated): the ignore_effects fix, vs baselines and reference")
        ax.set_ylim(-3, 103)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output, dpi=150)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", type=Path, required=True)
    parser.add_argument("--after-root", type=Path, required=True)
    parser.add_argument("--baselines-root", type=Path, required=True)
    parser.add_argument("--predicators-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    BallRingIgnoreEffects.render(
        before=args.before_root,
        after=args.after_root,
        baselines=args.baselines_root,
        predicators_json=args.predicators_json,
        output=args.output,
    )


if __name__ == "__main__":
    main()
