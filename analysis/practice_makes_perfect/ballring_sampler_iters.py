"""Post-run analysis for the Ball-Ring sampler-iteration sweep, which asked how many
gradient steps per sampler refit (`--sampler-max-train-iters`) maximize held-out
performance -- and did not answer it.

Reads only already-produced outputs (CLAUDE.md's analysis/ convention -- never runs a
simulation). Unlike the sibling Ball-Ring scripts, which glob
`DIR/<method>/<seed>/stats.json` straight out of a sweep directory, this one reads a
committed *aggregate* of those runs, `{arm: {seed: [[transitions, solved, total], ...]}}`
-- the raw sweep directories lived outside the repo and did not travel between machines,
so the aggregate is the only surviving record of these arms.

The figure exists to show how *weak* the separation between these arms is. The point
estimates trace an inverted U -- best at 10000, worse at both 1000 and 100000 -- but with
10 seeds per arm no pairwise difference reaches significance, and every arm's endpoint
lies inside predicators' own +-1sd band. The honest reading is "the point estimates order
this way and nothing is resolved", not "10000 is the optimum".

The right-hand panel therefore plots sd rather than stderr, and the paired p-values
against the 10000 arm are drawn directly on it: an error bar the reader has to mentally
convert into a comparison is exactly how the too-strong reading survives. sd is also the
quantity that matters for choosing a default -- an arm that is right on average but
wildly variable is not a good default.

The p-values are computed elsewhere (scipy is not a dependency of this project) and
quoted as constants here; see the experiment log for the command that produced them.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Arm key in the aggregate JSON -> the --sampler-max-train-iters value it was run at.
_ARM_ITERS = {
    "iters1k": 1000,
    "iters3k": 3000,
    "iters10k": 10000,
    "iters30k": 30000,
    "iters100k": 100000,
}

# predicators' own Ball-Ring reference, 10 seeds at its native
# sampler_mlp_classifier_max_itr = 100000: final-sweep mean and sd, in percent.
# Aggregated from its result pickles into predicators-ballring-25cyc.json.
_REFERENCE_MEAN = 91.0
_REFERENCE_SD = 12.0

# Paired two-sided t-test of each arm's per-seed endpoint against the 10000 arm's, over
# the 10 shared seeds. Not one reaches p < 0.05. Computed with scipy.stats.ttest_rel --
# scipy is not a project dependency, so the values are quoted rather than recomputed
# here; the experiment log records the exact command.
_PAIRED_P_VS_10K = {1000: 0.057, 3000: 0.350, 30000: 0.070, 100000: 0.085}


class BallRingSamplerIters:
    """A static-method container, never instantiated."""

    @staticmethod
    def load_arms(*, json_path: Path) -> dict[str, dict[str, list[list[int]]]]:
        """arm -> seed -> [[transitions, num_solved, num_total], ...]."""
        arms: dict[str, dict[str, list[list[int]]]] = json.loads(json_path.read_text())
        missing = sorted(set(_ARM_ITERS) - set(arms))
        if missing:
            raise ValueError(f"aggregate JSON is missing sampler-iteration arms: {missing}")
        return arms

    @staticmethod
    def endpoint_percents(*, seed_curves: dict[str, list[list[int]]]) -> list[float]:
        """Per-seed % solved at the LAST evaluation sweep of each seed's run.

        The endpoint is the arm's score: every arm ran the same fixed number of cycles,
        so the last sweep is the comparable point. Taking a per-seed max instead would
        read 100% for all five arms -- every seed touches a perfect sweep somewhere --
        which is why that is not the summary anyone should use here.
        """
        percents = []
        for rows in seed_curves.values():
            transitions, num_solved, num_total = max(rows, key=lambda row: row[0])
            del transitions
            percents.append(100.0 * num_solved / num_total)
        return percents

    @staticmethod
    def mean_curve(*, seed_curves: dict[str, list[list[int]]]) -> dict[int, tuple[float, float]]:
        """transitions -> (mean %, stderr %) across seeds."""
        by_transitions: dict[int, list[float]] = {}
        for rows in seed_curves.values():
            for transitions, num_solved, num_total in rows:
                by_transitions.setdefault(transitions, []).append(100.0 * num_solved / num_total)
        out: dict[int, tuple[float, float]] = {}
        for transitions, values in sorted(by_transitions.items()):
            stderr = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
            out[transitions] = (statistics.mean(values), stderr)
        return out

    @staticmethod
    def summary_table(
        *, arms: dict[str, dict[str, list[list[int]]]]
    ) -> list[tuple[int, float, float]]:
        """(iters, mean %, sd %) per arm, ascending in iters."""
        rows = []
        for arm, iters in sorted(_ARM_ITERS.items(), key=lambda item: item[1]):
            percents = BallRingSamplerIters.endpoint_percents(seed_curves=arms[arm])
            sd = statistics.stdev(percents) if len(percents) > 1 else 0.0
            rows.append((iters, statistics.mean(percents), sd))
        return rows

    @staticmethod
    def render(*, arms_json: Path, output: Path) -> None:
        arms = BallRingSamplerIters.load_arms(json_path=arms_json)
        fig, (curve_ax, endpoint_ax) = plt.subplots(1, 2, figsize=(13, 5))

        colors = plt.get_cmap("viridis")
        ordered = sorted(_ARM_ITERS.items(), key=lambda item: item[1])
        for index, (arm, iters) in enumerate(ordered):
            curve = BallRingSamplerIters.mean_curve(seed_curves=arms[arm])
            xs = sorted(curve)
            means = [curve[x][0] for x in xs]
            errs = [curve[x][1] for x in xs]
            color = colors(index / max(len(ordered) - 1, 1))
            curve_ax.plot(
                xs,
                means,
                label=f"{iters:,} iters",
                color=color,
                linewidth=2.5 if iters == 10000 else 1.6,
            )
            curve_ax.fill_between(
                xs,
                [m - e for m, e in zip(means, errs, strict=True)],
                [m + e for m, e in zip(means, errs, strict=True)],
                color=color,
                alpha=0.12,
            )
        curve_ax.set_xlabel("Number of online transitions")
        curve_ax.set_ylabel("% evaluation tasks solved")
        curve_ax.set_title("Learning curves by sampler training budget")
        curve_ax.set_ylim(-3, 103)
        curve_ax.legend(loc="lower right", fontsize=9)
        curve_ax.grid(True, alpha=0.3)

        table = BallRingSamplerIters.summary_table(arms=arms)
        iters = [row[0] for row in table]
        means = [row[1] for row in table]
        sds = [row[2] for row in table]
        endpoint_ax.axhspan(
            _REFERENCE_MEAN - _REFERENCE_SD,
            _REFERENCE_MEAN + _REFERENCE_SD,
            color="tab:green",
            alpha=0.12,
        )
        endpoint_ax.axhline(
            _REFERENCE_MEAN,
            color="tab:green",
            linestyle="--",
            linewidth=1.5,
            label=f"predicators reference ({_REFERENCE_MEAN:.0f}% +- {_REFERENCE_SD:.0f})",
        )
        endpoint_ax.errorbar(
            iters,
            means,
            yerr=sds,
            marker="o",
            capsize=4,
            color="tab:blue",
            linewidth=2,
            label="ours (mean +- sd over 10 seeds)",
        )
        # Label each arm with its paired p-value against the 10000 arm, rather than
        # calling 10000 "the optimum": every one of these is above 0.05.
        for arm_iters, mean, sd in table:
            p_value = _PAIRED_P_VS_10K.get(arm_iters)
            if p_value is None:
                continue
            # Below the lower whisker, on an opaque patch: these labels sat on top of
            # the error bars when they were offset from the marker instead.
            endpoint_ax.annotate(
                f"p={p_value:.3f}",
                xy=(arm_iters, mean - sd),
                xytext=(0, -14),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color="dimgrey",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )
        endpoint_ax.set_xscale("log")
        endpoint_ax.set_xlabel("--sampler-max-train-iters (log scale)")
        endpoint_ax.set_ylabel("% evaluation tasks solved, final sweep")
        endpoint_ax.set_title("Endpoint vs sampler training budget (10 seeds/arm)")
        endpoint_ax.set_ylim(-3, 103)
        endpoint_ax.legend(loc="lower left", fontsize=9)
        endpoint_ax.grid(True, alpha=0.3)

        fig.suptitle(
            "Ball-Ring (Simulated): sampler training budget -- point estimates peak at "
            "10,000, but no pair is separated at n=10"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=150)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    BallRingSamplerIters.render(arms_json=args.arms_json, output=args.output)
    for iters, mean, sd in BallRingSamplerIters.summary_table(
        arms=BallRingSamplerIters.load_arms(json_path=args.arms_json)
    ):
        print(f"{iters:>7,}  {mean:5.1f}  +- {sd:4.1f}")


if __name__ == "__main__":
    main()
