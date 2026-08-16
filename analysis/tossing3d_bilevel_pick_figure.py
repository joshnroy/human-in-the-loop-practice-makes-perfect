"""Plot the sampled-vs-hardcoded pick_cube standoff comparison. Post-run only."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tossing3d_bilevel_pick_summary import load  # noqa: E402

# Blue is the arm with a search mechanism available; orange the arm with none. Grey is
# the control, which is not the manipulation under test.
SAMPLED = "#0072B2"
HARDCODED = "#D55E00"
CONTROL = "#7F7F7F"

OUTCOME_ORDER = ("scored", "planned_not_scored", "plan_not_found", "crashed")
OUTCOME_MARKER = {
    "scored": "o",
    "planned_not_scored": "^",
    "plan_not_found": "x",
    "crashed": "s",
}


def main() -> None:
    """Draw the three-panel comparison figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    arms = [
        ("zero-parameter (baseline)", "base", HARDCODED),
        ("sampled (distance, rot)", "sampled", SAMPLED),
        ("RNG-shift control", "rngctl", CONTROL),
    ]
    budgets = [5, 25]
    data = {
        (prefix, budget): load(os.path.join(args.results_root, f"{prefix}_s{budget}"))
        for _, prefix, _ in arms
        for budget in budgets
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    _plot_counts(axes[0], arms, budgets, data)
    _plot_seed_grid(axes[1], arms, budgets, data)
    _plot_cost(axes[2], arms, budgets, data)

    fig.suptitle(
        "Tossing3D-o1, seeds 100-139: does letting the refiner sample pick_cube's "
        "standoff beat hardcoding it?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


def _plot_counts(ax, arms, budgets, data) -> None:
    """Scored count per arm, one bar group per sampling budget."""
    width = 0.26
    positions = np.arange(len(budgets))
    for index, (label, prefix, colour) in enumerate(arms):
        heights = [
            sum(1 for r in data[(prefix, b)] if r["outcome"] == "scored")
            for b in budgets
        ]
        offset = (index - (len(arms) - 1) / 2) * width
        bars = ax.bar(
            positions + offset,
            heights,
            width,
            color=colour,
            label=label,
            edgecolor="white",
        )
        for bar, height, budget in zip(bars, heights, budgets):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.4,
                f"{height}/{len(data[(prefix, budget)])}",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels([f"samples_per_step = {b}" for b in budgets])
    ax.set_ylabel("seeds scored")
    ax.set_ylim(0, 44)
    ax.set_title("Seeds that scored (of 40)")
    ax.legend(fontsize=8, loc="lower right")


def _plot_seed_grid(ax, arms, budgets, data) -> None:
    """Per-seed outcome, so the residual seed set is visible rather than asserted."""
    rows = []
    for budget in budgets:
        for label, prefix, colour in arms:
            rows.append((f"{label}\nspt={budget}", data[(prefix, budget)], colour))
    for row_index, (_, records, colour) in enumerate(rows):
        for record in records:
            outcome = record["outcome"]
            ax.scatter(
                record["seed"],
                row_index,
                marker=OUTCOME_MARKER.get(outcome, "s"),
                s=34 if outcome == "scored" else 52,
                color=colour,
                alpha=0.95 if outcome != "scored" else 0.35,
                linewidths=1.6,
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlabel("seed")
    ax.set_title("Per-seed outcome")
    ax.grid(axis="x", alpha=0.2)
    handles = [
        plt.Line2D(
            [],
            [],
            marker=OUTCOME_MARKER[o],
            linestyle="none",
            color="k",
            label=o,
            markerfacecolor="none",
        )
        for o in OUTCOME_ORDER[:3]
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower left")


def _plot_cost(ax, arms, budgets, data) -> None:
    """Per-seed wall clock: faint per-seed points under the bold arm mean."""
    positions = []
    labels = []
    for budget_index, budget in enumerate(budgets):
        for arm_index, (label, prefix, colour) in enumerate(arms):
            slot = budget_index * (len(arms) + 1) + arm_index
            seconds = [r["total_seconds"] for r in data[(prefix, budget)]]
            jitter = np.random.default_rng(0).normal(0, 0.06, len(seconds))
            ax.scatter(
                slot + jitter, seconds, s=12, color=colour, alpha=0.16, linewidths=0
            )
            mean = float(np.mean(seconds))
            ax.hlines(mean, slot - 0.34, slot + 0.34, color=colour, linewidth=2.3)
            ax.text(slot, mean * 1.14, f"{mean:.0f}s", ha="center", fontsize=8)
            positions.append(slot)
            labels.append(f"{label.split(' (')[0]}\nspt={budget}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylabel("wall clock per seed (s, log)")
    ax.set_title("Cost per seed — faint points are seeds, bold bar the mean")
    ax.add_artist(
        mpatches.Patch(color="none", label="")
    )  # keeps layout stable when empty


if __name__ == "__main__":
    main()
