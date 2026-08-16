"""Plot the sampled-vs-hardcoded pick_cube standoff comparison. Post-run only."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tossing3d_bilevel_pick_summary import load  # noqa: E402

# Blue is the arm that has a search mechanism available; orange the arms that have
# none (any fixed standoff, whichever one); grey the control, which is not the
# manipulation under test.
SAMPLED = "#0072B2"
FIXED = "#D55E00"
CONTROL = "#7F7F7F"

OUTCOME_MARKER = {"scored": "o", "planned_not_scored": "^", "plan_not_found": "x"}

# (legend label, directory, colour, hatch). Hatch is the within-colour subgroup: the
# solid orange bar is the standoff the controller actually shipped, hatched orange
# bars are the arbitrary alternatives measured to tell "sampling helps" apart from
# "this particular fixed point is bad".
ARMS_S5 = [
    ("zero-parameter (0.55, 0.00) — as shipped", "base_s5", FIXED, ""),
    ("fixed (0.50, 0.00)", "fixed_d050_r000", FIXED, "//"),
    ("fixed (0.58, +0.20)", "fixed_d058_rp020", FIXED, "\\\\"),
    ("fixed (0.55, -0.35)", "fixed_d055_rm035", FIXED, "xx"),
    ("RNG-shift control", "rngctl_s5", CONTROL, ""),
    ("sampled (distance, rot)", "sampled_s5", SAMPLED, ""),
]
ARMS_S25 = [
    ("zero-parameter (0.55, 0.00) — as shipped", "base_s25", FIXED, ""),
    ("RNG-shift control", "rngctl_s25", CONTROL, ""),
    ("sampled (distance, rot)", "sampled_s25", SAMPLED, ""),
]


def main() -> None:
    """Draw the three-panel comparison figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = {
        directory: load(os.path.join(args.results_root, directory))
        for _, directory, _, _ in ARMS_S5 + ARMS_S25
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    _plot_counts(axes[0], data)
    _plot_seed_grid(axes[1], data)
    _plot_cost(axes[2], data)

    fig.suptitle(
        "Tossing3D-o1, seeds 100-139, max_abstract_plans=1: does letting the refiner "
        "sample pick_cube's standoff beat hardcoding it?",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


def _scored(records) -> int:
    return sum(1 for r in records if r["outcome"] == "scored")


def _plot_counts(ax, data) -> None:
    """Seeds scored per arm, the two sampling budgets side by side."""
    labels, heights, colours, hatches = [], [], [], []
    for group, arms in (("spt=5", ARMS_S5), ("spt=25", ARMS_S25)):
        for label, directory, colour, hatch in arms:
            labels.append(f"{label}\n{group}")
            heights.append(_scored(data[directory]))
            colours.append(colour)
            hatches.append(hatch)
        labels.append("")
        heights.append(0)
        colours.append("none")
        hatches.append("")
    labels.pop()
    heights.pop()
    colours.pop()
    hatches.pop()

    positions = np.arange(len(heights))
    bars = ax.bar(positions, heights, 0.72, color=colours, edgecolor="white")
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for bar, height in zip(bars, heights):
        if height:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height}/40",
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.2, rotation=30, ha="right")
    ax.set_ylabel("seeds scored")
    ax.set_ylim(0, 45)
    ax.set_title("Seeds that scored (of 40)")


def _plot_seed_grid(ax, data) -> None:
    """Per-seed outcome, so the residual seed set is visible rather than asserted."""
    rows = []
    for group, arms in (("spt=5", ARMS_S5), ("spt=25", ARMS_S25)):
        for label, directory, colour, _ in arms:
            rows.append((f"{label.split(' —')[0]}  {group}", data[directory], colour))
    for row_index, (_, records, colour) in enumerate(rows):
        for record in records:
            outcome = record["outcome"]
            ax.scatter(
                record["seed"],
                row_index,
                marker=OUTCOME_MARKER.get(outcome, "s"),
                s=30 if outcome == "scored" else 55,
                color=colour,
                alpha=0.3 if outcome == "scored" else 1.0,
                linewidths=1.7,
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlabel("seed")
    ax.set_title("Per-seed outcome — faint circles scored")
    ax.grid(axis="x", alpha=0.2)
    handles = [
        plt.Line2D(
            [], [], marker=marker, linestyle="none", color="k", label=outcome, ms=7
        )
        for outcome, marker in OUTCOME_MARKER.items()
    ]
    ax.legend(handles=handles, fontsize=7.5, loc="center left")


def _plot_cost(ax, data) -> None:
    """Per-seed wall clock: faint per-seed points under the bold arm mean."""
    rng = np.random.default_rng(0)
    positions, labels = [], []
    slot = 0
    for group, arms in (("spt=5", ARMS_S5), ("spt=25", ARMS_S25)):
        for label, directory, colour, _ in arms:
            seconds = [r["total_seconds"] for r in data[directory]]
            ax.scatter(
                slot + rng.normal(0, 0.07, len(seconds)),
                seconds,
                s=13,
                color=colour,
                alpha=0.16,
                linewidths=0,
            )
            mean = float(np.mean(seconds))
            ax.hlines(mean, slot - 0.35, slot + 0.35, color=colour, linewidth=2.3)
            ax.text(slot, mean * 1.2, f"{mean:.0f}s", ha="center", fontsize=8)
            positions.append(slot)
            labels.append(f"{label.split(' —')[0]}\n{group}")
            slot += 1
        slot += 1
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6.2, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("wall clock per seed (s, log)")
    ax.set_title("Cost per seed — faint points are seeds, bold bar the mean")


if __name__ == "__main__":
    main()
