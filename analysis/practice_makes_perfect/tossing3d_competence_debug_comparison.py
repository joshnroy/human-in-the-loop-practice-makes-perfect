"""Compare preserved pilot and debugged run summaries without driving simulations."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


class DebugComparison:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--original", type=Path, required=True)
        parser.add_argument("--debugged", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
        original = json.loads(args.original.read_text())
        debugged = json.loads(args.debugged.read_text())
        assert original["valid"] and debugged["valid"]
        assert original["num_cycles"] == debugged["num_cycles"] == 10
        before = {run["name"]: run for run in original["runs"]}
        after = {run["name"]: run for run in debugged["runs"]}
        assert before.keys() == after.keys()
        # The requested pilot uses one paired seed; do not silently present a
        # multi-seed input as if these single-run traces were aggregated means.
        assert len(before) == 4 and all(name.endswith("/seed_00") for name in before)
        fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, squeeze=False)
        comparisons = []
        blue = "#0072B2"
        for row, name in enumerate(sorted(before)):
            title = (
                name
                .removesuffix("/seed_00")
                .replace("global_curve", "A")
                .replace("local_trend", "B")
            )
            comparison = {"arm": name}
            for label, run, linestyle in (
                ("Original", before[name], (0, (4, 2))),
                ("Debugged", after[name], "solid"),
            ):
                evaluation = [entry["solved"] for entry in run["evaluations"]]
                assert len(evaluation) == 11
                assert all(entry["total"] == 10 for entry in run["evaluations"])
                cost = [0.0, *[entry["cumulative_dispatched_cost"] for entry in run["cycles"]]]
                tosses = [0]
                for cycle in run["cycles"]:
                    tosses.append(tosses[-1] + cycle["sampler_attempts"])
                for column, values in enumerate((evaluation, cost, tosses)):
                    axes[row, column].plot(
                        range(11),
                        values,
                        color=blue,
                        alpha=0.16,
                        linewidth=0.8,
                        linestyle=linestyle,
                    )
                    axes[row, column].plot(
                        range(11),
                        values,
                        color=blue,
                        linewidth=2.3,
                        linestyle=linestyle,
                        label=f"{label} — mean, n=1",
                    )
                resets = sum(
                    entry["count"]
                    for skill, entry in run["dispatch_by_skill"].items()
                    if "reset" in skill
                )
                comparison[label.lower()] = {
                    "final_evaluation_solved": evaluation[-1],
                    "evaluation_tasks": 10,
                    "total_actions": run["total_dispatches"],
                    "total_resets": resets,
                    "total_tosses": tosses[-1],
                    "total_cost": cost[-1],
                    "successful_practice_tosses": run["sampler_by_skill"][
                        "MoveToTossLocationAndToss"
                    ]["successes"],
                }
            comparisons.append(comparison)
            for column, quantity in enumerate((
                "Evaluation tasks (of 10)",
                "Practice cost",
                "Toss attempts",
            )):
                axis = axes[row, column]
                axis.set_title(f"{title}: {quantity}")
                axis.grid(alpha=0.2)
                axis.set_xlim(0, 10)
                axis.set_xticks(range(0, 11, 2))
                axis.legend(fontsize=8)
            axes[row, 0].set_ylim(-0.2, 10.2)
            axes[row, 0].set_ylabel("Solved per seed")
            axes[row, 1].set_ylabel("Cumulative cost")
            axes[row, 2].set_ylabel("Cumulative attempts")
        for column in (1, 2):
            upper = max(axis.get_ylim()[1] for axis in axes[:, column])
            for axis in axes[:, column]:
                axis.set_ylim(0, upper)
        for axis in axes[-1]:
            axis.set_xlabel("Practice cycle")
        fig.suptitle(
            "Tossing3D before and after debugging · one paired seed · no statistical inference"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_dir / "before_after.png", dpi=170)
        fig.savefig(args.output_dir / "before_after.pdf")
        plt.close(fig)
        (args.output_dir / "before_after.json").write_text(
            json.dumps(
                {
                    "original_source": original["source_commit"],
                    "debugged_source": debugged["source_commit"],
                    "comparisons": comparisons,
                    "interpretation": (
                        "One paired seed; raw pilot results only, no statistical inference."
                    ),
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    DebugComparison.main()
