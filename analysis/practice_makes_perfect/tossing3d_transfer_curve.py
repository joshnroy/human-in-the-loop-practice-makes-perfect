"""Post-run mean ± sample standard deviation for the standard transfer benchmark.

The unit of replication is a training seed. Each checkpoint tests the same ten
far-side tasks within a seed. Align by cycle: autonomous practice may stall, so
seeds need not share an actual-action grid. Never silently drop missing seeds.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class TransferCurve:
    @staticmethod
    def aggregate(*, evaluations: list[list[list[int]]]) -> dict:
        if len(evaluations) < 2:
            raise ValueError(
                "At least two training seeds are required for sample standard deviation"
            )
        lengths = {len(run) for run in evaluations}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError("Every seed must have the same nonempty checkpoint grid")
        for run in evaluations:
            if any(len(row) != 3 or row[2] != 10 or not 0 <= row[1] <= 10 for row in run):
                raise ValueError("Every checkpoint must evaluate exactly ten tasks")
            if run[0][0] != 0 or any(b[0] < a[0] for a, b in zip(run, run[1:], strict=False)):
                raise ValueError("Action counts must start at zero and never decrease")
        data = np.asarray(evaluations, dtype=float)
        rates = data[:, :, 1] / 10
        actions = data[:, :, 0]
        return {
            "num_seeds": len(evaluations),
            "test_tasks_per_seed": 10,
            "std_ddof": 1,
            "cycle": list(range(data.shape[1])),
            "mean_success": rates.mean(axis=0).tolist(),
            "std_success": rates.std(axis=0, ddof=1).tolist(),
            "mean_actions": actions.mean(axis=0).tolist(),
            "std_actions": actions.std(axis=0, ddof=1).tolist(),
            "per_seed_success": rates.tolist(),
            "per_seed_actions": actions.tolist(),
        }

    @staticmethod
    def load(*, results_root: Path) -> dict:
        evaluations = []
        for seed in range(10):
            folder = results_root / "ees" / str(seed)
            stats = json.loads((folder / "stats.json").read_text())
            config = json.loads((folder / "config_snapshot.json").read_text())["args"]
            required = {
                "seed": str(seed),
                "layout": "same-side",
                "evaluation_layout": "barrier",
                "num_cycles": "100",
                "max_steps_per_interaction": "20",
                "num_test_tasks": "10",
                "sampler_max_train_iters": "10000",
                "practice_reset_policy": "never",
                "goal_pursuit_horizon": "None",
                "ask_for_reset_cube_bin_cost": "None",
            }
            if any(str(config.get(key)) != value for key, value in required.items()):
                raise ValueError(f"Seed {seed} does not match the standard transfer protocol")
            if len(stats["evaluations"]) != 101:
                raise ValueError(f"Seed {seed} is incomplete: expected 101 checkpoints")
            if stats["num_practice_resets"] or stats["num_human_interventions_recorded"]:
                raise ValueError(f"Seed {seed} contains a practice reset or human intervention")
            evaluations.append(stats["evaluations"])
        return TransferCurve.aggregate(evaluations=evaluations)

    @staticmethod
    def plot(*, summary: dict, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        x = np.asarray(summary["cycle"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        panels = (
            ("success", 100, "Far-side evaluation success (%)"),
            ("actions", 1, "Actual practice actions"),
        )
        for ax, (key, scale, label) in zip(axes, panels, strict=True):
            mean = np.asarray(summary[f"mean_{key}"]) * scale
            std = np.asarray(summary[f"std_{key}"]) * scale
            for row in summary[f"per_seed_{key}"]:
                ax.plot(x, np.asarray(row) * scale, color="#197d78", alpha=0.16, linewidth=0.7)
            ax.plot(x, mean, color="#197d78", label="Mean across 10 seeds", linewidth=2)
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color="#197d78",
                alpha=0.2,
                label="±1 sample SD (not SEM)",
            )
            ax.set_ylabel(label)
            ax.grid(alpha=0.15)
        axes[0].set_ylim(0, 100)
        axes[0].legend(loc="best")
        axes[1].set_xlabel("Training cycle (20-action maximum per cycle)")
        axes[1].set_ylim(bottom=0)
        fig.suptitle(
            "EES: same-side autonomous practice → far-side evaluation\n"
            "10 seeds × 10 fixed test tasks; 100 cycles; no practice resets"
        )
        fig.tight_layout()
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output_dir / f"learning-curve.{extension}", dpi=180)
        plt.close(fig)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        keys = ["cycle", "mean_success", "std_success", "mean_actions", "std_actions"]
        with (output_dir / "learning-curve.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(keys)
            writer.writerows(zip(*(summary[key] for key in keys), strict=True))

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        args = parser.parse_args()
        TransferCurve.plot(
            summary=TransferCurve.load(results_root=args.results_root), output_dir=args.output_dir
        )


if __name__ == "__main__":
    TransferCurve.main()
