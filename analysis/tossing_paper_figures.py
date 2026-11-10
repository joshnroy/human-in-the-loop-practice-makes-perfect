"""Publication figures from the audited Tossing v13 cached measurements only.

Render PDF/PNG figures and a per-seed table from a local audited analysis cache:

    scripts/with_env.sh python -m analysis.tossing_paper_figures \
        --input-dir /path/to/tossing-v13-progress \
        --output-dir /path/to/paper/figures/tossing-v13

The input directory supplies costs.json and curves50.json. Empirical inputs and
generated figures are supplied locally, not distributed with this script.

No simulator, policy pickle, hypothesis test, or extra evaluation is invoked. The
last-up-to-three snapshot aggregate deliberately retains the historical definition,
including the initial snapshot when cost-10000 runs have only two measurements.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from pydantic import BaseModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COSTS = ("0", "5", "10", "20", "100", "10000")
ARMS = (
    ("expectimax", "Expectimax (ours)", "#0072B2", 10),
    ("curtis", "Curtis (ours)", "#CC79A7", 10),
    ("ees", "EES", "#009E73", 10),
    ("dp", "DP", "#E69F00", 5),
    ("dpperiod", "DP-period", "#F0E442", 5),
)
Metric = Literal["solved", "resets", "total_cost"]


class PaperFigures:
    """Re-render existing measurements without changing their scientific meaning."""

    @staticmethod
    def run(*, input_dir: Path, output_dir: Path) -> None:
        sweep = FrozenSweep.load(input_dir=input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        })
        PaperFigures.learning(sweep=sweep, output_dir=output_dir)
        PaperFigures.tradeoff(sweep=sweep, output_dir=output_dir)
        PaperFigures.table(sweep=sweep, output_dir=output_dir)
        manifest = {
            "scope": "Replot audited cached measurements; no new experiments or tests.",
            "inputs": {
                name: hashlib.sha256((input_dir / name).read_bytes()).hexdigest()
                for name in ("costs.json", "curves50.json")
            },
            "curve_points": sum(map(len, sweep.curves.values())),
            "run_summaries": len(sweep.costs),
            "uncertainty": "one sample standard error across independent learning seeds",
            "solved_summary": "mean of last up to three snapshots, initial included",
            "stopped_curves": "end at last actual measurement; no extrapolation",
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @staticmethod
    def learning(*, sweep: "FrozenSweep", output_dir: Path) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.55), sharex=True, sharey=True)
        for index, (ax, cost) in enumerate(zip(axes.flat, COSTS, strict=True)):
            for arm, label, color, n in ARMS[:3]:
                for seed in range(n):
                    points = sweep.curves[f"{cost}/{arm}/{seed}"]
                    ax.plot(
                        [min(i * 3200, 200000) / 1000 for i, _ in points],
                        [value for _, value in points],
                        color=color,
                        alpha=0.16,
                        lw=0.45,
                    )
                steps, rows = sweep.shared_curve(cost=cost, arm=arm)
                mean, sem = PaperFigures.mean_sem(rows=rows)
                x = steps / 1000
                ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
                ax.plot(
                    x,
                    mean,
                    color=color,
                    lw=1.45,
                    marker="o" if len(x) == 2 else None,
                    ms=2.8,
                    label=f"{label}, n={n}",
                )
            ax.set_title(f"({chr(97 + index)}) Reset cost c = {int(cost):,}")
            ax.set_xlim(0, 200)
            ax.set_ylim(0, 52)
            ax.set_xticks([0, 100, 200])
            ax.set_yticks([0, 25, 50])
            ax.grid(alpha=0.18, lw=0.5)
            if index % 3 == 0:
                ax.set_ylabel("Tasks solved")
            if index >= 3:
                ax.set_xlabel("Practice steps (thousands)")
        axes[1, 2].text(
            0.98,
            0.37,
            "Ours: practice stopped\nafter the first session",
            transform=axes[1, 2].transAxes,
            ha="right",
            fontsize=7,
        )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(left=0.065, right=0.975, bottom=0.12, top=0.855, hspace=0.5, wspace=0.2)
        PaperFigures.save(fig=fig, output_dir=output_dir, stem="tossing-learning")

    @staticmethod
    def tradeoff(*, sweep: "FrozenSweep", output_dir: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
        metrics: tuple[Metric, ...] = ("solved", "resets", "total_cost")
        titles = ("(a) Tasks solved (of 50)", "(b) Human resets", "(c) Total practice cost")
        for ax, metric, title in zip(axes, metrics, titles, strict=True):
            selected = ARMS[:3] if metric == "solved" else ARMS
            for j, (arm, label, color, n) in enumerate(selected):
                x = np.arange(len(COSTS)) + (j - (len(selected) - 1) / 2) * 0.075
                cells = [sweep.summary(cost=c, arm=arm, metric=metric) for c in COSTS]
                rows = np.stack(cells, axis=1)
                for seed in range(n):
                    ax.scatter(
                        x + np.linspace(-0.035, 0.035, n)[seed],
                        rows[seed],
                        color=color,
                        s=8,
                        alpha=0.32,
                        edgecolors="none",
                    )
                mean, sem = PaperFigures.mean_sem(rows=rows)
                ax.errorbar(
                    x,
                    mean,
                    yerr=sem,
                    color=color,
                    lw=1.3,
                    marker="o",
                    ms=2.4,
                    capsize=1.5,
                    elinewidth=0.8,
                    label=f"{label}, n={n}",
                )
            ax.set_title(title)
            ax.set_xticks(range(len(COSTS)), ["0", "5", "10", "20", "100", "10,000"])
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_xlabel("Human-reset cost c")
            ax.grid(alpha=0.18, lw=0.5)
        axes[0].set_ylim(0, 52)
        axes[0].set_yticks([0, 25, 50])
        axes[1].set_ylim(bottom=0)
        axes[2].set_yscale("log")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.subplots_adjust(left=0.055, right=0.975, bottom=0.25, top=0.7, wspace=0.32)
        PaperFigures.save(fig=fig, output_dir=output_dir, stem="tossing-tradeoff")

    @staticmethod
    def mean_sem(*, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return rows.mean(axis=0), rows.std(axis=0, ddof=1) / np.sqrt(len(rows))

    @staticmethod
    def save(*, fig: plt.Figure, output_dir: Path, stem: str) -> None:
        fig.savefig(output_dir / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(output_dir / f"{stem}.png", dpi=220)
        plt.close(fig)

    @staticmethod
    def table(*, sweep: "FrozenSweep", output_dir: Path) -> None:
        with (output_dir / "per-seed.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "reset_cost",
                "method",
                "seed",
                "solved_out_of_50",
                "human_resets",
                "total_cost",
                "measured_practice_steps",
            ])
            for cost in COSTS:
                for arm, _, _, n in ARMS:
                    resets = sweep.summary(cost=cost, arm=arm, metric="resets")
                    total = sweep.summary(cost=cost, arm=arm, metric="total_cost")
                    solved = (
                        sweep.summary(cost=cost, arm=arm, metric="solved")
                        if arm in {"expectimax", "curtis", "ees"}
                        else [""] * n
                    )
                    for seed in range(n):
                        writer.writerow([
                            cost,
                            arm,
                            seed,
                            solved[seed],
                            resets[seed],
                            total[seed],
                            sweep.costs[f"{cost}/{arm}/{seed}"]["practice_steps"],
                        ])


class FrozenSweep(BaseModel):
    """Audited cached inputs, with explicit coverage and missing-score checks."""

    curves: dict[str, list[tuple[int, int]]]
    costs: dict[str, dict[str, float]]

    @classmethod
    def load(cls, *, input_dir: Path) -> "FrozenSweep":
        sweep = cls(
            curves=json.loads((input_dir / "curves50.json").read_text()),
            costs=json.loads((input_dir / "costs.json").read_text()),
        )
        expected_costs = {
            f"{c}/{arm}/{seed}" for c in COSTS for arm, _, _, n in ARMS for seed in range(n)
        }
        expected_curves = {
            f"{c}/{arm}/{seed}" for c in COSTS for arm, _, _, n in ARMS[:3] for seed in range(n)
        }
        if set(sweep.costs) != expected_costs or set(sweep.curves) != expected_curves:
            raise ValueError("Expected 240 run summaries and 180 re-scored learning curves")
        for points in sweep.curves.values():
            indices = [i for i, _ in points]
            if not indices or indices[0] != 0 or indices != sorted(set(indices)):
                raise ValueError("Snapshot indices must be unique, sorted, and start at zero")
            if any(not 0 <= score <= 50 or not 0 <= i <= 63 for i, score in points):
                raise ValueError("Invalid snapshot index or solved count")
        return sweep

    def shared_curve(self, *, cost: str, arm: str) -> tuple[np.ndarray, np.ndarray]:
        runs = [dict(self.curves[f"{cost}/{arm}/{seed}"]) for seed in range(10)]
        common = sorted(set.intersection(*(set(run) for run in runs)))
        steps = np.array([min(i * 3200, 200000) for i in common])
        rows = np.array([[run[i] for i in common] for run in runs])
        return steps, rows

    def summary(self, *, cost: str, arm: str, metric: Metric) -> np.ndarray:
        n = 5 if arm in {"dp", "dpperiod"} else 10
        values = []
        for seed in range(n):
            key = f"{cost}/{arm}/{seed}"
            if metric == "solved":
                if key not in self.curves:
                    raise ValueError(f"{arm} was not re-scored on 50 tasks")
                values.append(float(np.mean([score for _, score in self.curves[key][-3:]])))
            elif metric == "resets":
                values.append(self.costs[key]["human_invocations"])
            else:
                measured = self.costs[key]
                values.append(measured["robot_cost"] + measured["human_cost"])
        return np.array(values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    PaperFigures.run(input_dir=args.input_dir, output_dir=args.output_dir)
