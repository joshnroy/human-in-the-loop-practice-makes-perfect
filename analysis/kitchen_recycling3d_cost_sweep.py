"""Tossing env (`kitchen_recycling3d`) v10 + v10b human-reset-cost sweep: extract and report.

Two subcommands. `extract` reads one launch root's `results/reset-cost-<C>/<arm>/<seed>/`
trees and writes a compact JSON; it uses only the standard library, because it runs with
a bare `python3` on the cluster login node, beside the raw data. `report` merges the two
roots' JSON (v10b's planner rows replace v10's start-up failures; EES comes from v10),
then writes the committed data file, the figures and the per-seed tables. Post-run only.

    python3 analysis/kitchen_recycling3d_cost_sweep.py extract --root <v10 root> --output a.json
    scripts/with_env.sh python analysis/kitchen_recycling3d_cost_sweep.py report \
        --first a.json --rerun b.json --output-dir docs/experiment-logs --stamp "..."
"""

import argparse
import glob
import json
import os
from pathlib import Path

COSTS = ("0", "5", "10", "20", "100", "10000")
SEEDS = ("0", "1", "2")
ARMS = (
    ("expectimax", "expectimax (ours)", "#0072B2"),
    ("curtis", "Curtis", "#CC79A7"),
    ("ees", "EES", "#009E73"),
)
SAME_SIDE = "ResetCubeBinSameSide"
DUMP = "PourCubeOutOfBin"
PAIRS = (("expectimax", "curtis"), ("expectimax", "ees"), ("curtis", "ees"))
SLUG = "2026-11-04-tossing-env-v10-reset-cost-sweep"


class SweepExtract:
    """Raw run trees -> one compact record per (cost, arm, seed)."""

    @staticmethod
    def extract_root(*, root: Path) -> dict:
        runs = {}
        for cost in COSTS:
            for arm, _, _ in ARMS:
                for seed in SEEDS:
                    run_dir = Path(root) / "results" / f"reset-cost-{cost}" / arm / seed
                    if (run_dir / "step_events.jsonl").exists():
                        runs[f"{cost}/{arm}/{seed}"] = SweepExtract.extract_run(run_dir=run_dir)
        return runs

    @staticmethod
    def extract_run(*, run_dir: Path) -> dict:
        measured, resets, dumps, stops, last_step = {}, [], [], [], 0
        for line in (run_dir / "step_events.jsonl").open():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:  # a line still being written
                continue
            kind = event.get("event")
            if kind == "measurement":
                measured[event["index"]] = event["practice_steps"]
            elif kind == "human":
                resets.append([event["start_step"], event["skill"], event.get("cost", 0.0)])
            elif kind == "stop":
                stops.append(event["start_step"])
            elif kind == "robot" and event.get("label", "").startswith(DUMP):
                dumps.append(event["start_step"])
            last_step = max(last_step, event.get("end_step", 0))
        evaluations = []
        for path in glob.glob(str(run_dir / "evaluations" / "*" / "results.json")):
            index = int(os.path.basename(os.path.dirname(path)))
            try:
                result = json.loads(Path(path).read_text())
            except json.JSONDecodeError:
                continue
            if result.get("complete") and index in measured:
                evaluations.append([measured[index], result["num_solved"], result["num_total"]])
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        return {
            "evaluations": sorted(evaluations),
            "resets": resets,
            "dumps": dumps,
            "stop_steps": stops,
            "practice_steps": status.get("practice_steps", last_step),
            "phase": status.get("phase"),
        }


class SweepData:
    """Combines the two launch roots into the one 54-run grid."""

    @staticmethod
    def merge(*, first: dict, rerun: dict) -> dict:
        merged = dict(first)
        merged.update({key: run for key, run in rerun.items() if "/ees/" not in key})
        return merged


class RunSummary:
    """Per-run scalars for the tables and the summary figure."""

    @staticmethod
    def stop_streaks(*, stop_steps: list[int]) -> list[int]:
        streaks: list[int] = []
        for index, step in enumerate(stop_steps):
            if index and step == stop_steps[index - 1] + 1:
                streaks[-1] += 1
            else:
                streaks.append(1)
        return streaks

    @staticmethod
    def summarize(*, run: dict, cancelled: bool) -> dict:
        solved = [point[1] for point in run["evaluations"]]
        last_three = solved[-3:]
        streaks = RunSummary.stop_streaks(stop_steps=run["stop_steps"])
        same = sum(reset[1] == SAME_SIDE for reset in run["resets"])
        if cancelled:
            end = "ended by repeated STOP"
        elif run["phase"] == "complete":
            end = "budget"
        else:
            end = "running"
        return {
            "practice_steps": run["practice_steps"],
            "evaluations": len(solved),
            "last_step": run["evaluations"][-1][0] if solved else None,
            "last_solved": solved[-1] if solved else None,
            "num_total": run["evaluations"][-1][2] if solved else None,
            "mean_last_3": sum(last_three) / len(last_three) if last_three else None,
            "resets_same": same,
            "resets_opposite": len(run["resets"]) - same,
            "dumps": len(run["dumps"]),
            "stops": len(run["stop_steps"]),
            "longest_stop_streak": max(streaks, default=0),
            "final_stop_streak": streaks[-1] if streaks else 0,
            "human_cost": sum(reset[2] for reset in run["resets"]),
            "end": end,
        }


class PairedTests:
    """Exact two-sided Wilcoxon signed-rank across the seeds two arms share.

    Enumerated by hand: zero differences are dropped (Wilcoxon's own convention) and ties
    take average ranks. scipy instead falls back to a normal approximation as soon as one
    difference is zero, which at n=3 prints p-values below the exact floor of 0.25.
    """

    @staticmethod
    def wilcoxon(*, first: list[float], second: list[float]) -> dict:
        diffs = [a - b for a, b in zip(first, second, strict=True) if a != b]
        if not diffs:
            return {"n": len(first), "n_nonzero": 0, "p": None}
        order = sorted(range(len(diffs)), key=lambda i: abs(diffs[i]))
        ranks = [0.0] * len(diffs)
        start = 0
        while start < len(order):
            end = start
            while end + 1 < len(order) and abs(diffs[order[end + 1]]) == abs(diffs[order[start]]):
                end += 1
            for position in range(start, end + 1):
                ranks[order[position]] = (start + end) / 2 + 1
            start = end + 1
        centre = sum(ranks) / 2
        observed = abs(sum(r for r, d in zip(ranks, diffs, strict=True) if d > 0) - centre)
        patterns = range(2 ** len(diffs))
        extreme = sum(
            abs(sum(r for i, r in enumerate(ranks) if mask >> i & 1) - centre) >= observed - 1e-9
            for mask in patterns
        )
        return {"n": len(first), "n_nonzero": len(diffs), "p": extreme / len(patterns)}


class Report:
    """Figures, tables and paired tests for the experiment log."""

    @staticmethod
    def keys(*, cost: str, arm: str, runs: dict) -> list[str]:
        return [f"{cost}/{arm}/{seed}" for seed in SEEDS if f"{cost}/{arm}/{seed}" in runs]

    @staticmethod
    def per_cost_figure(*, cost: str, runs: dict, summaries: dict, stamp: str, path: Path) -> None:
        import matplotlib as mpl  # noqa: PLC0415 -- `extract` must run without matplotlib

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        grid = np.arange(0, 65001, 250)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for arm, name, colour in ARMS:
            keys = Report.keys(cost=cost, arm=arm, runs=runs)
            if not keys:
                continue
            by_step = []
            for key in keys:
                points = np.array(runs[key]["evaluations"], float)
                axes[0].plot(points[:, 0] / 1000, points[:, 1], color=colour, alpha=0.16, lw=0.8)
                Report.end_marker(
                    ax=axes[0],
                    x=points[-1, 0] / 1000,
                    y=points[-1, 1],
                    colour=colour,
                    summary=summaries[key],
                )
                by_step.append({int(p[0]): p[1] for p in points})
            shared = sorted(set.intersection(*(set(d) for d in by_step)))
            axes[0].plot(
                np.array(shared) / 1000,
                [np.mean([d[s] for d in by_step]) for s in shared],
                color=colour,
                lw=2.3,
                label=f"{name} — mean, n={len(keys)}",
            )
            for ax, field in ((axes[1], "resets"), (axes[2], "dumps")):
                curves = []
                for key in keys:
                    events = runs[key][field]
                    steps = np.sort([e[0] if field == "resets" else e for e in events]).astype(
                        float
                    )
                    curve = np.searchsorted(steps, grid, side="right").astype(float)
                    curve[grid > runs[key]["practice_steps"]] = np.nan
                    curves.append(curve)
                    ax.plot(grid / 1000, curve, color=colour, alpha=0.16, lw=0.8)
                totals = "/".join(str(len(runs[key][field])) for key in keys)
                ax.plot(
                    grid / 1000,
                    np.mean(curves, axis=0),
                    color=colour,
                    lw=2.3,
                    label=f"{name} — mean, n={len(keys)} (per seed {totals})",
                )
        axes[0].set_title("held-out tasks solved (of 10)", fontsize=10)
        axes[0].set_ylabel("solved per seed")
        axes[0].set_ylim(-0.3, 10.5)
        axes[1].set_title("human resets (same + opposite side)", fontsize=10)
        axes[1].set_ylabel("cumulative human resets per seed")
        axes[2].set_title("dumping (PourCubeOutOfBin)", fontsize=10)
        axes[2].set_ylabel("cumulative dumps per seed")
        for ax in axes:
            ax.set_xlabel("practice steps (thousands)")
            ax.set_xlim(0, 66)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        source = (
            "v10b (Della 16845106) for expectimax/Curtis" if cost in ("100", "10000") else "v10"
        )
        fig.suptitle(
            f"Tossing env, human reset cost {cost} — {source}; EES from v10 (Della "
            f"16840531). Hollow circle: ended by repeated STOP; hollow square: still "
            f"running. Data as of {stamp}",
            fontsize=9.5,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    @staticmethod
    def end_marker(*, ax: object, x: float, y: float, colour: str, summary: dict) -> None:
        if summary["end"] == "budget":
            return
        marker = "o" if summary["end"] == "ended by repeated STOP" else "s"
        ax.scatter([x], [y], s=40, marker=marker, facecolors="none", edgecolors=colour, zorder=5)

    @staticmethod
    def summary_figure(*, runs: dict, summaries: dict, stamp: str, path: Path) -> None:
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        panels = (
            ("solved (of 10), mean of last 3 evaluations", "solved per seed", "mean_last_3", False),
            ("human resets (same + opposite)", "human resets per seed", "resets", False),
            ("dumping (PourCubeOutOfBin)", "dumps per seed", "dumps", False),
            ("STOP decisions", "STOPs per seed (symlog)", "stops", True),
            ("human cost paid", "human cost per seed (symlog)", "human_cost", True),
        )
        xs = np.arange(len(COSTS))
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for offset, (arm, name, colour) in zip((-0.12, 0.0, 0.12), ARMS, strict=True):
            for ax, (_, _, field, _) in zip(axes, panels, strict=True):
                means = []
                for i, cost in enumerate(COSTS):
                    keys = Report.keys(cost=cost, arm=arm, runs=runs)
                    values = []
                    for key in keys:
                        s = summaries[key]
                        value = (
                            s["resets_same"] + s["resets_opposite"]
                            if field == "resets"
                            else s[field]
                        )
                        values.append(value)
                        hollow = s["end"] != "budget"
                        marker = "s" if s["end"] == "running" else "o"
                        ax.scatter(
                            i + offset,
                            value,
                            s=18,
                            marker=marker,
                            edgecolors=colour,
                            facecolors="none" if hollow else colour,
                            alpha=0.7,
                        )
                    means.append(np.mean(values) if values else np.nan)
                ax.plot(
                    xs + offset,
                    means,
                    color=colour,
                    lw=2.3,
                    marker="_",
                    ms=14,
                    label=f"{name} — mean, n=3",
                )
        for ax, (title, ylabel, _, symlog) in zip(axes, panels, strict=True):
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_xticks(xs, COSTS)
            ax.set_xlabel("human reset cost")
            ax.grid(alpha=0.3)
            if symlog:
                ax.set_yscale("symlog", linthresh=1)
                ax.set_ylim(-0.2, ax.get_ylim()[1] * 3)
        axes[0].set_ylim(-0.3, 10.5)
        axes[0].legend(fontsize=8, frameon=False, loc="lower left")
        fig.text(
            0.5,
            0.01,
            "Filled: ran the full 65,000 steps. Hollow circle: cost-10000 "
            "planner runs cancelled mid STOP streak, counted as ended by repeated STOP "
            "(rule applied post hoc; last evaluation kept). Hollow square: still running. "
            "n=3 per point; no inference is supported.",
            ha="center",
            fontsize=8.5,
        )
        fig.suptitle(
            f"Tossing env v10 + v10b — against human reset cost (seeds 0–2). Data as of {stamp}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(path, dpi=110)
        plt.close(fig)

    @staticmethod
    def table(*, runs: dict, summaries: dict) -> str:
        lines = [
            "| cost | arm | seed | end | steps | last solved | mean last 3 | resets same "
            "| resets opp. | dumps | STOPs | longest STOP streak | human cost |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cost in COSTS:
            for arm, name, _ in ARMS:
                for key in Report.keys(cost=cost, arm=arm, runs=runs):
                    s = summaries[key]
                    lines.append(
                        f"| {cost} | {name} | {key[-1]} | {s['end']} | {s['practice_steps']:,} "
                        f"| {s['last_solved']}/{s['num_total']} (at {s['last_step']:,}) "
                        f"| {s['mean_last_3']:.2f}/10 | {s['resets_same']} "
                        f"| {s['resets_opposite']} | {s['dumps']} | {s['stops']} "
                        f"| {s['longest_stop_streak']} | {s['human_cost']:,.0f} |"
                    )
        return "\n".join(lines)

    @staticmethod
    def paired(*, runs: dict, summaries: dict) -> dict:
        out = {}
        for cost in COSTS:
            for first, second in PAIRS:
                a = Report.keys(cost=cost, arm=first, runs=runs)
                b = Report.keys(cost=cost, arm=second, runs=runs)
                if len(a) != len(SEEDS) or len(b) != len(SEEDS):
                    continue
                for field in ("mean_last_3", "human_cost"):
                    out[f"{cost}/{first}-{second}/{field}"] = {
                        first: [summaries[k][field] for k in a],
                        second: [summaries[k][field] for k in b],
                        **PairedTests.wilcoxon(
                            first=[summaries[k][field] for k in a],
                            second=[summaries[k][field] for k in b],
                        ),
                    }
        return out

    @staticmethod
    def run(*, first: Path, rerun: Path, output_dir: Path, stamp: str) -> None:
        runs = SweepData.merge(
            first=json.loads(first.read_text()), rerun=json.loads(rerun.read_text())
        )
        cancelled = {f"10000/{arm}/{seed}" for arm in ("expectimax", "curtis") for seed in SEEDS}
        summaries = {
            key: RunSummary.summarize(run=run, cancelled=key in cancelled)
            for key, run in runs.items()
        }
        data = {
            "data_as_of": stamp,
            "sources": {
                "v10": "Della array 16840531 (EES at every cost; planners at 0, 5, 10, 20)",
                "v10b": "Della array 16845106 (expectimax and Curtis at 100 and 10000)",
            },
            "cancelled_ended_by_repeated_stop": sorted(cancelled),
            "summaries": summaries,
            "paired_wilcoxon": Report.paired(runs=runs, summaries=summaries),
            "runs": runs,
        }
        (output_dir / f"{SLUG}.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        for cost in COSTS:
            Report.per_cost_figure(
                cost=cost,
                runs=runs,
                summaries=summaries,
                stamp=stamp,
                path=output_dir / f"{SLUG}-cost-{cost}.png",
            )
        Report.summary_figure(
            runs=runs, summaries=summaries, stamp=stamp, path=output_dir / f"{SLUG}-summary.png"
        )
        print(Report.table(runs=runs, summaries=summaries))
        for key, test in data["paired_wilcoxon"].items():
            print(key, test)


class Cli:
    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        sub = parser.add_subparsers(dest="command", required=True)
        extract = sub.add_parser("extract")
        extract.add_argument("--root", type=Path, required=True)
        extract.add_argument("--output", type=Path, required=True)
        report = sub.add_parser("report")
        report.add_argument("--first", type=Path, required=True)
        report.add_argument("--rerun", type=Path, required=True)
        report.add_argument("--output-dir", type=Path, required=True)
        report.add_argument("--stamp", required=True)
        args = parser.parse_args()
        if args.command == "extract":
            runs = SweepExtract.extract_root(root=args.root)
            args.output.write_text(json.dumps(runs))
            print(len(runs), "runs")
        else:
            Report.run(
                first=args.first, rerun=args.rerun, output_dir=args.output_dir, stamp=args.stamp
            )


if __name__ == "__main__":
    Cli.main()
