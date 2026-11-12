"""Two Routes (`--env tworoutes`): EES, DP and ours over human cost, budget and Route B's rate.

Two subcommands, both post-run. `extract` reads the launcher's tree
(`scripts/tworoutes/launch_sweeps.sh`: `<root>/<sweep>/<cell>/<method>/<seed>/`) into one
row per run. `report` turns the rows into the per-cell table, the paired tests and one
figure per sweep.

    scripts/with_env.sh python -m analysis.tworoutes_three_methods extract \
        --root <results root> --output rows.json
    scripts/with_env.sh python -m analysis.tworoutes_three_methods report \
        --rows rows.json --output-dir docs/experiment-logs --slug 2026-11-12-tworoutes

A run's reward is read off its final evaluation: each held-out task is one attempt of
one route, worth that route's reward if it succeeded and nothing if not.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUTE_A = "RouteA"
ROUTE_B = "RouteB"
# Okabe-Ito, fixed project-wide. The two blues are ours at two values of lambda: the
# larger lambda is the darker shade.
ARM_STYLES: dict[str, tuple[str, str]] = {
    "ees": ("EES", "#009E73"),
    "dp": ("DP", "#CC79A7"),
    "pomdp": ("ours (Model B)", "#00466E"),
    "pomdp-small-lambda": ("ours (Model B)", "#0072B2"),
}
ARM_ORDER = ("ees", "dp", "pomdp", "pomdp-small-lambda")
SWEEP_AXES = {
    "cost": ("human reset cost c", True),
    "budget": ("practice budget (cost units)", True),
    "rate": ("Route B gain per practice attempt", False),
}
PANELS = (
    ("reward", "reward at evaluation", "Reward (Route A pays 1, Route B pays 3)"),
    ("attempts_b", "practice attempts", "Route B practice"),
    ("attempts_a", "practice attempts", "Route A practice"),
    ("human_resets", "human resets", "Human resets asked for"),
)
PAIRS = (("pomdp", "dp"), ("pomdp", "ees"), ("dp", "ees"), ("pomdp", "pomdp-small-lambda"))
TEST_METRICS = ("reward", "expected_reward", "practice_cost")


class TwoRoutesExtract:
    """Run trees -> one row per (sweep, cell, arm, seed)."""

    @staticmethod
    def extract_root(*, root: Path) -> list[dict[str, Any]]:
        rows = []
        for run_dir in sorted(Path(root).glob("*/*/*/*")):
            if not run_dir.is_dir() or not run_dir.name.isdigit():
                continue
            cell, method = run_dir.parents[1].name, run_dir.parent.name
            sweep = run_dir.parents[2].name
            small_lambda = sweep.startswith("lambda")
            rows.append({
                # The smaller-lambda runs of ours are a second arm of the cost sweep.
                "sweep": "cost" if small_lambda else sweep,
                "cell": cell,
                "x": float(cell[1:]),
                "arm": f"{method}-small-lambda" if small_lambda else method,
                "seed": int(run_dir.name),
                **TwoRoutesExtract.extract_run(run_dir=run_dir),
            })
        return sorted(rows, key=lambda r: (r["sweep"], r["x"], r["arm"], r["seed"]))

    @staticmethod
    def lines(*, path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def extract_run(*, run_dir: Path) -> dict[str, Any]:
        evaluations_path = run_dir / "step_evaluations.json"
        if not evaluations_path.exists():
            return {"complete": False}
        final = json.loads(evaluations_path.read_text())[-1]
        if not final.get("complete"):
            return {"complete": False}
        args = json.loads((run_dir / "config_snapshot.json").read_text())["args"]
        costs = json.loads((run_dir / "practice_costs.json").read_text())
        rewards = {ROUTE_A: float(args["route_a_reward"]), ROUTE_B: float(args["route_b_reward"])}
        attempts_log = run_dir / "tworoutes_attempts.jsonl"
        attempts = {ROUTE_A: 0, ROUTE_B: 0}
        if attempts_log.exists() and attempts_log.read_text().strip():
            attempts = TwoRoutesExtract.lines(path=attempts_log)[-1]["attempts"]
        cap = float(args["competence_cap"])
        p = {
            ROUTE_A: min(
                cap,
                float(args["route_a_start"]) + float(args["route_a_rate"]) * attempts[ROUTE_A],
            ),
            ROUTE_B: min(
                cap,
                float(args["route_b_start"]) + float(args["route_b_rate"]) * attempts[ROUTE_B],
            ),
        }
        solved = {task["task"]: bool(task["solved"]) for task in final["tasks"]}
        # A task's route is its first action: an evaluation is one attempt.
        routes: dict[int, str] = {}
        actions_path = run_dir / "evaluations" / f"{final['index']:04d}" / "actions.jsonl"
        if actions_path.exists():
            for action in TwoRoutesExtract.lines(path=actions_path):
                routes.setdefault(action["task"], action["label"].split("(")[0])
        deployed = {name: sum(route == name for route in routes.values()) for name in rewards}
        num_tasks = len(solved)
        reward_sum = sum(rewards[routes[task]] for task, won in solved.items() if won)
        ends = [
            e
            for e in TwoRoutesExtract.lines(path=run_dir / "step_events.jsonl")
            if e.get("event") == "practice_end"
        ]
        status = json.loads((run_dir / "status.json").read_text())
        return {
            "complete": True,
            "human_cost": float(costs["human_skills"]["human_reset_b"]["cost"]["value"]),
            "objective_lambda": float(costs["objective_lambda"]),
            "budget": int(args["practice_step_budget"]),
            "route_b_rate": float(args["route_b_rate"]),
            "num_tasks": num_tasks,
            "num_solved": sum(solved.values()),
            "reward_sum": reward_sum,
            "reward": reward_sum / num_tasks,
            "deployed_a": deployed[ROUTE_A],
            "deployed_b": deployed[ROUTE_B],
            "p_a": p[ROUTE_A],
            "p_b": p[ROUTE_B],
            "expected_reward": sum(rewards[name] * p[name] * deployed[name] for name in rewards)
            / num_tasks,
            "best_expected_reward": max(rewards[name] * p[name] for name in rewards),
            "attempts_a": attempts[ROUTE_A],
            "attempts_b": attempts[ROUTE_B],
            "human_resets": int(final["human_invocations"]),
            "practice_cost": float(final["physical_cost"]),
            "practice_steps": int(status["practice_steps"]),
            "robot_steps": int(status["robot_steps"]),
            "practice_end": ends[-1]["reason"] if ends else None,
            "deployment_policy": final["deployment_policy"]["kind"],
        }


class TwoRoutesReport:
    """Rows -> per-cell summary, paired tests, figures."""

    @staticmethod
    def complete(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if row.get("complete")]

    @staticmethod
    def cells(*, rows: list[dict[str, Any]]) -> list[tuple[str, float, str]]:
        return sorted({(r["sweep"], r["x"], r["cell"]) for r in rows})

    @staticmethod
    def summarize(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = TwoRoutesReport.complete(rows=rows)
        summary = []
        for sweep, x, cell in TwoRoutesReport.cells(rows=rows):
            for arm in ARM_ORDER:
                group = [r for r in rows if (r["sweep"], r["cell"], r["arm"]) == (sweep, cell, arm)]
                if not group:
                    continue
                summary.append({
                    "sweep": sweep,
                    "cell": cell,
                    "x": x,
                    "arm": arm,
                    "runs": len(group),
                    **{
                        metric: float(np.mean([r[metric] for r in group]))
                        for metric in (
                            "reward",
                            "expected_reward",
                            "best_expected_reward",
                            "attempts_a",
                            "attempts_b",
                            "human_resets",
                            "practice_cost",
                        )
                    },
                    "solved": f"{sum(r['num_solved'] for r in group)}/"
                    f"{sum(r['num_tasks'] for r in group)}",
                    # A run deploys one route on every task; counted by its majority.
                    "runs_deploying_b": f"{sum(r['deployed_b'] > r['deployed_a'] for r in group)}/"
                    f"{len(group)}",
                })
        return summary

    @staticmethod
    def paired_test(
        *, rows: list[dict[str, Any]], first: str, second: str, metric: str
    ) -> dict[str, Any]:
        """Wilcoxon signed-rank on `first - second` over the seeds both arms ran.
        `p_value` is None when every difference is zero: nothing to test."""
        rows = TwoRoutesReport.complete(rows=rows)
        by_arm = {
            arm: {r["seed"]: float(r[metric]) for r in rows if r["arm"] == arm}
            for arm in (first, second)
        }
        seeds = sorted(set(by_arm[first]) & set(by_arm[second]))
        differences = np.array([by_arm[first][s] - by_arm[second][s] for s in seeds])
        nonzero = differences[differences != 0]
        return {
            "first": first,
            "second": second,
            "metric": metric,
            "n": len(seeds),
            "mean_difference": float(differences.mean()) if len(seeds) else None,
            "wins": int((differences > 0).sum()),
            "losses": int((differences < 0).sum()),
            "p_value": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else None,
        }

    @staticmethod
    def tests(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = TwoRoutesReport.complete(rows=rows)
        results = []
        for sweep, x, cell in TwoRoutesReport.cells(rows=rows):
            group = [r for r in rows if (r["sweep"], r["cell"]) == (sweep, cell)]
            arms = {r["arm"] for r in group}
            for first, second in PAIRS:
                if first not in arms or second not in arms:
                    continue
                for metric in TEST_METRICS:
                    results.append({
                        "sweep": sweep,
                        "cell": cell,
                        "x": x,
                        **TwoRoutesReport.paired_test(
                            rows=group, first=first, second=second, metric=metric
                        ),
                    })
        return results

    @staticmethod
    def arm_label(*, arm: str, group: list[dict[str, Any]], seeds: int) -> str:
        label = ARM_STYLES[arm][0]
        lambdas = sorted({r["objective_lambda"] for r in group if "objective_lambda" in r})
        if arm.startswith("pomdp") and len(lambdas) == 1:
            label += f", λ = {lambdas[0]:g}"
        return f"{label} — mean, n={seeds}"

    @staticmethod
    def figures(*, rows: list[dict[str, Any]], output_dir: Path, slug: str) -> list[Path]:
        rows = TwoRoutesReport.complete(rows=rows)
        paths = []
        for sweep in sorted({r["sweep"] for r in rows}):
            group = [r for r in rows if r["sweep"] == sweep]
            axis_label, log_x = SWEEP_AXES.get(sweep, (sweep, False))
            figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
            for axis, (metric, y_label, title) in zip(axes.flat, PANELS, strict=True):
                for arm in ARM_ORDER:
                    arm_rows = [r for r in group if r["arm"] == arm]
                    if not arm_rows:
                        continue
                    color = ARM_STYLES[arm][1]
                    seeds = sorted({r["seed"] for r in arm_rows})
                    # Faint per-seed traces first, under the bold mean.
                    for seed in seeds:
                        trace = sorted((r["x"], r[metric]) for r in arm_rows if r["seed"] == seed)
                        axis.plot(*zip(*trace, strict=True), color=color, alpha=0.16, lw=0.8)
                    xs = sorted({r["x"] for r in arm_rows})
                    means = [np.mean([r[metric] for r in arm_rows if r["x"] == x]) for x in xs]
                    axis.plot(
                        xs,
                        means,
                        color=color,
                        lw=2.3,
                        marker="o",
                        ms=4,
                        label=TwoRoutesReport.arm_label(arm=arm, group=arm_rows, seeds=len(seeds)),
                    )
                if log_x:
                    axis.set_xscale("log")
                axis.set_xticks(sorted({r["x"] for r in group}))
                axis.set_xticklabels([f"{x:g}" for x in sorted({r["x"] for r in group})])
                axis.minorticks_off()
                axis.set_title(title, fontsize=10)
                axis.set_ylabel(y_label)
                axis.grid(alpha=0.25)
            for axis in axes[1]:
                axis.set_xlabel(axis_label)
            axes[0][0].legend(fontsize=8, loc="best")
            fixed = TwoRoutesReport.fixed_settings(sweep=sweep, group=group)
            figure.suptitle(f"Two Routes (toy environment): {axis_label} sweep{fixed}", fontsize=12)
            figure.tight_layout()
            path = Path(output_dir) / f"{slug}-{sweep}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            paths.append(path)
        return paths

    @staticmethod
    def fixed_settings(*, sweep: str, group: list[dict[str, Any]]) -> str:
        """What the sweep holds fixed, read off the rows that carry it."""
        parts = []
        for key, name in (("budget", "budget"), ("human_cost", "c"), ("route_b_rate", "B rate")):
            values = sorted({r[key] for r in group if key in r})
            if len(values) == 1 and (sweep, key) not in {
                ("budget", "budget"),
                ("cost", "human_cost"),
            }:
                parts.append(f"{name} = {values[0]:g}")
        return f" ({', '.join(parts)})" if parts else ""

    @staticmethod
    def markdown(*, summary: list[dict[str, Any]]) -> str:
        lines = [
            "| sweep | cell | arm | runs | reward (mean) | expected reward | best achievable | "
            "tasks solved | runs deploying B | A attempts | B attempts | human resets | "
            "practice cost |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in summary:
            lines.append(
                f"| {row['sweep']} | {row['cell']} | {row['arm']} | {row['runs']} | "
                f"{row['reward']:.2f} | {row['expected_reward']:.2f} | "
                f"{row['best_expected_reward']:.2f} | {row['solved']} | "
                f"{row['runs_deploying_b']} | {row['attempts_a']:.1f} | {row['attempts_b']:.1f} | "
                f"{row['human_resets']:.1f} | {row['practice_cost']:.0f} |"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def tests_markdown(*, tests: list[dict[str, Any]]) -> str:
        lines = [
            "| sweep | cell | pair | metric | n | mean difference | wins / losses | p |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for test in tests:
            p = (
                "none: all differences zero"
                if test["p_value"] is None
                else f"{test['p_value']:.4f}"
            )
            lines.append(
                f"| {test['sweep']} | {test['cell']} | {test['first']} − {test['second']} | "
                f"{test['metric']} | {test['n']} | {test['mean_difference']:+.2f} | "
                f"{test['wins']}/{test['n']} / {test['losses']}/{test['n']} | {p} |"
            )
        return "\n".join(lines) + "\n"


class TwoRoutesCli:
    @staticmethod
    def main(*, argv: list[str] | None = None) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest="command", required=True)
        extract = commands.add_parser("extract")
        extract.add_argument("--root", type=Path, required=True)
        extract.add_argument("--output", type=Path, required=True)
        report = commands.add_parser("report")
        report.add_argument("--rows", type=Path, required=True)
        report.add_argument("--output-dir", type=Path, required=True)
        report.add_argument("--slug", required=True)
        args = parser.parse_args(argv)
        if args.command == "extract":
            rows = TwoRoutesExtract.extract_root(root=args.root)
            args.output.write_text(json.dumps(rows, indent=1) + "\n")
            incomplete = sum(not r["complete"] for r in rows)
            print(f"{len(rows)} runs, {incomplete} incomplete -> {args.output}")
            return
        rows = json.loads(args.rows.read_text())
        summary = TwoRoutesReport.summarize(rows=rows)
        tests = TwoRoutesReport.tests(rows=rows)
        output_dir = args.output_dir
        (output_dir / f"{args.slug}-summary.md").write_text(
            TwoRoutesReport.markdown(summary=summary)
        )
        (output_dir / f"{args.slug}-tests.md").write_text(
            TwoRoutesReport.tests_markdown(tests=tests)
        )
        for path in TwoRoutesReport.figures(rows=rows, output_dir=output_dir, slug=args.slug):
            print(path)


if __name__ == "__main__":
    TwoRoutesCli.main()
