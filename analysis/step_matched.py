"""Post-run or partial-run figures on actual counted practice steps, never filled data."""

import argparse
import json
from pathlib import Path
from typing import Any


class StepMatchedAnalysis:
    COLORS = {"ees": "#009E73", "pddl": "#0072B2", "full-agentic": "#CC79A7"}
    LABELS = {"ees": "EES", "pddl": "PDDL", "full-agentic": "Full-agentic"}

    @staticmethod
    def load_run(*, root: Path, method: str) -> dict[str, Any]:
        status = json.loads((root / "status.json").read_text())
        endpoint = status.get(
            "practice_steps", status.get("control_steps", 0) + status.get("human_requests", 0)
        )
        measurements_path = root / "measurements.json"
        if measurements_path.exists():
            measurements = json.loads(measurements_path.read_text())
        else:
            measurements = [
                json.loads(p.read_text())
                for p in sorted((root / "measurements").glob("*/snapshot.json"))
            ]
        merged = {r["index"]: dict(r) for r in measurements}
        completed_path = root / "step_evaluations.json"
        if completed_path.exists():
            for record in json.loads(completed_path.read_text()):
                merged.setdefault(record["index"], {}).update(record)
        for path in sorted((root / "evaluations").glob("*/results.json")):
            record = json.loads(path.read_text())
            index = int(path.parent.name)
            merged.setdefault(index, {}).update(record)
        evaluations, pending = [], []
        expected = 10
        protocol = root / "step_protocol.json"
        if protocol.exists():
            configuration = json.loads(protocol.read_text())
            expected = configuration.get("num_test_tasks", 10)
        for index, record in sorted(merged.items()):
            if (
                record.get("practice_steps", endpoint) > endpoint
                or record.get("error")
                or record.get("complete") is False
                or record.get("num_total") != expected
                or "practice_steps" not in record
            ):
                pending.append(index)
                continue
            solved, total = record["num_solved"], record["num_total"]
            if not 0 <= solved <= total:
                raise ValueError("Invalid solved/total counts")
            evaluations.append(
                dict(
                    index=index,
                    practice_steps=record["practice_steps"],
                    num_solved=solved,
                    num_total=total,
                    policy_sha256=record.get("policy_sha256", record.get("sha256")),
                )
            )
        resets = [dict(practice_steps=0, robot_side=0, opposite_side=0)]
        event_path = root / "step_events.jsonl"
        if event_path.exists():
            lines = event_path.read_text().splitlines(keepends=True)
            for line in lines:
                if not line.endswith("\n"):
                    continue  # A live writer may not have finished its last record.
                event = json.loads(line)
                if event.get("event") == "human" and event["end_step"] <= endpoint:
                    row = dict(resets[-1], practice_steps=event["end_step"])
                    row[event["destination"]] += 1
                    resets.append(row)
        else:
            for _, record in sorted(merged.items()):
                sides = record.get("human_by_side", record.get("human_resets_by_destination"))
                if sides is not None and record["practice_steps"] <= endpoint:
                    resets.append(dict(practice_steps=record["practice_steps"], **sides))
        sides = status.get("human_resets_by_destination", status.get("human_by_side"))
        if sides is not None and endpoint >= resets[-1]["practice_steps"]:
            resets.append(dict(practice_steps=endpoint, **sides))
        return dict(
            method=method,
            seed=0,
            phase=status["phase"],
            endpoint_steps=endpoint,
            evaluations=evaluations,
            pending_indices=pending,
            human_resets=resets,
            status=status,
        )

    @staticmethod
    def pilot_points(
        *, robot_steps: int, human_requests: int, initial_solved: int, final_solved: int, tasks: int
    ) -> list[dict[str, Any]]:
        return [
            dict(practice_steps=steps, num_solved=solved, num_total=tasks, protocol="old_pilot")
            for steps, solved in ((0, initial_solved), (robot_steps + human_requests, final_solved))
        ]

    @staticmethod
    def plot(*, runs: list[dict[str, Any]], output: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
        for run in runs:
            method = run["method"]
            color = StepMatchedAnalysis.COLORS[method]
            label = StepMatchedAnalysis.LABELS[method] + ", n=1"
            points = run["evaluations"]
            axes[0].plot(
                [p["practice_steps"] for p in points],
                [p["num_solved"] for p in points],
                color=color,
                marker="o",
                markersize=3,
                linewidth=2,
                label=label,
            )
            for axis, side in zip(axes[1:], ("robot_side", "opposite_side"), strict=True):
                axis.step(
                    [p["practice_steps"] for p in run["human_resets"]],
                    [p[side] for p in run["human_resets"]],
                    where="post",
                    color=color,
                    linewidth=2,
                    label=label,
                )
                if run["phase"] in {"failed", "evaluation_failed"}:
                    last = run["human_resets"][-1]
                    axis.scatter(last["practice_steps"], last[side], color=color, marker="x", s=70)
        axes[0].set(title="Held-out tasks (of 10)", ylabel="Solved per seed", ylim=(-0.3, 10.3))
        for axis, title in zip(
            axes[1:], ("Bin reset to robot side", "Bin reset across wall"), strict=True
        ):
            axis.set(title=title, ylabel="Cumulative human resets")
        for axis in axes:
            axis.set_xlabel("Counted practice steps")
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
        phase = "partial" if any(r["phase"] != "complete" for r in runs) else "completed"
        missing = (
            "Full-agentic not included; "
            if all(r["method"] != "full-agentic" for r in runs)
            else ""
        )
        fig.suptitle(
            f"Tossing3D · seed 0 · {phase} step-matched runs\n{missing}pending evaluations omitted",
            fontsize=12,
        )
        fig.savefig(output, dpi=180)
        plt.close(fig)

    @staticmethod
    def plot_calibration(*, evidence: Path, output: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        records = [json.loads(p.read_text()) for p in sorted(evidence.glob("*-seed*.json"))]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for method in ("ees", "pddl"):
            selected = [r for r in records if r["method"] == method]
            sessions = [s["counted_steps"] for r in selected for s in r["sessions"]]
            episodes = [e["ticks"] for r in selected for e in r["evaluation_episodes"]]
            for axis, values in zip(axes, (sessions, episodes), strict=True):
                axis.hist(
                    values,
                    bins=25,
                    histtype="step",
                    linewidth=2,
                    color=StepMatchedAnalysis.COLORS[method],
                    label=f"{StepMatchedAnalysis.LABELS[method]}, n={len(values)}",
                )
        for axis, limit, title in zip(
            axes,
            (1700, 500),
            ("Prior practice-session lengths", "Prior held-out episode lengths"),
            strict=True,
        ):
            axis.axvline(limit, color="#999999", linestyle=":", label=f"Chosen: {limit} steps")
            axis.set(
                title=title,
                xlabel="Controller steps + human duration" if limit == 1700 else "Controller steps",
                ylabel="Recorded intervals",
            )
            axis.legend(fontsize=8)
        fig.suptitle("Tossing3D · calibration from 12 archived runs")
        fig.savefig(output, dpi=180)
        plt.close(fig)

    @staticmethod
    def plot_pilot(*, comparison: dict[str, Any], output: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for position, method in enumerate(("ees", "pddl")):
            selected = [r for r in comparison["prior_runs"] if r["method"] == method]
            for index, run in enumerate(selected):
                offset = (index - 2.5) * 0.065
                for axis, key in zip(axes, ("solved", "human_resets"), strict=True):
                    axis.plot(
                        [position + offset - 0.015, position + offset + 0.015],
                        [run["before"][key], run["after"][key]],
                        marker="o",
                        markersize=4,
                        color=StepMatchedAnalysis.COLORS[method],
                        alpha=0.6,
                    )
        for axis, value in zip(axes, (10, 101), strict=True):
            axis.scatter(
                [2], [value], color=StepMatchedAnalysis.COLORS["full-agentic"], marker="D", s=60
            )
            axis.set_xticks(
                [0, 1, 2],
                ["EES\n6 archived runs", "PDDL\n6 archived runs", "Old full-agentic\n1 pilot"],
            )
            axis.grid(axis="y", alpha=0.2)
        axes[0].set(title="Held-out tasks (of 10)", ylabel="Solved per seed", ylim=(-0.3, 10.5))
        axes[1].set(title="Human requests by this point", ylabel="Cumulative requests")
        fig.suptitle(
            "Tossing3D · exploratory comparison near 33,978 counted steps\n"
            "Each archived run shows bracketing observations; protocols differ",
            fontsize=11,
        )
        fig.savefig(output, dpi=180)
        plt.close(fig)

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--ees", type=Path)
        parser.add_argument("--pddl", type=Path)
        parser.add_argument("--full-agentic", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        runs = [
            StepMatchedAnalysis.load_run(root=path, method=method)
            for method, path in (
                ("ees", args.ees),
                ("pddl", args.pddl),
                ("full-agentic", args.full_agentic),
            )
            if path
        ]
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "curves.json").write_text(json.dumps(runs, indent=2) + "\n")
        StepMatchedAnalysis.plot(runs=runs, output=args.output / "curves.png")


if __name__ == "__main__":
    StepMatchedAnalysis.main()
