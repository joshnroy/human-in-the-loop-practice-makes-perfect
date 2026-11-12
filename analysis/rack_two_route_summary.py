"""Summarise two-route Serving runs: practice choices, plate counts and reward by count.

Reads `<results-root>/<method>/<seed>/` as `scripts.run_sweep` wrote it. Practice comes
from `step_events.jsonl` (each robot event's `skill_trace.plate_count` is the count
after it); evaluation from each `evaluations/<index>/` (`results.json`, and
`actions.jsonl` for the plate count after a task's bulk skill). A task the evaluation
clock stopped at its goal (all five settled) has no action event and counts as 5.
Reward by count: 5 on the plate is 5, 3 or 4 is 3, fewer is 0. Post-run only.
"""

import argparse
import json
from pathlib import Path
from typing import Any

BULK = ("Sweep", "GatherAndSweep")
NEEDED = {"Sweep": 3, "GatherAndSweep": 5}


class RackTwoRouteSummary:
    @staticmethod
    def lines(*, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    @staticmethod
    def reward(*, count: int) -> int:
        return 5 if count >= 5 else 3 if count >= 3 else 0

    @staticmethod
    def skill(*, label: str) -> str:
        return label.split("(")[0]

    @staticmethod
    def practice(*, run: Path) -> dict[str, Any]:
        events = RackTwoRouteSummary.lines(path=run / "step_events.jsonl")
        robot = [e for e in events if e.get("event") == "robot"]
        human = [e for e in events if e.get("event") == "human"]
        bulk: dict[str, Any] = {}
        for name in BULK:
            # An attempt the practice budget cut off mid-skill is not an outcome.
            tries = [
                e
                for e in robot
                if RackTwoRouteSummary.skill(label=e["label"]) == name and not e.get("interrupted")
            ]
            counts = [(e.get("skill_trace") or {}).get("plate_count") for e in tries]
            bulk[name] = {
                "attempts": len(tries),
                "successes": sum(c is not None and c >= NEEDED[name] for c in counts),
                "plate_counts": counts,
            }
        resets: dict[str, int] = {}
        for event in human:
            resets[event["destination"]] = resets.get(event["destination"], 0) + 1
        others: dict[str, int] = {}
        for event in robot:
            name = RackTwoRouteSummary.skill(label=event["label"])
            if name not in BULK:
                others[name] = others.get(name, 0) + 1
        ends = [e for e in events if e.get("event") == "practice_end"]
        return {
            "bulk": bulk,
            "other_robot_skills": others,
            "human_resets": resets,
            "skill_errors": [
                {"label": e["label"], "error": e["skill_error"]}
                for e in robot
                if e.get("skill_error")
            ],
            "interrupted": sum(bool(e.get("interrupted")) for e in robot),
            "practice_end": ends[-1] if ends else None,
        }

    @staticmethod
    def evaluation(*, directory: Path) -> dict[str, Any]:
        results = json.loads((directory / "results.json").read_text())
        actions = RackTwoRouteSummary.lines(path=directory / "actions.jsonl")
        tasks = []
        for task in results["tasks"]:
            own = [a for a in actions if a["task"] == task["task"]]
            bulk = [a for a in own if RackTwoRouteSummary.skill(label=a["label"]) in BULK]
            count = 5 if task["solved"] else 0
            if bulk and not task["solved"]:
                count = bulk[-1]["skill_trace"]["plate_count"]
            tasks.append({
                "task": task["task"],
                "bulk_skill": RackTwoRouteSummary.skill(label=bulk[-1]["label"]) if bulk else None,
                "plate_count": count,
                "reward": RackTwoRouteSummary.reward(count=count),
                "reason": task["reason"],
                "errors": [a["skill_error"] for a in own if a.get("skill_error")],
            })
        return {
            "policy": results["deployment_policy"]["kind"],
            "tasks": tasks,
            "total_reward": sum(t["reward"] for t in tasks),
            "max_reward": 5 * len(tasks),
        }

    @staticmethod
    def run(*, run: Path) -> dict[str, Any]:
        evaluations = sorted(
            d for d in (run / "evaluations").glob("*") if (d / "results.json").exists()
        )
        return {
            "practice": RackTwoRouteSummary.practice(run=run),
            "evaluations": {
                d.name: RackTwoRouteSummary.evaluation(directory=d) for d in evaluations
            },
        }

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--results-root", type=Path, required=True)
        args = parser.parse_args()
        summary = {
            f"{run.parent.name}/{run.name}": RackTwoRouteSummary.run(run=run)
            for run in sorted(args.results_root.glob("*/*"))
            if (run / "step_events.jsonl").exists()
        }
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    RackTwoRouteSummary.main()
