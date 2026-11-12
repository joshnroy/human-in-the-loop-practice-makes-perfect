"""Summarise two-route Serving runs: practice choices, plate counts and reward by skill.

Reads `<results-root>/<method>/<seed>/` as `scripts.run_sweep` wrote it. Practice comes
from `step_events.jsonl` (each robot event's `skill_trace.plate_count` is the count
after it); evaluation from each `evaluations/<index>/` (`results.json`, and
`actions.jsonl` for the plate count after a task's bulk skill). A task is solved by
the skill that ended it (Josh, 2026-11-12): a Sweep that left 3 or more on the plate
is solved and worth 3, a GatherAndSweep that left all 5 is solved and worth 5, and
anything else is unsolved and worth 0, a GatherAndSweep that landed 3 or 4 included.
`results_solved` is the run's own `solved`; runs from before that rule (which stopped a
task the moment all five settled, leaving no action event: counted as 5 here) can
disagree with `solved`. Post-run only.
"""

import argparse
import json
from pathlib import Path
from typing import Any

BULK = ("Sweep", "GatherAndSweep")
NEEDED = {"Sweep": 3, "GatherAndSweep": 5}
REWARD = {"Sweep": 3, "GatherAndSweep": 5}


class RackTwoRouteSummary:
    @staticmethod
    def lines(*, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    @staticmethod
    def solved(*, skill: str | None, count: int) -> bool:
        return skill in NEEDED and count >= NEEDED[skill]

    @staticmethod
    def reward(*, skill: str | None, count: int) -> int:
        return (
            REWARD[skill] if skill and RackTwoRouteSummary.solved(skill=skill, count=count) else 0
        )

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
        # The recording names every dispatched skill, the goal-stopped one included.
        dispatched: dict[int, str] = {}
        recording = directory / "recording.jsonl"
        if recording.exists():
            with recording.open() as stream:
                for line in stream:
                    if line.startswith('{"kind": "action"'):
                        record = json.loads(line)
                        if record.get("skill") in BULK:
                            dispatched[record["task_index"]] = record["skill"]
        tasks = []
        for task in results["tasks"]:
            own = [a for a in actions if a["task"] == task["task"]]
            bulk = [a for a in own if RackTwoRouteSummary.skill(label=a["label"]) in BULK]
            count = 5 if task["solved"] else 0
            if bulk:
                count = bulk[-1]["skill_trace"]["plate_count"]
            skill = dispatched.get(task["task"]) or (
                RackTwoRouteSummary.skill(label=bulk[-1]["label"]) if bulk else None
            )
            tasks.append({
                "task": task["task"],
                "bulk_skill": skill,
                "plate_count": count,
                "solved": RackTwoRouteSummary.solved(skill=skill, count=count),
                "results_solved": task["solved"],
                "reward": RackTwoRouteSummary.reward(skill=skill, count=count),
                "reason": task["reason"],
                "errors": [a["skill_error"] for a in own if a.get("skill_error")],
            })
        return {
            "policy": results["deployment_policy"]["kind"],
            "tasks": tasks,
            "num_solved": sum(t["solved"] for t in tasks),
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
