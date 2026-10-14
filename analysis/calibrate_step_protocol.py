"""Reconstruct counted controller experience from archived poster logs."""

import argparse
import bisect
import collections
import json
import statistics
from pathlib import Path


class PriorCalibration:
    @staticmethod
    def summary(*, values):
        values = sorted(values)
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "total": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": values[0],
            "max": values[-1],
            "p90": values[round(0.90 * (len(values) - 1))],
            "p95": values[round(0.95 * (len(values) - 1))],
            "p99": values[round(0.99 * (len(values) - 1))],
        }

    @staticmethod
    def analyze(*, run: Path, machine: str, method: str, seed: int):
        stats = json.loads((run / "stats.json").read_text())
        sweeps = [json.loads(s) for s in (run / "progress.jsonl").open()]
        blocks = []
        with (run / "tossing3d_state_log.jsonl").open("rb") as stream:
            for line in stream:
                if line.startswith(b'{"kind": "skill"'):
                    row = json.loads(line)
                    blocks.append({
                        "name": row["name"],
                        "params": row["params"],
                        "time": row["elapsed_seconds"],
                        "ticks": 0,
                    })
                elif line.startswith(b'{"kind": "tick"'):
                    if not blocks:
                        raise ValueError("Tick before any skill")
                    blocks[-1]["ticks"] += 1
        times = [b["time"] for b in blocks]
        assert times == sorted(times)
        episodes = collections.defaultdict(list)
        for line in (run / "episode_traces.jsonl").open():
            row = json.loads(line)
            episodes[row["checkpoint"], row["task_index"]].append(row)
        eval_lengths, solved_lengths, episode_rows = [], [], []
        for (checkpoint, task), rows in episodes.items():
            end = bisect.bisect_right(times, rows[0]["elapsed_seconds"])
            chosen = blocks[end - len(rows) : end]
            assert len(chosen) == len(rows)
            for block, row in zip(chosen, rows, strict=True):
                assert "phase" not in block, (method, seed, checkpoint, task)
                expected_name = (
                    "unknown skill id -1"
                    if row["action"][0] == -1
                    else row["action_label"].split("(")[0]
                )
                assert block["name"] == expected_name, (block, row)
                block["phase"] = "evaluation"
                block["checkpoint"] = checkpoint
            ticks = sum(b["ticks"] for b in chosen)
            eval_lengths.append(ticks)
            if rows[0]["solved"]:
                solved_lengths.append(ticks)
            episode_rows.append({
                "checkpoint": checkpoint,
                "task": task,
                "ticks": ticks,
                "skills": len(rows),
                "solved": rows[0]["solved"],
            })
        sweep_times = [s["elapsed_seconds"] for s in sweeps]
        practice = collections.defaultdict(list)
        excluded = []
        for block in blocks:
            if "phase" in block:
                continue
            cycle = bisect.bisect_right(sweep_times, block["time"]) - 1
            if 0 <= cycle < len(sweeps) - 1:
                practice[cycle].append(block)
            else:
                excluded.append(block)
        sessions = []
        skills = collections.defaultdict(list)
        for cycle, session in enumerate(stats["practice_session_ends"]):
            robot = practice[cycle]
            expected = stats["evaluations"][cycle + 1][0] - stats["evaluations"][cycle][0]
            assert len(robot) == expected, (method, seed, cycle, len(robot), expected)
            human = session["actions_executed"] - len(robot)
            assert human >= 0
            for block in robot:
                skills[block["name"]].append(block["ticks"])
            sessions.append({
                "cycle": cycle,
                "robot_skills": len(robot),
                "human": human,
                "robot_steps": sum(b["ticks"] for b in robot),
                "counted_steps": sum(b["ticks"] for b in robot) + human,
            })
        assert sum(s["human"] for s in sessions) == stats["num_human_interventions_recorded"]
        result = {
            "machine": machine,
            "method": method,
            "seed": seed,
            "source": str(run),
            "sessions": sessions,
            "session_summary": PriorCalibration.summary(
                values=[s["counted_steps"] for s in sessions]
            ),
            "practice_skill_lengths": {
                k: PriorCalibration.summary(values=v) for k, v in skills.items()
            },
            "evaluation_summary": PriorCalibration.summary(values=eval_lengths),
            "solved_evaluation_summary": PriorCalibration.summary(values=solved_lengths),
            "evaluation_episodes": episode_rows,
            "excluded_blocks": excluded,
        }
        return result

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--workstation-root", type=Path, required=True)
        parser.add_argument("--della-root", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        args.output.mkdir(parents=True, exist_ok=True)
        records = []
        arms = {
            "ees": "ees-gpoff/results/ees-lamcost5/ees",
            "pddl": "gpoff/results/local_trend-lam3e-6/pomdp",
        }
        for machine, root in (("workstation", args.workstation_root), ("della", args.della_root)):
            for method, arm in arms.items():
                for seed in range(3):
                    result = PriorCalibration.analyze(
                        run=root / arm / str(seed), machine=machine, method=method, seed=seed
                    )
                    records.append(result)
                    (args.output / f"{machine}-{method}-seed{seed}.json").write_text(
                        json.dumps(result, indent=2) + "\n"
                    )
        summary = dict(
            runs=[
                {
                    k: v
                    for k, v in r.items()
                    if k not in {"sessions", "evaluation_episodes", "excluded_blocks"}
                }
                for r in records
            ],
            pooled_sessions=PriorCalibration.summary(
                values=[s["counted_steps"] for r in records for s in r["sessions"]]
            ),
            pooled_evaluation=PriorCalibration.summary(
                values=[e["ticks"] for r in records for e in r["evaluation_episodes"]]
            ),
            pooled_solved_evaluation=PriorCalibration.summary(
                values=[
                    e["ticks"] for r in records for e in r["evaluation_episodes"] if e["solved"]
                ]
            ),
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    PriorCalibration.main()
