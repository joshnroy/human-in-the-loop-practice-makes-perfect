"""Plot measured EES practice outcomes, keeping unscored final actions explicit."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


class AutonomousEvidence:
    @staticmethod
    def summarize(*, stats: dict) -> dict:
        if stats["num_practice_resets"] != 0:
            raise ValueError("Practice reset detected")
        if stats["num_human_interventions_recorded"] != 0:
            raise ValueError("A human intervention was recorded")
        skills: dict[str, dict[str, int]] = {}
        for cycle in stats["practice_outcomes_per_cycle"]:
            for name, tally in cycle.items():
                totals = skills.setdefault(name, {"attempts": 0, "successes": 0})
                totals["attempts"] += tally["num_attempts"]
                totals["successes"] += tally["num_successes"]
        executed = stats["evaluations"][-1][0]
        scored = sum(tally["attempts"] for tally in skills.values())
        if scored > executed:
            raise ValueError(
                "Scored outcomes exceed executed actions; check double-observe settings"
            )
        return {
            "executed_actions": executed,
            "scored_actions": scored,
            "unscored_actions": executed - scored,
            "skills": skills,
            "practice_resets": stats["num_practice_resets"],
            "human_interventions": stats["num_human_interventions_recorded"],
        }

    @staticmethod
    def plot(*, run_dir: Path, output: Path) -> dict:
        summary = AutonomousEvidence.summarize(
            stats=json.loads((run_dir / "stats.json").read_text())
        )
        throws = [
            json.loads(line)
            for line in (run_dir / "sampler_draws.jsonl").read_text().splitlines()
            if json.loads(line)["skill"] == "MoveToTossLocationAndToss"
        ]
        fig, (ax, outcomes) = plt.subplots(
            2, 1, figsize=(9, 5.5), gridspec_kw={"height_ratios": [2, 1]}
        )
        names = list(summary["skills"])
        successes = [summary["skills"][name]["successes"] for name in names]
        failures = [
            summary["skills"][name]["attempts"] - success
            for name, success in zip(names, successes, strict=True)
        ]
        labels = [
            name.replace("MoveToTossLocationAndToss", "Throw").replace("PickCubeFrom", "Pick from ")
            for name in names
        ]
        ax.barh(labels, successes, label="Succeeded", color="#197d78")
        ax.barh(labels, failures, left=successes, label="Failed", color="#ce7058")
        for index, name in enumerate(names):
            tally = summary["skills"][name]
            ax.text(tally["attempts"] + 0.1, index, f"{tally['successes']}/{tally['attempts']}")
        ax.set_xlim(0, max((summary["skills"][name]["attempts"] for name in names), default=1) + 2)
        ax.set_xlabel("Scored practice actions")
        ax.legend(loc="lower right")
        for index, throw in enumerate(throws, 1):
            hit = throw["success"]
            outcomes.scatter(index, int(hit), color="#197d78" if hit else "#ce7058", s=75)
        outcomes.set_yticks([0, 1], ["Miss", "Hit"])
        outcomes.set_xticks(range(1, len(throws) + 1))
        outcomes.set_xlabel("Observed throw number (sampler log)")
        outcomes.set_ylim(-0.4, 1.4)
        fig.suptitle(
            f"EES autonomous practice: {summary['executed_actions']} executed actions, "
            f"{summary['unscored_actions']} unscored\n"
            "No practice resets or human interventions; no scripted skill sequence"
        )
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    @staticmethod
    def main() -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--run-dir", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        AutonomousEvidence.plot(run_dir=args.run_dir, output=args.output)


if __name__ == "__main__":
    AutonomousEvidence.main()
