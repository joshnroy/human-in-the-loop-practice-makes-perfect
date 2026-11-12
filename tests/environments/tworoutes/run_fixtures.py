"""Run `--env tworoutes` through the global CLI, in process, for tests."""

import json
from pathlib import Path
from typing import Any

from hitl_pmp.cli import Cli


class TwoRoutesRun:
    @staticmethod
    def costs(*, directory: Path, human_cost: float = 1.0, objective_lambda: float = 0.002) -> Path:
        path = directory / "costs.json"
        path.write_text(
            json.dumps({
                "robot_step": {"value": 1},
                "human_skills": {
                    "human_reset_b": {"cost": {"value": human_cost}, "duration": {"value": 1}}
                },
                "objective_lambda": objective_lambda,
            })
        )
        return path

    @staticmethod
    def run(
        *,
        directory: Path,
        method: str,
        budget: int = 30,
        human_cost: float = 1.0,
        extra: tuple[str, ...] = (),
    ) -> Path:
        output = directory / "run"
        Cli.main(
            argv=[
                "--env",
                "tworoutes",
                "--method",
                method,
                "--seed",
                "0",
                "--practice-step-budget",
                str(budget),
                "--practice-cost-config",
                str(TwoRoutesRun.costs(directory=directory, human_cost=human_cost)),
                "--practice-reset-policy",
                "never",
                "--defer-rendering",
                "--output-dir",
                str(output),
                "--num-test-tasks",
                "8",
                "--evaluation-max-pour-attempts",
                "1",
                "--measurement-interval-steps",
                "100000",
                "--max-steps-per-interaction",
                "10",
                *extra,
            ]
        )
        return output

    @staticmethod
    def events(*, output: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in (output / "step_events.jsonl").read_text().splitlines()
        ]

    @staticmethod
    def evaluation_actions(*, output: Path, index: int) -> list[dict[str, Any]]:
        path = output / "evaluations" / f"{index:04d}" / "actions.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]
