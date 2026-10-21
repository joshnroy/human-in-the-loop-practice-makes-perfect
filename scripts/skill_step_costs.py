"""Measure the belief-space planner's fixed per-skill prices from a skill benchmark.

    python -m scripts.skill_step_costs <report dir> [--out <table.json>]

`<report dir>` is one `benchmark-reports/<domain>/<commit>/` directory: its `report.json`
and the per-case `results-*.jsonl` beside it. Each skill's price is the mean
`controller_steps` over the cases that executed, successful AND failed
(`success`, `execution_failed`, `effects_missing`, ...). A case refused at planning
(`no_plan`, `plan_blocked`) ran no control step and is excluded and counted in
`planning_refusals_excluded`. The table records the benchmark commit and the sha256 of
`report.json`, so a table can always be traced to the report it was measured from.

Re-run at every launch commit whose controllers change a skill's duration.
"""

import argparse
import hashlib
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

from hitl_pmp.core.practice_costs import SkillStepCosts, SkillStepCostSource

STATISTIC = (
    "mean controller_steps per skill over every benchmark case that executed, successful "
    "and failed; cases refused at planning (0 control steps) are excluded"
)


class SkillStepCostTable:
    """A static-method container, never instantiated."""

    @staticmethod
    def measure(*, report_dir: Path) -> SkillStepCosts:
        report_path = report_dir / "report.json"
        raw = report_path.read_bytes()
        report = json.loads(raw)
        if report.get("git_dirty", False):
            raise SystemExit(f"{report_path}: benchmarked from a dirty checkout; refusing")
        executed: dict[str, list[int]] = defaultdict(list)
        refused: dict[str, int] = defaultdict(int)
        for results in sorted(report_dir.glob("results-*.jsonl")):
            for line in results.read_text().splitlines():
                if not line.strip():
                    continue
                case = json.loads(line)
                steps = int(case["controller_steps"])
                if steps > 0:
                    executed[case["skill"]].append(steps)
                else:
                    refused[case["skill"]] += 1
        if not executed:
            raise SystemExit(f"{report_dir}: no executed cases in results-*.jsonl")
        return SkillStepCosts(
            source=SkillStepCostSource(
                benchmark_commit=report["git_commit"],
                report_sha256=hashlib.sha256(raw).hexdigest(),
                report_dir=str(report_dir),
                statistic=STATISTIC,
                cases={skill: len(steps) for skill, steps in sorted(executed.items())},
                planning_refusals_excluded={
                    skill: refused.get(skill, 0) for skill in sorted(executed)
                },
                command=shlex.join(["python", "-m", "scripts.skill_step_costs", str(report_dir)]),
            ),
            skill_steps={
                skill: round(sum(steps) / len(steps), 1)
                for skill, steps in sorted(executed.items())
            },
        )


def main(*, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("report_dir", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "sweepoffshelf3d_skill_step_costs.json",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    table = SkillStepCostTable.measure(report_dir=args.report_dir)
    args.out.write_text(json.dumps(table.model_dump(mode="json"), indent=2) + "\n")
    for skill, steps in table.skill_steps.items():
        print(f"{skill:28s} {steps:8.1f}  (n={table.source.cases[skill]})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
