"""Success/total per robot skill, from any step-protocol run's logs.

    scripts/with_env.sh python -m scripts.skill_health <run dir or log> [...] \
        [--json out.json] [--fail-on-flag]

Reads every `step_events.jsonl` (practice) and `evaluations/*/actions.jsonl`
(deployment) under the given paths. Each attempt is counted under who chose it --
the planner, random exploration (a `random: ` label), or an evaluation -- and each
failure as a plan-time failure (no control step taken: nothing moved) or an execution
failure. An attempt cut off by the run's step budget is counted apart, not as a failure.

Success is the logged `skill_trace`'s verdict -- every promised effect holds -- when the
run logged one; a run from before traces existed has only the controller's error, so
there success means "the controller raised nothing" (`traced` counts which is which).

A skill is flagged when it falls below its benchmark target: every attempt, for a skill
with no continuous parameters. `SweepShelf`'s target is that *some* setting of its
continuous parameters (`[offset, speed]`) succeeds from each state for each target
cube, which a run's sampled settings cannot judge, so from logs it is flagged only when
no attempt succeeded at all.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S

RANDOM_PREFIX = "random: "
TOP_REASONS = 3


class SkillHealthReport(BaseModel):
    skills: dict[str, "SkillTally"] = Field(default_factory=dict)
    files: list[str] = Field(default_factory=list)


class SkillTally(BaseModel):
    planner: "Count" = Field(default_factory=lambda: Count())
    random: "Count" = Field(default_factory=lambda: Count())
    evaluation: "Count" = Field(default_factory=lambda: Count())
    plan_failures: int = 0
    execution_failures: int = 0
    interrupted: int = 0
    traced: int = 0
    reasons: Counter[str] = Field(default_factory=Counter)

    @property
    def successes(self) -> int:
        return self.planner.successes + self.random.successes + self.evaluation.successes

    @property
    def total(self) -> int:
        return self.planner.total + self.random.total + self.evaluation.total


class Count(BaseModel):
    successes: int = 0
    total: int = 0

    def text(self) -> str:
        return f"{self.successes}/{self.total}"


class SkillHealth:
    """A static-method container, never instantiated."""

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        parser.add_argument("paths", nargs="+", type=Path)
        parser.add_argument("--json", type=Path, default=None)
        parser.add_argument("--fail-on-flag", action="store_true")
        args = parser.parse_args(argv)
        report = SkillHealth.collect(paths=args.paths)
        flagged = SkillHealth.flagged(report=report)
        print(SkillHealth.render(report=report))
        if args.json is not None:
            payload = {
                "files": report.files,
                "flagged": flagged,
                "skills": {
                    name: {
                        "planner": tally.planner.model_dump(),
                        "random": tally.random.model_dump(),
                        "evaluation": tally.evaluation.model_dump(),
                        "successes": tally.successes,
                        "total": tally.total,
                        "plan_failures": tally.plan_failures,
                        "execution_failures": tally.execution_failures,
                        "interrupted": tally.interrupted,
                        "traced": tally.traced,
                        "top_reasons": tally.reasons.most_common(TOP_REASONS),
                    }
                    for name, tally in sorted(report.skills.items())
                },
            }
            args.json.write_text(json.dumps(payload, indent=2) + "\n")
        return 1 if args.fail_on_flag and flagged else 0

    @staticmethod
    def collect(*, paths: list[Path]) -> SkillHealthReport:
        report = SkillHealthReport()
        for path, source in SkillHealth.logs(paths=paths):
            report.files.append(str(path))
            for record in SkillHealth.records(path=path):
                if source == "practice" and record.get("event") != "robot":
                    continue
                SkillHealth.count(report=report, record=record, source=source)
        return report

    @staticmethod
    def logs(*, paths: list[Path]) -> Iterator[tuple[Path, str]]:
        for path in paths:
            if path.is_file():
                yield path, "evaluation" if path.name == "actions.jsonl" else "practice"
                continue
            for found in sorted(path.rglob("step_events.jsonl")):
                yield found, "practice"
            for found in sorted(path.rglob("actions.jsonl")):
                if found.parent.parent.name == "evaluations":
                    yield found, "evaluation"

    @staticmethod
    def records(*, path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def count(*, report: SkillHealthReport, record: dict[str, Any], source: str) -> None:
        label = str(record.get("label", ""))
        chooser = source
        if label.startswith(RANDOM_PREFIX):
            label, chooser = label[len(RANDOM_PREFIX) :], "random"
        elif source == "practice":
            chooser = "planner"
        skill = label.split("(")[0].strip()
        if not skill or skill in {"STOP", "no-op"}:
            return
        tally = report.skills.setdefault(skill, SkillTally())
        error = record.get("skill_error")
        if record.get("interrupted") or (error or "").startswith("interrupted:"):
            tally.interrupted += 1
            return
        trace = record.get("skill_trace")
        steps = int(record.get("end_step", 0)) - int(record.get("start_step", 0))
        if trace is not None:
            tally.traced += 1
            outcome = str(trace["outcome"])
            success = outcome == "success"
            plan_failure = outcome == "plan_failure"
        else:
            success = error is None
            plan_failure = not success and steps == 0
        count: Count = getattr(tally, chooser)
        count.total += 1
        if success:
            count.successes += 1
            return
        if plan_failure:
            tally.plan_failures += 1
        else:
            tally.execution_failures += 1
        tally.reasons[SkillHealth.reason(error=error, trace=trace)] += 1

    @staticmethod
    def reason(*, error: str | None, trace: dict[str, Any] | None) -> str:
        if error is None and trace is not None:
            return "missing: " + ", ".join(trace.get("missing_effects", [])[:3])
        return (error or "unknown")[:120]

    @staticmethod
    def has_continuous_parameters(*, skill: str) -> bool:
        params = {s.name: s.param_dim for s in S.robot_skills()}
        return params.get(skill, 0) > 0

    @staticmethod
    def flagged(*, report: SkillHealthReport) -> list[str]:
        out = []
        for name, tally in sorted(report.skills.items()):
            if tally.total == 0:
                continue
            if SkillHealth.has_continuous_parameters(skill=name):
                if tally.successes == 0:
                    out.append(name)
            elif tally.successes < tally.total:
                out.append(name)
        return out

    @staticmethod
    def render(*, report: SkillHealthReport) -> str:
        flagged = set(SkillHealth.flagged(report=report))
        header = (
            f"{'skill':<28}{'all':>9}{'planner':>9}{'random':>9}{'eval':>9}"
            f"{'plan-fail':>11}{'exec-fail':>11}{'cut-off':>9}{'traced':>8}  target"
        )
        lines = [f"{len(report.files)} log file(s)", header]
        for name, tally in sorted(report.skills.items()):
            target = (
                "some setting per state and target cube"
                if SkillHealth.has_continuous_parameters(skill=name)
                else "every attempt"
            )
            mark = "  BELOW TARGET" if name in flagged else ""
            total = Count(successes=tally.successes, total=tally.total)
            lines.append(
                f"{name:<28}{total.text():>9}{tally.planner.text():>9}{tally.random.text():>9}"
                f"{tally.evaluation.text():>9}{tally.plan_failures:>11}"
                f"{tally.execution_failures:>11}{tally.interrupted:>9}"
                f"{tally.traced:>8}  {target}{mark}"
            )
            for reason, n in tally.reasons.most_common(TOP_REASONS):
                lines.append(f"    {n} x {reason}")
        return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(SkillHealth.main())
