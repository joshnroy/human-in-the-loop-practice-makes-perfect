"""Does the untrained policy leave practice anything to show? Read from one evaluation.

    scripts/with_env.sh python -m scripts.launch_headroom <evaluation dir> [...] \
        [--max-solved 7] [--report-only]

An evaluation directory is a step-protocol run's `evaluations/<index>/`; the step-0 one
is `evaluations/0000/`. `scripts/sweepoffshelf3d_launch.py headroom` runs exactly that
evaluation for each arm and reads it back through here.

Read from `recording.jsonl`, not `actions.jsonl`: the call the goal latch stops is
logged only in the recording (as `goal_reached`), so `actions.jsonl` misses the very
call that solved each task. Per skill: calls, and their verdicts -- a call the goal
latch stopped is a success, one the evaluation step budget cut off has no verdict and is
counted apart. Per episode: the calls up to and including the one that reached the goal,
in all and of continuous-parameter skills only. Per continuous-parameter skill: success
in equal-width bins of each parameter, over the range the evaluation drew.

There is no headroom -- practice cannot move the learning curve -- when either:
- the step-0 policy solves more than `--max-solved` of the held-out tasks, or
- the median episode reaches the goal within one continuous-parameter call (unsolved
  episodes rank last, and the upper median is taken, so this is "more than half the
  episodes"). A skill that solves the task in its first call has nothing to learn.
Launch shelf-20261023b hit both: 8/10 solved at step 0, 7/10 by the first SweepShelf.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from scripts.skill_health import SkillHealth

RECORDING = "recording.jsonl"
DEFAULT_MAX_SOLVED = 7
PARAMETER_BINS = 3
INTERRUPTED_PREFIX = "interrupted:"
LABEL_PARAMS = re.compile(r"params=\[([^\]]*)\]")


class EvaluationHeadroom(BaseModel):
    solved: int = 0
    total: int = 0
    task_seeds: list[int] = Field(default_factory=list)
    skills: dict[str, "SkillUse"] = Field(default_factory=dict)
    calls_to_goal: list[int | None] = Field(default_factory=list)
    continuous_calls_to_goal: list[int | None] = Field(default_factory=list)
    parameter_bins: dict[str, list["ParameterBin"]] = Field(default_factory=dict)

    @property
    def one_call_episodes(self) -> int:
        """Episodes that reached the goal within one continuous-parameter call."""
        return sum(n is not None and n <= 1 for n in self.continuous_calls_to_goal)


class SkillUse(BaseModel):
    calls: int = 0
    successes: int = 0
    goal_reached: int = 0
    failures: int = 0
    interrupted: int = 0

    @property
    def judged(self) -> int:
        return self.successes + self.failures


class ParameterBin(BaseModel):
    dimension: int
    low: float
    high: float
    successes: int = 0
    total: int = 0


class LaunchHeadroom:
    """A static-method container, never instantiated."""

    @staticmethod
    def main(*, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
        parser.add_argument("evaluations", nargs="+", type=Path)
        parser.add_argument("--max-solved", type=int, default=DEFAULT_MAX_SOLVED)
        parser.add_argument("--report-only", action="store_true", help="Always exit 0.")
        args = parser.parse_args(argv)
        blocked = False
        for evaluation in args.evaluations:
            headroom = LaunchHeadroom.analyse(evaluation=evaluation)
            print(
                LaunchHeadroom.render(
                    arm=str(evaluation), headroom=headroom, max_solved=args.max_solved
                )
            )
            blocked |= bool(LaunchHeadroom.problems(headroom=headroom, max_solved=args.max_solved))
        return 1 if blocked and not args.report_only else 0

    @staticmethod
    def analyse(*, evaluation: Path) -> EvaluationHeadroom:
        headroom = EvaluationHeadroom()
        draws: dict[str, list[tuple[list[float], bool]]] = {}
        calls = continuous = 0
        for record in LaunchHeadroom.records(path=evaluation / RECORDING):
            if record.get("phase") != "evaluation":
                continue
            kind = record.get("kind")
            if kind == "segment":
                calls = continuous = 0
                headroom.task_seeds.append(int(record["task_seed"]))
            elif kind == "action" and record.get("actor") == "robot":
                skill = str(record["skill"])
                calls += 1
                has_parameters = SkillHealth.has_continuous_parameters(skill=skill)
                continuous += has_parameters
                verdict = LaunchHeadroom.count(
                    use=headroom.skills.setdefault(skill, SkillUse()), record=record
                )
                if has_parameters and verdict is not None:
                    params = LaunchHeadroom.parameters(record=record)
                    draws.setdefault(skill, []).append((params, verdict))
            elif kind == "segment_end":
                solved = bool(record["success"])
                headroom.total += 1
                headroom.solved += solved
                headroom.calls_to_goal.append(calls if solved else None)
                headroom.continuous_calls_to_goal.append(continuous if solved else None)
        headroom.parameter_bins = {
            skill: LaunchHeadroom.bins(draws=found) for skill, found in sorted(draws.items())
        }
        return headroom

    @staticmethod
    def records(*, path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def count(*, use: SkillUse, record: dict[str, Any]) -> bool | None:
        """Tally one call; return its verdict, or `None` when the budget cut it off."""
        use.calls += 1
        if record.get("goal_reached"):
            use.goal_reached += 1
            use.successes += 1
            return True
        if str(record.get("outcome") or "").startswith(INTERRUPTED_PREFIX):
            use.interrupted += 1
            return None
        if record.get("success"):
            use.successes += 1
            return True
        use.failures += 1
        return False

    @staticmethod
    def parameters(*, record: dict[str, Any]) -> list[float]:
        """The recording's `params`, else the ones the label prints (2 d.p.)."""
        if record.get("params"):
            return [float(p) for p in record["params"]]
        match = LABEL_PARAMS.search(str(record.get("label", "")))
        if match is None or not match.group(1).strip():
            return []
        return [float(p) for p in match.group(1).split(",")]

    @staticmethod
    def bins(*, draws: list[tuple[list[float], bool]]) -> list[ParameterBin]:
        out: list[ParameterBin] = []
        for dimension in range(min((len(p) for p, _ in draws), default=0)):
            values = [p[dimension] for p, _ in draws]
            low, high = min(values), max(values)
            width = (high - low) / PARAMETER_BINS
            dimension_bins = [
                ParameterBin(dimension=dimension, low=low + i * width, high=low + (i + 1) * width)
                for i in range(PARAMETER_BINS if width > 0 else 1)
            ]
            dimension_bins[-1].high = high
            for params, success in draws:
                index = (
                    min(int((params[dimension] - low) / width), PARAMETER_BINS - 1)
                    if width > 0
                    else 0
                )
                dimension_bins[index].total += 1
                dimension_bins[index].successes += success
            out.extend(dimension_bins)
        return out

    @staticmethod
    def problems(*, headroom: EvaluationHeadroom, max_solved: int) -> list[str]:
        out = []
        if headroom.solved > max_solved:
            out.append(
                f"the step-0 policy already solves {headroom.solved}/{headroom.total} held-out "
                f"tasks, above the headroom threshold of at most {max_solved}/{headroom.total}"
            )
        one_call = headroom.one_call_episodes
        if headroom.total and 2 * one_call > headroom.total:
            out.append(
                f"{one_call}/{headroom.total} episodes reach the goal within one "
                "continuous-parameter skill call, so the median episode does: a skill that "
                "solves the task in one call leaves practice nothing to learn"
            )
        return out

    @staticmethod
    def render(*, arm: str, headroom: EvaluationHeadroom, max_solved: int) -> str:
        def calls(*, values: list[int | None]) -> str:
            return " ".join("-" if n is None else str(n) for n in values)

        lines = [
            f"{arm}: step-0 policy solved {headroom.solved}/{headroom.total} held-out tasks "
            f"(task seeds {headroom.task_seeds})",
            f"  calls to goal, all skills:        {calls(values=headroom.calls_to_goal)}",
            f"  calls to goal, continuous skills: {calls(values=headroom.continuous_calls_to_goal)}"
            f"  ({headroom.one_call_episodes}/{headroom.total} within one)",
            f"  {'skill':<28}{'calls':>7}{'success':>10}{'goal':>6}{'failed':>8}{'cut-off':>9}",
        ]
        for name, use in sorted(headroom.skills.items()):
            lines.append(
                f"  {name:<28}{use.calls:>7}{f'{use.successes}/{use.judged}':>10}"
                f"{use.goal_reached:>6}{use.failures:>8}{use.interrupted:>9}"
            )
        for name, found in headroom.parameter_bins.items():
            for b in found:
                lines.append(
                    f"  {name} param {b.dimension} in [{b.low:.3f}, {b.high:.3f}]: "
                    f"{b.successes}/{b.total}"
                )
        problems = LaunchHeadroom.problems(headroom=headroom, max_solved=max_solved)
        lines.extend(f"  NO HEADROOM: {problem}" for problem in problems)
        if not problems:
            lines.append("  headroom OK")
        return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(LaunchHeadroom.main())
