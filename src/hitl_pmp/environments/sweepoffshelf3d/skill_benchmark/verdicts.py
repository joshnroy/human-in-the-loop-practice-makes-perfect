"""Labelling each benchmark run, and judging every robot skill against its target.

Targets (Josh, 2026-10-15): a skill with no continuous parameters must succeed from
every benchmark state; a skill with continuous parameters (only `SweepShelf`) must have,
from every state, at least one setting on `SWEEP_GRID` that succeeds. A skill with no
valid benchmark case has not been shown to meet anything, so it does not pass.

## Which simulator a failure belongs to

KINDER's controllers plan in PyBullet and execute in MuJoCo; that gap is deliberate (it
stands in for planning in simulation and running on a robot). The attempt's trace
(`skill_trace.py`) records every candidate, so the stage that failed says where:
every candidate unplannable in PyBullet is `no_plan`; a PyBullet plan the MuJoCo
re-check refused is `plan_blocked`; a plan that ran in MuJoCo and still failed is
`execution_failed`. `SweepShelf` and `ReturnWiper` are scripted on MuJoCo alone, so
they have no PyBullet verdict.
"""

from collections import Counter, defaultdict

from hitl_pmp.environments.sweepoffshelf3d.skills import SweepOffShelf3DSkills as S
from hitl_pmp.environments.sweepoffshelf3d.types import SkillNote, SkillTrace

from .types import CaseResult, SkillVerdict

SWEEP_GRID: tuple[tuple[float, float], ...] = tuple(
    (height, speed) for height in (0.0, 0.01, 0.02, 0.03, 0.04) for speed in (0.05, 0.15, 0.25)
)
SCRIPTED_SKILLS = frozenset({S.SWEEP_SHELF.name, S.RETURN_WIPER.name})
CANDIDATE_KINDS = frozenset({"no_plan", "plan_blocked", "executed", "execution_failed"})
PLANNED_KINDS = frozenset({"plan_blocked", "executed", "execution_failed"})


class Verdicts:
    """A static-method container, never instantiated."""

    @staticmethod
    def label(*, trace: SkillTrace) -> tuple[str, str]:
        """(label, pybullet) for one run: the label is "success" or a
        `types.FAILURE_LABELS` entry; pybullet is "planned", "no_plan" or "n/a"."""
        tail = Verdicts.failing_stage(notes=trace.notes)
        kinds = {note.kind for note in tail}
        if trace.skill in SCRIPTED_SKILLS:
            pybullet = "n/a"
        else:
            pybullet = "planned" if kinds & PLANNED_KINDS else "no_plan"
        if trace.outcome == "success":
            return "success", pybullet
        if trace.outcome == "interrupted":
            return "budget", pybullet
        if trace.outcome == "effects_missing":
            return "effects_missing", pybullet
        if kinds & {"executed", "execution_failed"}:
            return "execution_failed", pybullet
        if "plan_blocked" in kinds:
            return "plan_blocked", pybullet
        if tail:
            return "no_plan", pybullet
        return ("no_plan" if trace.outcome == "plan_failure" else "execution_failed"), pybullet

    @staticmethod
    def failing_stage(*, notes: list[SkillNote]) -> list[SkillNote]:
        """The candidate notes of the last stage the skill reached."""
        candidates = [note for note in notes if note.kind in CANDIDATE_KINDS]
        if not candidates:
            return []
        stage = candidates[-1].stage
        tail: list[SkillNote] = []
        for note in reversed(candidates):
            if note.stage != stage:
                break
            tail.append(note)
        return tail[::-1]

    @staticmethod
    def has_continuous_parameters(*, skill: str) -> bool:
        return {s.name: s.param_dim for s in S.robot_skills()}.get(skill, 0) > 0

    @staticmethod
    def judge(*, results: list[CaseResult]) -> list[SkillVerdict]:
        by_skill: dict[str, list[CaseResult]] = defaultdict(list)
        for result in results:
            if result.label != "invalid":
                by_skill[result.skill].append(result)
        verdicts = []
        for skill in (s.name for s in S.robot_skills()):
            runs = by_skill.get(skill, [])
            successes = sum(r.label == "success" for r in runs)
            labels = dict(Counter(r.label for r in runs))
            pybullet = (
                None if skill in SCRIPTED_SKILLS else sum(r.pybullet == "planned" for r in runs)
            )
            if Verdicts.has_continuous_parameters(skill=skill):
                per_state: dict[str, list[int]] = {}
                for run in runs:
                    tally = per_state.setdefault(run.state_id, [0, 0])
                    tally[0] += run.label == "success"
                    tally[1] += 1
                passed = bool(per_state) and all(ok > 0 for ok, _ in per_state.values())
                verdicts.append(
                    SkillVerdict(
                        skill=skill,
                        target="some setting per state",
                        passed=passed,
                        successes=successes,
                        total=len(runs),
                        labels=labels,
                        per_state=per_state,
                        pybullet_planned=pybullet,
                    )
                )
            else:
                verdicts.append(
                    SkillVerdict(
                        skill=skill,
                        target="every state",
                        passed=bool(runs) and successes == len(runs),
                        successes=successes,
                        total=len(runs),
                        labels=labels,
                        pybullet_planned=pybullet,
                    )
                )
        return verdicts

    @staticmethod
    def passed(*, verdicts: list[SkillVerdict]) -> bool:
        return bool(verdicts) and all(v.passed for v in verdicts)
