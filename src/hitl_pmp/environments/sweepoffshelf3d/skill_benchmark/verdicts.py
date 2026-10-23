"""Labelling each benchmark run, and judging every robot skill against its target.

Targets (Josh, 2026-10-15): a skill with no continuous parameters must succeed from
every benchmark state; a skill with continuous parameters (only `SweepShelf`) must have,
from every state, at least one setting on `SWEEP_GRID` that succeeds. Since SweepShelf
aims one stroke at one cube (2026-10-23) that is per (state, target cube), and
`SETTING_THRESHOLDS` relaxes "every pair" to a pair count. A skill with no
valid benchmark case has not been shown to meet anything, so it does not pass.

`THRESHOLDS` (Josh, 2026-10-20) replaces "every state" for four skills with x/y of their
gated runs. `ReturnCubeToShelfFromBox` gates on the states harvested from practice only;
its synthetic container states still run and are reported, but do not gate.

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

from .types import CaseResult, SettingThreshold, SkillThreshold, SkillVerdict

# The count, not a fraction: a state set that gains or loses a case changes the
# denominator, and the gate then refuses instead of passing at a rate never agreed.
# PickCubeFromShelf is a deliberate gate lowering (Josh, 2026-10-23): #539's neighbour
# refusals leave about 1/4 of shelf picks refused, measured 75/100 in the wiper-on-top
# scene with a 6 mm wiper margin; the gate is that minus 3. A fix to the shelf pick's
# grasp and base generation is in progress; raise this back when it lands.
THRESHOLDS: dict[str, SkillThreshold] = {
    S.PICK_CUBE_FROM_SHELF.name: SkillThreshold(min_successes=72, out_of=100),
    S.PICK_CUBE_FROM_FLOOR.name: SkillThreshold(min_successes=11, out_of=11),
    S.RETURN_CUBE_TO_SHELF_FROM_FLOOR.name: SkillThreshold(min_successes=11, out_of=11),
    S.RETURN_CUBE_TO_SHELF_FROM_BOX.name: SkillThreshold(
        min_successes=5, out_of=8, families=frozenset({"harvested"})
    ),
}
# A pair-count target for a continuous-parameter skill (Josh, 2026-10-23): the
# afb353fc1 gate run found a succeeding setting for 38/40 (state, target cube) pairs; the
# two without one are corner cubes at |y| ~ 0.225 m, beyond the +/- 0.17 m lane limit.
SETTING_THRESHOLDS: dict[str, SettingThreshold] = {
    S.SWEEP_SHELF.name: SettingThreshold(min_pairs=38, out_of=40),
}
# (lane offset [m], pull speed [m/s]) over SweepShelf's whole allowed range.
SWEEP_GRID: tuple[tuple[float, float], ...] = tuple(
    (offset, speed)
    for offset in (-0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.10)
    for speed in (0.05, 0.15, 0.25)
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
            threshold = THRESHOLDS.get(skill)
            if threshold is not None:
                verdicts.append(
                    Verdicts.against_threshold(
                        skill=skill,
                        threshold=threshold,
                        runs=runs,
                        labels=labels,
                        pybullet_planned=pybullet,
                    )
                )
            elif Verdicts.has_continuous_parameters(skill=skill):
                per_state: dict[str, list[int]] = {}
                for run in runs:
                    key = run.state_id if run.cube is None else f"{run.state_id}/{run.cube}"
                    tally = per_state.setdefault(key, [0, 0])
                    tally[0] += run.label == "success"
                    tally[1] += 1
                covered = sum(ok > 0 for ok, _ in per_state.values())
                setting = SETTING_THRESHOLDS.get(skill)
                reason = ""
                if setting is None:
                    target = "some setting per state and target cube"
                    passed = bool(per_state) and covered == len(per_state)
                else:
                    target = setting.target()
                    if len(per_state) != setting.out_of:
                        reason = (
                            f"the benchmark ran {len(per_state)} (state, target cube) pair(s), "
                            f"the threshold is out of {setting.out_of}; the state set "
                            "changed, so set the threshold again"
                        )
                    passed = not reason and covered >= setting.min_pairs
                verdicts.append(
                    SkillVerdict(
                        skill=skill,
                        target=target,
                        passed=passed,
                        reason=reason,
                        gated_pairs=covered,
                        gated_pairs_total=len(per_state),
                        successes=successes,
                        total=len(runs),
                        gated_successes=successes,
                        gated_total=len(runs),
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
                        gated_successes=successes,
                        gated_total=len(runs),
                        labels=labels,
                        pybullet_planned=pybullet,
                    )
                )
        return verdicts

    @staticmethod
    def against_threshold(
        *,
        skill: str,
        threshold: SkillThreshold,
        runs: list[CaseResult],
        labels: dict[str, int],
        pybullet_planned: int | None,
    ) -> SkillVerdict:
        gates = [threshold.families is None or r.family in threshold.families for r in runs]
        gated = [r for r, g in zip(runs, gates, strict=True) if g]
        reported = [r for r, g in zip(runs, gates, strict=True) if not g]
        gated_successes = sum(r.label == "success" for r in gated)
        reason = ""
        if len(gated) != threshold.out_of:
            reason = (
                f"the benchmark ran {len(gated)} gated case(s), the threshold is out of "
                f"{threshold.out_of}; the state set changed, so set the threshold again"
            )
        return SkillVerdict(
            skill=skill,
            target=threshold.target(),
            passed=not reason and bool(gated) and gated_successes >= threshold.min_successes,
            successes=sum(r.label == "success" for r in runs),
            total=len(runs),
            gated_successes=gated_successes,
            gated_total=len(gated),
            reported_successes=sum(r.label == "success" for r in reported),
            reported_total=len(reported),
            labels=labels,
            pybullet_planned=pybullet_planned,
            reason=reason,
        )

    @staticmethod
    def passed(*, verdicts: list[SkillVerdict]) -> bool:
        return bool(verdicts) and all(v.passed for v in verdicts)
