"""Labelling each pick run, and judging both pick skills against their target.

Neither pick has continuous parameters, so the target is Josh's first one: success
from every benchmark state the operator admits. "Admits" leaves `PickPlannable` out on
purpose -- the dry run predicts the controller, it is not a property of the world, and
the states where it says no are exactly the ones this benchmark exists to measure. A
state outside the operator's other preconditions (a tipped bin, a full hand) is run and
reported but never counted.
"""

from collections import Counter

from .types import PickResult, PickVerdict

PICK_SKILLS = ("PickCube", "PickCubeFromBin")


class PickVerdicts:
    """A static-method container, never instantiated."""

    @staticmethod
    def label(*, steps: int, terminated: bool, error: str | None, holding: bool) -> str:
        if holding:
            return "success"
        if steps == 0:
            return "no_plan"
        if error is None and not terminated:
            return "budget"
        return "execution_failed"

    @staticmethod
    def judge(*, results: list[PickResult]) -> list[PickVerdict]:
        verdicts = []
        for skill in PICK_SKILLS:
            runs = [r for r in results if r.skill == skill and r.label != "invalid"]
            counted = [r for r in runs if r.in_target]
            successes = sum(r.label == "success" for r in counted)
            per_family: dict[str, list[int]] = {}
            for run in counted:
                tally = per_family.setdefault(run.family, [0, 0])
                tally[0] += run.label == "success"
                tally[1] += 1
            judged = [r for r in counted if r.dry_run_plannable is not None]
            verdicts.append(
                PickVerdict(
                    skill=skill,
                    successes=successes,
                    total=len(counted),
                    passed=bool(counted) and successes == len(counted),
                    labels=dict(Counter(r.label for r in counted)),
                    pybullet_planned=sum(r.pybullet == "planned" for r in counted),
                    dry_run_plannable=sum(bool(r.dry_run_plannable) for r in counted),
                    dry_run_agrees=sum(
                        bool(r.dry_run_plannable) == (r.pybullet == "planned") for r in judged
                    ),
                    outside_operator=sum(not r.in_target for r in runs),
                    per_family=per_family,
                )
            )
        return verdicts

    @staticmethod
    def passed(*, verdicts: list[PickVerdict]) -> bool:
        return bool(verdicts) and all(v.passed and v.total > 0 for v in verdicts)
