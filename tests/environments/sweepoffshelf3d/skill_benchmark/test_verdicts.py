"""Labelling each benchmark run from its trace, and judging each skill's target."""

import pytest

from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import (
    CaseResult,
    SkillThreshold,
    SkillVerdict,
)
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.verdicts import (
    SWEEP_GRID,
    THRESHOLDS,
    Verdicts,
)
from hitl_pmp.environments.sweepoffshelf3d.types import SkillNote, SkillTrace
from hitl_pmp.launch_gate import LaunchGate


def _trace(*, skill: str, outcome: str, notes: list[SkillNote], steps: int = 100) -> SkillTrace:
    return SkillTrace(
        skill=skill,
        objects=[],
        params=[],
        outcome=outcome,
        error=None if outcome == "success" else "failed",
        controller_steps=steps,
        missing_effects=[],
        added=[],
        removed=[],
        notes=notes,
    )


def _note(*, stage: str, kind: str) -> SkillNote:
    return SkillNote(stage=stage, kind=kind, reason="r", params=[0.0])


def test_every_candidate_unplannable_is_a_controller_gap() -> None:
    trace = _trace(
        skill="PickWiper",
        outcome="plan_failure",
        notes=[_note(stage="pick_wiper", kind="no_plan")] * 3,
        steps=0,
    )
    assert Verdicts.label(trace=trace) == ("no_plan", "no_plan")


def test_planned_in_pybullet_but_refused_in_mujoco_is_the_simulator_gap() -> None:
    trace = _trace(
        skill="PickCubeFromShelf",
        outcome="plan_failure",
        notes=[
            _note(stage="pick_cube_from_shelf", kind="no_plan"),
            _note(stage="pick_cube_from_shelf", kind="plan_blocked"),
        ],
        steps=0,
    )
    assert Verdicts.label(trace=trace) == ("plan_blocked", "planned")


def test_the_failing_stage_decides_and_an_executed_earlier_stage_does_not_mask_it() -> None:
    # The pick executed; then every placement failed to plan.
    trace = _trace(
        skill="ReturnCubeToShelfFromBox",
        outcome="execution_failure",
        notes=[
            _note(stage="return_pick", kind="executed"),
            _note(stage="return_place", kind="no_plan"),
            _note(stage="return_place", kind="plan_blocked"),
            _note(stage="return_place_retract", kind="retract"),
        ],
    )
    assert Verdicts.label(trace=trace) == ("plan_blocked", "planned")


def test_a_plan_that_ran_and_failed_is_an_execution_failure() -> None:
    trace = _trace(
        skill="PickWiper",
        outcome="execution_failure",
        notes=[
            _note(stage="pick_wiper", kind="no_plan"),
            _note(stage="pick_wiper", kind="executed"),
        ],
    )
    assert Verdicts.label(trace=trace) == ("execution_failed", "planned")


def test_scripted_skills_have_no_pybullet_verdict() -> None:
    sweep = _trace(skill="SweepShelf", outcome="execution_failure", notes=[])
    assert Verdicts.label(trace=sweep) == ("execution_failed", "n/a")
    route = _trace(skill="ReturnWiper", outcome="plan_failure", notes=[], steps=0)
    assert Verdicts.label(trace=route) == ("no_plan", "n/a")
    ok = _trace(skill="ReturnWiper", outcome="success", notes=[])
    assert Verdicts.label(trace=ok) == ("success", "n/a")


def test_missing_effects_and_budget_cut_offs_keep_their_own_labels() -> None:
    assert Verdicts.label(
        trace=_trace(skill="PlaceCubeInBox", outcome="effects_missing", notes=[])
    )[0] == ("effects_missing")
    assert Verdicts.label(trace=_trace(skill="PlaceCubeInBox", outcome="interrupted", notes=[]))[
        0
    ] == ("budget")


def _result(*, skill: str, state: str, label: str, params: list[float] | None = None) -> CaseResult:
    return CaseResult(
        case_id=f"{state}-{skill}-{params}",
        state_id=state,
        family="f",
        skill=skill,
        params=params or [],
        label=label,
    )


def test_a_skill_without_continuous_parameters_must_succeed_from_every_state() -> None:
    verdicts = {
        v.skill: v
        for v in Verdicts.judge(
            results=[
                _result(skill="PickWiper", state="s0", label="success"),
                _result(skill="PickWiper", state="s1", label="no_plan"),
                _result(skill="PlaceCubeInBox", state="s0", label="success"),
                _result(skill="PlaceCubeInBox", state="s9", label="invalid"),
            ]
        )
    }
    assert (verdicts["PickWiper"].successes, verdicts["PickWiper"].total) == (1, 2)
    assert not verdicts["PickWiper"].passed
    assert verdicts["PickWiper"].labels == {"success": 1, "no_plan": 1}
    # An invalid case is not a run of the skill at all.
    assert (verdicts["PlaceCubeInBox"].successes, verdicts["PlaceCubeInBox"].total) == (1, 1)
    assert verdicts["PlaceCubeInBox"].passed


def test_sweep_passes_when_every_state_has_some_succeeding_setting() -> None:
    grid = list(SWEEP_GRID)
    results = [
        _result(
            skill="SweepShelf",
            state="a",
            label="success" if i == 0 else "execution_failed",
            params=list(p),
        )
        for i, p in enumerate(grid)
    ] + [_result(skill="SweepShelf", state="b", label="success", params=list(p)) for p in grid]
    (sweep,) = [v for v in Verdicts.judge(results=results) if v.skill == "SweepShelf"]
    assert sweep.target == "some setting per state"
    assert sweep.passed
    assert sweep.per_state == {"a": [1, len(grid)], "b": [len(grid), len(grid)]}
    results += [
        _result(skill="SweepShelf", state="c", label="execution_failed", params=list(p))
        for p in grid
    ]
    (sweep,) = [v for v in Verdicts.judge(results=results) if v.skill == "SweepShelf"]
    assert not sweep.passed


def test_a_skill_with_no_benchmark_case_does_not_pass() -> None:
    verdicts = {v.skill: v for v in Verdicts.judge(results=[])}
    assert set(verdicts) == {
        "PickWiper",
        "SweepShelf",
        "ReturnWiper",
        "PickCubeFromShelf",
        "PickCubeFromFloor",
        "PlaceCubeInBox",
        "ReturnCubeToShelfFromBox",
        "ReturnCubeToShelfFromFloor",
    }
    assert not any(v.passed for v in verdicts.values())
    assert Verdicts.passed(verdicts=list(verdicts.values())) is False


def test_the_sweep_grid_spans_the_whole_allowed_range() -> None:
    heights = sorted({h for h, _ in SWEEP_GRID})
    speeds = sorted({s for _, s in SWEEP_GRID})
    assert heights[0] == 0.0 and heights[-1] == 0.04
    assert speeds[0] == 0.05 and speeds[-1] == 0.25


# Per-skill thresholds (Josh, 2026-10-20): each is x/y of a skill's gated cases, and the
# gated case count must equal the threshold's denominator.


def _runs(
    *, skill: str, family: str, successes: int, total: int, prefix: str = "s"
) -> list[CaseResult]:
    return [
        CaseResult(
            case_id=f"{prefix}{i}-{skill}",
            state_id=f"{prefix}{i}",
            family=family,
            skill=skill,
            label="success" if i < successes else "execution_failed",
        )
        for i in range(total)
    ]


def _verdict(*, skill: str, results: list[CaseResult]) -> SkillVerdict:
    (verdict,) = [v for v in Verdicts.judge(results=results) if v.skill == skill]
    return verdict


@pytest.mark.parametrize(
    ("skill", "needed", "out_of"),
    [
        ("PickCubeFromShelf", 97, 100),
        ("PickCubeFromFloor", 11, 12),
        ("ReturnCubeToShelfFromFloor", 11, 12),
    ],
)
def test_a_thresholded_skill_passes_at_exactly_its_threshold_and_fails_one_below(
    *, skill: str, needed: int, out_of: int
) -> None:
    assert THRESHOLDS[skill] == SkillThreshold(min_successes=needed, out_of=out_of)
    at = _verdict(
        skill=skill, results=_runs(skill=skill, family="start", successes=needed, total=out_of)
    )
    assert at.passed
    assert (at.gated_successes, at.gated_total) == (needed, out_of)
    assert (at.reported_successes, at.reported_total) == (0, 0)
    assert at.target == f"at least {needed}/{out_of}"
    below = _verdict(
        skill=skill,
        results=_runs(skill=skill, family="start", successes=needed - 1, total=out_of),
    )
    assert not below.passed
    assert (below.gated_successes, below.gated_total) == (needed - 1, out_of)


def test_the_box_return_gates_on_harvested_states_and_only_reports_synthetic_ones() -> None:
    skill = "ReturnCubeToShelfFromBox"
    assert THRESHOLDS[skill] == SkillThreshold(
        min_successes=3, out_of=8, families=frozenset({"harvested"})
    )
    at = _verdict(
        skill=skill,
        results=_runs(skill=skill, family="harvested", successes=3, total=8, prefix="h")
        + _runs(skill=skill, family="near_walls", successes=0, total=13, prefix="c"),
    )
    assert at.passed
    assert (at.gated_successes, at.gated_total) == (3, 8)
    assert (at.reported_successes, at.reported_total) == (0, 13)
    assert (at.successes, at.total) == (3, 21)
    assert at.target == "at least 3/8 of harvested"
    below = _verdict(
        skill=skill,
        results=_runs(skill=skill, family="harvested", successes=2, total=8, prefix="h")
        + _runs(skill=skill, family="near_walls", successes=13, total=13, prefix="c"),
    )
    assert not below.passed
    assert (below.reported_successes, below.reported_total) == (13, 13)


@pytest.mark.parametrize("total", [99, 101])
def test_a_changed_case_count_is_refused_rather_than_judged(*, total: int) -> None:
    verdict = _verdict(
        skill="PickCubeFromShelf",
        results=_runs(skill="PickCubeFromShelf", family="start", successes=total, total=total),
    )
    assert not verdict.passed
    assert f"{total} gated case(s)" in verdict.reason
    assert "out of 100" in verdict.reason


def test_an_invalid_gated_case_changes_the_denominator_and_is_refused() -> None:
    results = _runs(skill="PickCubeFromFloor", family="near_walls", successes=12, total=12)
    results[0] = results[0].model_copy(update={"label": "invalid"})
    verdict = _verdict(skill="PickCubeFromFloor", results=results)
    assert not verdict.passed
    assert "11 gated case(s)" in verdict.reason


def test_a_thresholded_skill_with_no_valid_case_does_not_pass() -> None:
    verdict = _verdict(skill="ReturnCubeToShelfFromBox", results=[])
    assert not verdict.passed
    assert (verdict.gated_total, verdict.reported_total) == (0, 0)


@pytest.mark.parametrize("skill", ["PickWiper", "ReturnWiper", "PlaceCubeInBox"])
def test_the_unthresholded_skills_still_must_succeed_from_every_state(*, skill: str) -> None:
    assert skill not in THRESHOLDS
    every = _verdict(skill=skill, results=_runs(skill=skill, family="f", successes=6, total=6))
    assert every.passed
    assert every.target == "every state"
    assert (every.gated_successes, every.gated_total) == (6, 6)
    assert (every.reported_successes, every.reported_total) == (0, 0)
    one_short = _verdict(
        skill=skill, results=_runs(skill=skill, family="f", successes=99, total=100)
    )
    assert not one_short.passed


def test_the_sweep_keeps_its_some_setting_per_state_target() -> None:
    assert "SweepShelf" not in THRESHOLDS
    results = [
        _result(skill="SweepShelf", state="a", label="success", params=list(p)) for p in SWEEP_GRID
    ]
    sweep = _verdict(skill="SweepShelf", results=results)
    assert sweep.target == "some setting per state"
    assert sweep.passed
    assert (sweep.gated_successes, sweep.gated_total) == (len(SWEEP_GRID), len(SWEEP_GRID))


def test_the_thresholds_name_only_robot_skills_and_families_the_state_set_has() -> None:
    from pathlib import Path

    from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateSet

    states = StateSet.load(
        path=Path(__file__).resolve().parents[4] / "benchmarks/sweepoffshelf3d/states.jsonl"
    )
    families = {state.family for state in states}
    skills = {case.skill for state in states for case in state.cases}
    for skill, threshold in THRESHOLDS.items():
        assert skill in skills
        assert threshold.families is None or threshold.families <= families
        gated = sum(
            case.skill == skill
            and (threshold.families is None or state.family in threshold.families)
            for state in states
            for case in state.cases
        )
        assert gated == threshold.out_of, skill


def test_a_verdict_summarises_gated_and_reported_only_counts() -> None:
    skill = "ReturnCubeToShelfFromBox"
    results = _runs(skill=skill, family="harvested", successes=2, total=8, prefix="h") + _runs(
        skill=skill, family="near_walls", successes=5, total=13, prefix="c"
    )
    summary = LaunchGate.describe(skill=_verdict(skill=skill, results=results).model_dump())
    assert summary.startswith(skill)
    assert "gated 2/8" in summary
    assert "at least 3/8 of harvested" in summary
    assert "reported-only 5/13" in summary
