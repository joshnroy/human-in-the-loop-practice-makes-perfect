"""Labelling each benchmark run from its trace, and judging each skill's target."""

from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import CaseResult
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.verdicts import SWEEP_GRID, Verdicts
from hitl_pmp.environments.sweepoffshelf3d.types import SkillNote, SkillTrace


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
