"""Each skill attempt's full account -- outcome, promised effects, every rejected
candidate -- built without KINDER, so it is checked on any machine."""

import json

import numpy as np

from hitl_pmp.environments.sweepoffshelf3d.environment import SKILL_IDS
from hitl_pmp.environments.sweepoffshelf3d.skill_trace import (
    MAX_NOTES,
    MAX_REASON_CHARS,
    SkillTraces,
)
from hitl_pmp.environments.sweepoffshelf3d.types import SkillNote

START = frozenset({
    ("HandEmpty", ("robot",)),
    ("WiperHome", ("wiper_0",)),
    ("AnyCubeOnShelf", ()),
    *((("OnShelf", (f"cube_{i}",))) for i in range(5)),
})


def _action(*, skill: str, cube: int = 0, height: float = 0.0, speed: float = 0.0) -> np.ndarray:
    return np.array([float(SKILL_IDS[skill]), float(cube), height, speed])


def _picked_wiper() -> frozenset:
    return (START - {("HandEmpty", ("robot",)), ("WiperHome", ("wiper_0",))}) | {
        ("HoldingWiper", ("robot", "wiper_0")),
        ("WiperAway", ("wiper_0",)),
    }


def test_a_skill_whose_promised_effects_all_hold_is_a_success() -> None:
    trace = SkillTraces.build(
        action=_action(skill="PickWiper"),
        atoms_before=START,
        atoms_after=_picked_wiper(),
        error=None,
        controller_steps=743,
        notes=[],
    )
    assert trace.skill == "PickWiper"
    assert trace.outcome == "success"
    assert trace.missing_effects == []
    assert trace.added == ["HoldingWiper(robot,wiper_0)", "WiperAway(wiper_0)"]
    assert trace.removed == ["HandEmpty(robot)", "WiperHome(wiper_0)"]


def test_a_failure_that_moved_nothing_is_a_plan_failure_and_keeps_every_rejection() -> None:
    notes = [
        SkillNote(stage="pick_wiper", kind="no_plan", params=[0.7, -3.1416], reason="IK failed"),
        SkillNote(stage="pick_wiper", kind="plan_blocked", params=[0.6, 0.0], reason="hits board"),
    ]
    trace = SkillTraces.build(
        action=_action(skill="PickWiper"),
        atoms_before=START,
        atoms_after=START,
        error="PickWiper: pick_wiper: no candidate planned (17 tried)",
        controller_steps=0,
        notes=notes,
    )
    assert trace.outcome == "plan_failure"
    assert [note.kind for note in trace.notes] == ["no_plan", "plan_blocked"]
    assert trace.missing_effects == [
        "HoldingWiper(robot,wiper_0)",
        "WiperAway(wiper_0)",
        "not HandEmpty(robot)",
        "not WiperHome(wiper_0)",
    ]


def test_a_failure_after_the_robot_moved_is_an_execution_failure() -> None:
    trace = SkillTraces.build(
        action=_action(skill="PickCubeFromShelf", cube=2),
        atoms_before=START,
        atoms_after=START,
        error="pick_cube_from_shelf: cube_2 did not come with the hand",
        controller_steps=812,
        notes=[],
    )
    assert trace.outcome == "execution_failure"
    assert trace.objects == ["robot", "cube_2"]


def test_no_error_but_a_promised_effect_missing_is_not_a_success() -> None:
    after = START - {("OnShelf", ("cube_0",))} | {("InBox", ("cube_0", "box_0"))}
    holding = _picked_wiper()
    trace = SkillTraces.build(
        action=_action(skill="SweepShelf", height=0.005, speed=0.05),
        atoms_before=holding,
        atoms_after=after | {("HoldingWiper", ("robot", "wiper_0"))},
        error=None,
        controller_steps=2852,
        notes=[],
    )
    # Every cube on the shelf before the sweep is promised InBox (a conditional effect).
    assert trace.outcome == "effects_missing"
    assert "InBox(cube_1,box_0)" in trace.missing_effects
    assert "InBox(cube_0,box_0)" not in trace.missing_effects
    assert trace.params == [0.005, 0.05]


def test_an_interrupted_skill_is_its_own_outcome() -> None:
    trace = SkillTraces.build(
        action=_action(skill="ReturnWiper"),
        atoms_before=_picked_wiper(),
        atoms_after=_picked_wiper(),
        error="interrupted: practice_step_budget",
        controller_steps=40,
        notes=[],
    )
    assert trace.outcome == "interrupted"


def test_the_trace_stays_compact_however_many_candidates_were_rejected() -> None:
    notes = [
        SkillNote(stage="return_place", kind="no_plan", params=[0.1] * 3, reason="x" * 5000)
        for _ in range(10 * MAX_NOTES)
    ]
    trace = SkillTraces.build(
        action=_action(skill="ReturnCubeToShelfFromBox", cube=4),
        atoms_before=START,
        atoms_after=START,
        error="e" * 5000,
        controller_steps=0,
        notes=notes,
    )
    assert len(trace.notes) == MAX_NOTES
    assert trace.notes_dropped == 9 * MAX_NOTES
    assert all(len(note.reason) <= MAX_REASON_CHARS for note in trace.notes)
    assert trace.error is not None and len(trace.error) <= MAX_REASON_CHARS
    assert len(json.dumps(trace.model_dump(mode="json"))) < 20_000


def test_a_noop_and_an_unknown_skill_are_traced_without_effects() -> None:
    noop = SkillTraces.build(
        action=np.array([-1.0, 0.0, 0.0, 0.0]),
        atoms_before=START,
        atoms_after=START,
        error=None,
        controller_steps=0,
        notes=[],
    )
    assert (noop.skill, noop.outcome) == ("noop", "success")
    unknown = SkillTraces.build(
        action=np.array([42.0, 0.0, 0.0, 0.0]),
        atoms_before=START,
        atoms_after=START,
        error="unknown skill id 42",
        controller_steps=0,
        notes=[],
    )
    assert (unknown.skill, unknown.outcome) == ("unknown:42", "plan_failure")


def test_an_overflowing_trace_keeps_the_candidates_that_decided_the_outcome() -> None:
    # The last notes say how the attempt ended (the executed candidate); a cap that kept
    # only the first ones dropped exactly those.
    notes = [
        SkillNote(stage="pick_cube_from_shelf", kind="no_plan", params=[float(i)], reason="r")
        for i in range(3 * MAX_NOTES)
    ]
    notes.append(SkillNote(stage="pick_cube_from_shelf", kind="executed", reason="700 steps"))
    trace = SkillTraces.build(
        action=_action(skill="PickCubeFromShelf", cube=1),
        atoms_before=START,
        atoms_after=START,
        error=None,
        controller_steps=700,
        notes=notes,
    )
    assert len(trace.notes) == MAX_NOTES
    assert trace.notes[-1].kind == "executed"
    assert trace.notes[0].params == [0.0]
    assert trace.notes_dropped == len(notes) - MAX_NOTES
