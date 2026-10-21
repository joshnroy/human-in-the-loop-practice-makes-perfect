"""Parsing each source of the real-run dashboard.

`fixtures/run9/` is the 2026-10-20 pilot's run directory (`tidybot-real/pilot-20261020/
run9`), copied whole except `pomdp_decisions.jsonl`, which keeps only the fields the
dashboard reads (the full file is 10 MB of belief dumps), and `config_snapshot.json`,
which keeps five flags. `fixtures/skill_calls.jsonl` is that session's skill server log
(`tidybot-real/raw/2026-10-20/18-10-36`), unchanged. run9 predates the session log, so
its lines here are written in `RealSessionLog`'s format by hand.
"""

import json
from pathlib import Path

import pytest

from analysis.real_dashboard.sources import (
    DecisionLog,
    Evaluations,
    SessionLog,
    SkillCalls,
    StepEvents,
)

FIXTURES = Path(__file__).parent / "fixtures"
RUN9 = FIXTURES / "run9"


def _records(*, path: Path) -> list[tuple[int, dict]]:
    lines = path.read_text().splitlines()
    return [(number, json.loads(line)) for number, line in enumerate(lines, start=1)]


def test_run9_step_events_parse_into_robot_actions_resets_and_one_measurement():
    events = StepEvents.parse(records=_records(path=RUN9 / "step_events.jsonl"))
    assert [action.skill for action in events.robot_actions].count("PickCube") == 8
    tosses = events.tosses()
    assert len(tosses) == 7
    first = tosses[0]
    assert first.toss is not None
    assert first.toss.standoff == pytest.approx(1.4038102717139824)
    assert first.toss.speed_deg_s == pytest.approx(135.10115623487184)
    assert first.toss.release_ms == pytest.approx(530.3852052752129)
    assert first.toss.bin_side == "opposite_side"
    assert tosses[1].toss is not None
    assert tosses[1].toss.bin_side == "robot_side"
    assert len(events.human_resets) == 4
    assert {reset.destination for reset in events.human_resets} == {"robot_side"}
    assert len(events.measurements) == 1
    assert events.practice_steps == 1586
    assert events.cycle == 0
    assert events.practice_end == "practice_step_budget"
    assert events.discarded_lines == 0
    assert events.session_seconds == pytest.approx(1188.6193530530436)


def test_a_resume_discards_the_interrupted_lines_and_starts_a_new_session():
    records = [
        (1, {"event": "session_start", "cycle": 0, "practice_steps": 0, "elapsed_seconds": 5.0}),
        (
            2,
            {
                "event": "robot",
                "cycle": 0,
                "label": "PickCube(robot, cube_0)",
                "action": [0.0],
                "start_step": 0,
                "end_step": 119,
                "skill_error": None,
                "elapsed_seconds": 60.0,
            },
        ),
        (
            3,
            {
                "event": "learning_update",
                "cycle": 0,
                "practice_steps": 119,
                "elapsed_seconds": 61.0,
            },
        ),
        (4, {"event": "session_start", "cycle": 1, "practice_steps": 119, "elapsed_seconds": 62.0}),
        (
            5,
            {
                "event": "robot",
                "cycle": 1,
                "label": "PickCube(robot, cube_0)",
                "action": [0.0],
                "start_step": 119,
                "end_step": 238,
                "skill_error": None,
                "elapsed_seconds": 100.0,
            },
        ),
        (
            6,
            {
                "event": "cycle_discarded",
                "cycle": 1,
                "first_line": 4,
                "last_line": 5,
                "practice_steps": 119,
                "elapsed_seconds": 3.0,
            },
        ),
        (
            7,
            {
                "event": "resume",
                "cycle": 1,
                "practice_steps": 119,
                "learning_updates": 1,
                "interrupted_first_line": 4,
                "interrupted_last_line": 5,
                "elapsed_seconds": 3.5,
            },
        ),
        (8, {"event": "practice_scene_restored", "cycle": 1, "elapsed_seconds": 30.0}),
        (9, {"event": "something_new", "elapsed_seconds": 31.0}),
    ]
    events = StepEvents.parse(records=records)
    assert [action.end_step for action in events.robot_actions] == [119]
    assert events.practice_steps == 119
    assert events.discarded_lines == 2
    assert events.cycle == 1
    assert events.learning_updates == 1
    assert [note.event for note in events.notes] == [
        "cycle_discarded",
        "resume",
        "practice_scene_restored",
        "something_new",
    ]
    # Two processes: 100 s before the interruption, 31 s since the resume.
    assert events.session_seconds == pytest.approx(131.0)


def test_run9_evaluations_read_one_finished_evaluation(*, tmp_path):
    evaluations = Evaluations.read(run_dir=RUN9)
    assert len(evaluations) == 1
    only = evaluations[0]
    assert (only.index, only.practice_steps, only.learning_updates) == (0, 1586, 0)
    assert (only.num_solved, only.num_total, only.complete) == (0, 1, True)


def test_an_evaluation_in_progress_is_read_from_its_progress_file(*, tmp_path):
    (tmp_path / "step_evaluations.json").write_text((RUN9 / "step_evaluations.json").read_text())
    running = tmp_path / "evaluations" / "0001"
    running.mkdir(parents=True)
    tasks = [{"task": index, "solved": index != 1} for index in range(3)]
    (running / "progress.json").write_text(json.dumps({"tasks": tasks}))
    (tmp_path / "measurements.json").write_text(
        json.dumps([
            {"index": 0, "practice_steps": 1586},
            {"index": 1, "practice_steps": 3000, "learning_updates": 2},
        ])
    )
    evaluations = Evaluations.read(run_dir=tmp_path)
    assert [evaluation.index for evaluation in evaluations] == [0, 1]
    last = evaluations[1]
    assert not last.complete
    assert (last.practice_steps, last.learning_updates) == (3000, 2)
    assert (last.num_solved, last.num_total) == (2, 3)


def test_a_finished_evaluation_is_read_from_its_results_before_the_summary_exists(*, tmp_path):
    # Run 10 (2026-10-21) had evaluations/0000/results.json and no step_evaluations.json.
    folder = tmp_path / "evaluations" / "0000"
    folder.mkdir(parents=True)
    tasks = [{"task": index, "solved": index != 3} for index in range(5)]
    (folder / "results.json").write_text(json.dumps({"complete": True, "tasks": tasks}))
    (folder / "progress.json").write_text(json.dumps({"tasks": tasks[:2]}))
    (tmp_path / "measurements.json").write_text(
        json.dumps([{"index": 0, "practice_steps": 1883, "learning_updates": 1}])
    )
    (only,) = Evaluations.read(run_dir=tmp_path)
    assert (only.num_solved, only.num_total, only.complete) == (4, 5, True)
    assert (only.practice_steps, only.learning_updates) == (1883, 1)


def test_a_half_written_evaluations_file_reads_as_none(*, tmp_path):
    (tmp_path / "step_evaluations.json").write_text('[{"index": 0, "practi')
    assert Evaluations.read(run_dir=tmp_path) is None


def test_run9_decision_log_keeps_competences_and_outcomes():
    log = DecisionLog()
    for line in (RUN9 / "pomdp_decisions.jsonl").read_text().splitlines():
        log.ingest(line=line)
    assert len(log.competences) == 19
    last = log.competences[-1]
    assert last.values["PickCube"] == pytest.approx(0.9176025527220768)
    assert last.values["MoveToTossLocationAndToss"] == pytest.approx(0.3000176137169695)
    assert len(log.outcomes) == 18
    tosses = [outcome for outcome in log.outcomes if outcome.skill == "MoveToTossLocationAndToss"]
    assert [outcome.success for outcome in tosses] == [False] * 6 + [True]


def test_run9_skill_calls_parse_tosses_picks_and_observations():
    calls = SkillCalls()
    for line in (FIXTURES / "skill_calls.jsonl").read_text().splitlines():
        calls.ingest(line=line)
    assert len(calls.calls) == 38
    tosses = [call for call in calls.calls if call.call == "toss"]
    assert len(tosses) == 7
    assert tosses[0].toss is not None
    assert tosses[0].toss.landed == pytest.approx((0.20350115761629423, 1.458790924299131))
    assert tosses[0].toss.in_bin is False
    assert tosses[0].toss.inside_bin is True
    assert tosses[5].toss is not None
    assert tosses[5].toss.landed is None
    assert tosses[6].toss is not None
    assert tosses[6].toss.in_bin is True
    picks = [call for call in calls.calls if call.call == "pick"]
    assert [call.held for call in picks] == [True] * 8
    observe = calls.last_observation()
    assert observe is not None
    assert observe.bin is not None
    assert observe.robot is not None


TOSS = "MoveToTossLocationAndToss(standoff 1.40 m, 135 deg/s, release 530 ms)"
SESSION = f"""\
2026-10-21 14:00:01.123 cycle 3 [####------] action 8/20 · steps 4200/85000
2026-10-21 14:00:01.200 planning the next skill ...
2026-10-21 14:00:03.500 {TOSS} -- run it? [y/n]
2026-10-21 14:00:05.000 answer: 'y'
2026-10-21 14:00:05.010 running {TOSS} ...
"""


def test_the_session_log_names_the_running_skill_and_no_prompt_once_answered():
    log = SessionLog()
    for line in SESSION.splitlines():
        log.ingest(line=line)
    state = log.state()
    assert state.prompt is None
    assert state.progress == "cycle 3 [####------] action 8/20 · steps 4200/85000"
    assert state.running == TOSS
    assert state.last_time == "2026-10-21 14:00:05.010"


def test_the_session_log_shows_an_unanswered_prompt_and_the_line_before_it():
    log = SessionLog()
    lines = [
        *SESSION.splitlines(),
        f"2026-10-21 14:00:50.000 {TOSS}: ok (45 s); observing ...",
        "2026-10-21 14:01:10.000 reset_cube_and_bin: FAR (bin at the far spot, across the "
        "tape); cube anywhere on our side, on a face; do not move the robot",
        "2026-10-21 14:01:10.100 Are the cube and the bin in place? [y/n]",
    ]
    for line in lines:
        log.ingest(line=line)
    state = log.state()
    assert state.running is None
    assert state.prompt == "Are the cube and the bin in place? [y/n]"
    assert state.prompt_since == "2026-10-21 14:01:10.100"
    assert state.prompt_context is not None
    assert state.prompt_context.startswith("reset_cube_and_bin: FAR")
    log.ingest(line="2026-10-21 14:01:40.000 answer: 'y'")
    assert log.state().prompt is None


def test_a_session_log_line_without_a_timestamp_continues_the_previous_one():
    log = SessionLog()
    log.ingest(line="2026-10-21 14:00:01.123 first")
    log.ingest(line="  more of it")
    assert log.state().tail == ["2026-10-21 14:00:01.123 first\n  more of it"]
