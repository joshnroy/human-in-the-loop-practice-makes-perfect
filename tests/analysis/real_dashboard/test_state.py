"""What the dashboard shows, built from run9's files (see `test_sources.py`)."""

import json
import shutil
from pathlib import Path

import pytest

from analysis.real_dashboard.dashboard import RealRunDashboard

FIXTURES = Path(__file__).parent / "fixtures"
RUN9 = FIXTURES / "run9"


@pytest.fixture
def run9_state():
    dashboard = RealRunDashboard(run_dir=RUN9, skill_calls=FIXTURES / "skill_calls.jsonl")
    return dashboard.refresh()


def test_run9_status(*, run9_state):
    status = run9_state.status
    assert (status.cycle, status.practice_steps, status.step_budget) == (0, 1586, 1500)
    assert status.human_resets == {"SAME": 4, "FAR": 0}
    assert status.phase == "complete"
    assert status.last_skill is not None
    assert status.last_skill.startswith("PickCube")
    assert status.session_seconds == pytest.approx(1188.6193530530436)
    assert status.competences["PickCube"] == pytest.approx(0.9176025527220768)


def test_run9_eval_curve_is_one_marker_at_the_practice_steps_it_was_taken(*, run9_state):
    curve = run9_state.eval_curve
    assert [(point.practice_steps, point.cycle) for point in curve.points] == [(1586, 0)]
    assert [(point.solved, point.total) for point in curve.points] == [(0, 1)]
    assert curve.points[0].label == "0/1"
    # pomdp + local_trend is ours, Model B; the human reset was available.
    assert curve.method_label == "ours, Model B"
    assert curve.colour == "#0072B2"
    assert curve.dashed is False
    assert curve.legend == "ours, Model B — solved per evaluation, n=1"


def test_the_eval_curve_counts_five_task_evaluations_and_skips_none_complete(*, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RUN9, run)
    records = [
        {
            "index": index,
            "practice_steps": 2000 * (index + 1),
            "learning_updates": index + 1,
            "num_solved": solved,
            "num_total": 5,
            "complete": True,
        }
        for index, solved in enumerate([0, 2, 5])
    ]
    (run / "step_evaluations.json").write_text(json.dumps(records))
    curve = RealRunDashboard(run_dir=run).refresh().eval_curve
    assert [point.label for point in curve.points] == ["0/5", "2/5", "5/5"]
    assert [point.cycle for point in curve.points] == [1, 2, 3]
    assert curve.legend == "ours, Model B — solved per evaluation, n=3"


def test_run9_tosses_are_matched_to_the_skill_server_by_their_parameters(*, run9_state):
    tosses = run9_state.tosses
    assert [toss.in_bin for toss in tosses.points] == [False] * 6 + [True]
    assert tosses.outcome_source == "skill_calls"
    assert tosses.in_bin_label == "1/7"
    assert tosses.rolling_label == "1/7"
    assert tosses.points[0].landed == pytest.approx((0.20350115761629423, 1.458790924299131))
    assert tosses.points[0].spot == "FAR"
    assert tosses.points[1].spot == "SAME"
    assert tosses.points[5].landed is None


def test_without_the_skill_server_log_toss_outcomes_come_from_the_decision_log():
    state = RealRunDashboard(run_dir=RUN9).refresh()
    tosses = state.tosses
    assert tosses.outcome_source == "pomdp_decisions"
    assert [toss.in_bin for toss in tosses.points] == [False] * 6 + [True]
    assert all(toss.landed is None for toss in tosses.points)
    assert state.room.landings == []


def test_run9_room_map_has_both_bin_spots_the_tape_and_the_landings(*, run9_state):
    room = run9_state.room
    assert {spot.name for spot in room.bin_spots} == {"SAME", "FAR"}
    # The room fact `y = 0.349 + 0.0267 x`, as `ROOM_FRAME` carries it.
    assert room.tape_intercept == pytest.approx(0.349, abs=1e-3)
    assert room.tape_slope == pytest.approx(0.0267, abs=1e-4)
    # Six of the seven tosses were seen landing.
    assert len(room.landings) == 6
    assert sum(landing.in_bin for landing in room.landings) == 1
    assert room.observed_bin is not None


def test_a_refresh_picks_up_what_the_run_appended_since_the_last(*, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(RUN9, run)
    calls = tmp_path / "skill_calls.jsonl"
    shutil.copy(FIXTURES / "skill_calls.jsonl", calls)
    dashboard = RealRunDashboard(run_dir=run, skill_calls=calls)
    assert len(dashboard.refresh().tosses.points) == 7
    toss = {
        "event": "robot",
        "cycle": 1,
        "label": "MoveToTossLocationAndToss(robot, bin_0, cube_0, cuboid_barrier, "
        "opposite_side), params=[1.5, 120.0, 600.0]",
        "action": [1.0, 1.5, 3.1, 120.0, 600.0],
        "start_step": 1586,
        "end_step": 1676,
        "skill_error": None,
        "timestamp": "2026-10-20T22:40:00+00:00",
        "elapsed_seconds": 1800.0,
    }
    reply = {"tossed": True, "landed": [0.25, 1.40], "in_bin": True, "inside_bin": True}
    call = {
        "i": 39,
        "call": "toss",
        "args": {"standoff": 1.5, "speed": 120.0, "release_ms": 600.0},
        "reply": reply,
        "duration_s": 40.0,
    }
    with (run / "step_events.jsonl").open("a") as stream:
        stream.write(json.dumps(toss) + "\n")
    with calls.open("a") as stream:
        stream.write(json.dumps(call) + "\n")
    state = dashboard.refresh()
    assert len(state.tosses.points) == 8
    assert state.tosses.in_bin_label == "2/8"
    assert state.tosses.points[-1].spot == "FAR"
    assert state.status.practice_steps == 1676
    assert state.status.cycle == 1


def test_run9_skill_timing_and_pick_success(*, run9_state):
    timing = run9_state.timing
    by_call = {row.name: row for row in timing.skill_server}
    assert by_call["pick"].n == 8
    assert by_call["toss"].n == 7
    assert by_call["observe"].n == 21
    assert by_call["toss"].median_s == pytest.approx(45.0, abs=6.0)
    assert timing.pick_label == "8/8"
    assert timing.server_pick_label == "8/8"
    practice = {row.name: row for row in timing.practice}
    assert practice["PickCube"].n == 8
    assert practice["MoveToTossLocationAndToss"].n == 7
