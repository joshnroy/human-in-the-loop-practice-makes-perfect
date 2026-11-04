import json
from pathlib import Path

from analysis.kitchen_recycling3d_cost_sweep import (
    PairedTests,
    RunSummary,
    SweepData,
    SweepExtract,
)


def _write_run(*, run_dir: Path, events: list[dict], evaluations: dict[int, int]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "step_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events) + "{not json\n"
    )
    for index, solved in evaluations.items():
        (run_dir / "evaluations" / str(index)).mkdir(parents=True)
        (run_dir / "evaluations" / str(index) / "results.json").write_text(
            json.dumps({"complete": True, "num_solved": solved, "num_total": 10})
        )
    (run_dir / "status.json").write_text(json.dumps({"phase": "running", "practice_steps": 9}))


def test_extract_reads_evaluations_resets_dumps_and_stops(*, tmp_path: Path) -> None:
    events = [
        {"event": "measurement", "index": 0, "practice_steps": 0},
        {
            "event": "human",
            "start_step": 1,
            "end_step": 2,
            "skill": "ResetCubeBinSameSide",
            "cost": 5.0,
        },
        {"event": "robot", "start_step": 2, "end_step": 4, "label": "PourCubeOutOfBin(cube)"},
        {"event": "robot", "start_step": 4, "end_step": 5, "label": "PickCube(cube)"},
        {"event": "stop", "start_step": 5, "end_step": 6},
        {"event": "stop", "start_step": 6, "end_step": 7},
        {"event": "measurement", "index": 1, "practice_steps": 7},
    ]
    _write_run(
        run_dir=tmp_path / "results/reset-cost-5/curtis/1",
        events=events,
        evaluations={0: 1, 1: 4, 2: 9},  # index 2 has no measurement event: dropped
    )
    run = SweepExtract.extract_root(root=tmp_path)["5/curtis/1"]
    assert run["evaluations"] == [[0, 1, 10], [7, 4, 10]]
    assert run["resets"] == [[1, "ResetCubeBinSameSide", 5.0]]
    assert run["dumps"] == [2]
    assert run["stop_steps"] == [5, 6]
    assert run["practice_steps"] == 9
    assert run["phase"] == "running"


def test_merge_takes_planner_rows_from_the_rerun_and_ees_from_the_first_launch() -> None:
    first = {"100/ees/0": {"src": "v10"}, "5/curtis/0": {"src": "v10"}}
    rerun = {"100/ees/0": {"src": "v10b"}, "100/curtis/0": {"src": "v10b"}}
    merged = SweepData.merge(first=first, rerun=rerun)
    assert merged == {
        "100/ees/0": {"src": "v10"},
        "5/curtis/0": {"src": "v10"},
        "100/curtis/0": {"src": "v10b"},
    }


def test_stop_streaks_counts_consecutive_steps_inclusively() -> None:
    assert RunSummary.stop_streaks(stop_steps=[3, 10, 11, 12, 20, 21]) == [1, 3, 2]
    assert RunSummary.stop_streaks(stop_steps=[]) == []


def test_summary_of_a_run_cancelled_mid_streak() -> None:
    run = {
        "evaluations": [[0, 1, 10], [3200, 2, 10]],
        "resets": [[5, "ResetCubeBinSameSide", 1e4], [9, "ResetCubeBinOppositeSide", 1e4]],
        "dumps": [],
        "stop_steps": list(range(100, 140)),
        "practice_steps": 140,
        "phase": "running",
    }
    summary = RunSummary.summarize(run=run, cancelled=True)
    assert summary["last_solved"] == 2
    assert summary["mean_last_3"] == 1.5
    assert summary["resets_same"] == 1
    assert summary["resets_opposite"] == 1
    assert summary["human_cost"] == 2e4
    assert summary["stops"] == 40
    assert summary["final_stop_streak"] == 40
    assert summary["end"] == "ended by repeated STOP"


def test_three_pairs_cannot_reach_below_a_quarter() -> None:
    result = PairedTests.wilcoxon(first=[1.0, 2.0, 3.0], second=[2.0, 4.0, 6.0])
    assert result == {"n": 3, "n_nonzero": 3, "p": 0.25}
    assert PairedTests.wilcoxon(first=[1.0, 1.0, 1.0], second=[1.0, 1.0, 1.0]) == {
        "n": 3,
        "n_nonzero": 0,
        "p": None,
    }


def test_a_zero_difference_is_dropped_never_approximated() -> None:
    # Two non-zero pairs, same sign: the exact floor is 2/4 = 0.5, not a normal approximation.
    result = PairedTests.wilcoxon(first=[10.0, 9.0, 8.0], second=[8.0, 5.0, 8.0])
    assert result == {"n": 3, "n_nonzero": 2, "p": 0.5}
    # Mixed signs with tied magnitudes: average ranks, exact enumeration.
    assert PairedTests.wilcoxon(first=[1.0, 3.0, 5.0], second=[2.0, 2.0, 4.0])["p"] == 1.0
