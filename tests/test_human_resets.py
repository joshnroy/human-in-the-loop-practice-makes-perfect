import argparse
import json
from pathlib import Path

from hitl_pmp.human_resets import HUMAN_RESETS_FILENAME, HumanReset, HumanResetLog


def _reset(*, cycle: int = 3, destination: str | None = "robot_side") -> HumanReset:
    return HumanReset(
        cycle=cycle,
        step_index=7,
        num_online_transitions=41,
        destination=destination,
        skill="ask_for_reset_cube_bin_only",
        cost=1.5,
    )


def test_no_output_dir_means_no_log() -> None:
    assert HumanResetLog.open_if_requested(args=argparse.Namespace(output_dir=None)) is None


def test_the_log_exists_and_is_empty_before_any_reset(*, tmp_path: Path) -> None:
    """Created at open, unlike episode_traces.jsonl: an empty file is the measurement
    "no human reset happened", and a missing one must keep meaning "not recorded"."""
    log = HumanResetLog.open_if_requested(args=argparse.Namespace(output_dir=tmp_path))
    assert log is not None
    assert (tmp_path / HUMAN_RESETS_FILENAME).read_text() == ""


def test_each_reset_is_one_flushed_json_line(*, tmp_path: Path) -> None:
    log = HumanResetLog.open(output_path=tmp_path / HUMAN_RESETS_FILENAME)
    log.record(reset=_reset(cycle=0))
    # Readable before close: a crashed run keeps every reset it got to.
    lines = (tmp_path / HUMAN_RESETS_FILENAME).read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert {
        key: record[key]
        for key in (
            "cycle",
            "step_index",
            "num_online_transitions",
            "destination",
            "skill",
            "cost",
        )
    } == {
        "cycle": 0,
        "step_index": 7,
        "num_online_transitions": 41,
        "destination": "robot_side",
        "skill": "ask_for_reset_cube_bin_only",
        "cost": 1.5,
    }
    assert "timestamp" in record
    assert "elapsed_seconds" in record


def test_read_round_trips_every_record_in_order(*, tmp_path: Path) -> None:
    log = HumanResetLog.open(output_path=tmp_path / HUMAN_RESETS_FILENAME)
    written = [_reset(cycle=0), _reset(cycle=1, destination="opposite_side"), _reset(cycle=1)]
    for reset in written:
        log.record(reset=reset)
    assert HumanResetLog.read(path=tmp_path / HUMAN_RESETS_FILENAME) == written


def test_reopening_truncates_a_previous_runs_log(*, tmp_path: Path) -> None:
    """stats.json is overwritten by a re-run into the same directory, so this must be
    too, or the two would disagree."""
    path = tmp_path / HUMAN_RESETS_FILENAME
    HumanResetLog.open(output_path=path).record(reset=_reset())
    HumanResetLog.open(output_path=path)
    assert path.read_text() == ""


def test_per_cycle_counts_split_by_destination() -> None:
    records = [
        _reset(cycle=0, destination="robot_side"),
        _reset(cycle=0, destination="opposite_side"),
        _reset(cycle=2, destination="robot_side"),
        _reset(cycle=2, destination="robot_side"),
    ]
    assert HumanResetLog.counts_by_cycle(records=records, num_cycles=3) == [
        {"robot_side": 1, "opposite_side": 1},
        {},
        {"robot_side": 2},
    ]
