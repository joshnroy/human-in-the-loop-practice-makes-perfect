"""Prior cycle calibration must exclude held-out physics and charge human duration."""

import json
from pathlib import Path

import pytest

from analysis.calibrate_step_protocol import PriorCalibration


def test_aligns_evaluation_blocks_before_counting_practice(*, tmp_path: Path):
    stats = dict(
        practice_session_ends=[dict(actions_executed=3)],
        evaluations=[[0, []], [2, []]],
        num_human_interventions_recorded=1,
    )
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    (tmp_path / "progress.jsonl").write_text(
        "\n".join(json.dumps(dict(elapsed_seconds=t)) for t in [0, 20]) + "\n"
    )
    events = []
    for name, time, ticks in [
        ("PickCube", 1, 2),
        ("PickCube", 3, 3),
        ("MoveToTossLocationAndToss", 6, 4),
        ("PickCube", 21, 2),
    ]:
        events.append(dict(kind="skill", name=name, params=[], elapsed_seconds=time))
        events.extend(dict(kind="tick") for _ in range(ticks))
    (tmp_path / "tossing3d_state_log.jsonl").write_text("\n".join(map(json.dumps, events)) + "\n")
    episodes = [
        dict(
            checkpoint=i,
            task_index=0,
            elapsed_seconds=t,
            action=[0],
            action_label="PickCube(robot)",
            solved=False,
        )
        for i, t in [(0, 2), (1, 22)]
    ]
    (tmp_path / "episode_traces.jsonl").write_text("\n".join(map(json.dumps, episodes)) + "\n")
    result = PriorCalibration.analyze(run=tmp_path, machine="fixture", method="ees", seed=0)
    assert result["sessions"][0]["robot_steps"] == 7
    assert result["sessions"][0]["human"] == 1
    assert result["sessions"][0]["counted_steps"] == 8
    assert result["evaluation_summary"]["total"] == 4
    stats["num_human_interventions_recorded"] = 2
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    with pytest.raises(AssertionError):
        PriorCalibration.analyze(run=tmp_path, machine="fixture", method="ees", seed=0)
