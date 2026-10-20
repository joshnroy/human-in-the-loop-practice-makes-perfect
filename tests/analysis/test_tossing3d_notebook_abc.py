import json
from pathlib import Path

import pytest
import zstandard

from analysis.tossing3d_notebook_abc import TOSS, RunSummary
from hitl_pmp.results_writer.jsonl_log import JsonlLog


def _run_dir(*, root: Path, layout: str) -> Path:
    root.mkdir()
    (root / "progress.jsonl").write_text(
        "".join(json.dumps({"num_solved": solved}) + "\n" for solved in (0, 3))
    )
    (root / "stats.json").write_text(
        json.dumps({
            "evaluations": [[0, 0, 4], [10, 3, 4]],
            "practice_outcomes_per_cycle": [
                {TOSS: {"num_attempts": 2, "num_successes": 1}},
            ],
        })
    )
    history = [
        {"filtered_competence": 0.4, "smoothed_competence": 0.5, "filtered_learning_rate": 0.1}
    ]
    events = [
        {"event": "outcome", "skill": TOSS, "competence_conditioned": True},
        {"event": "refit", "beliefs": {TOSS: {"total_training_examples": 2}}},
        {"event": "smoothing", "history": {TOSS: history}},
    ]
    payload = "".join(json.dumps(event) + "\n" for event in events).encode()
    if layout == "zst":
        (root / "pomdp_decisions.jsonl.zst").write_bytes(
            zstandard.ZstdCompressor().compress(payload)
        )
    elif layout == "zst-frames":
        # What a run writes live: one frame per record.
        for event in events:
            JsonlLog.append(path=root / "pomdp_decisions.jsonl.zst", line=json.dumps(event) + "\n")
    else:
        (root / "pomdp_decisions.jsonl").write_bytes(payload)
    return root


@pytest.mark.parametrize("layout", ["plain", "zst", "zst-frames"])
def test_extract_reads_the_decision_log_plain_or_compressed(*, tmp_path: Path, layout: str) -> None:
    summary = RunSummary.extract(run_dir=_run_dir(root=tmp_path / "run", layout=layout))
    assert summary["clock"] == [2]
    assert summary["smoothed_competence"] == [0.5]
    assert summary["toss_outcomes_conditioned"] == [1, 1]


def test_a_multi_frame_log_extracts_exactly_as_the_plain_log(*, tmp_path: Path) -> None:
    plain = RunSummary.extract(run_dir=_run_dir(root=tmp_path / "plain", layout="plain"))
    frames = RunSummary.extract(run_dir=_run_dir(root=tmp_path / "frames", layout="zst-frames"))
    assert frames == plain
