"""Human corrections beside a run (`--annotations`): shown, never written over the
recorded numbers. The annotation is run 10's first, from
`tidybot-real/full-run-20261021.annotations.jsonl`, with `why` shortened."""

import json
import shutil
from pathlib import Path

from analysis.real_dashboard.dashboard import RealRunDashboard

RUN9 = Path(__file__).parent / "fixtures" / "run9"
ANNOTATION = {
    "time": "2026-10-21T11:01",
    "by": "Josh (confirmed by eye at the reset); recorded by Claude",
    "evaluation_index": 0,
    "after_cycle": 0,
    "practice_steps": 1883,
    "task": 3,
    "recorded": {"solved": False, "reason": "no_plan"},
    "corrected": {"solved": True},
    "evaluation_recorded": "4/5",
    "evaluation_corrected": "5/5",
    "why": "The toss's tracker logged landed (0.073, 1.383) in_bin=true.",
    "raw_numbers_untouched": True,
}


def _run(*, tmp_path: Path) -> Path:
    run = tmp_path / "run"
    shutil.copytree(RUN9, run)
    records = [
        {
            "index": index,
            "practice_steps": steps,
            "learning_updates": index,
            "num_solved": solved,
            "num_total": 5,
            "complete": True,
        }
        for index, steps, solved in ((0, 1883, 4), (1, 4000, 2))
    ]
    (run / "step_evaluations.json").write_text(json.dumps(records))
    return run


def test_an_annotation_adds_a_corrected_value_and_never_changes_the_recorded_one(*, tmp_path):
    run = _run(tmp_path=tmp_path)
    annotations = tmp_path / "run.annotations.jsonl"
    unknown = {**ANNOTATION, "evaluation_index": 7}
    annotations.write_text(f"{json.dumps(ANNOTATION)}\nnot json\n{json.dumps(unknown)}\n")
    curve = RealRunDashboard(run_dir=run, annotations=annotations).refresh().eval_curve
    first, second = curve.points
    assert (first.solved, first.label) == (4, "4/5")
    assert (first.corrected, first.corrected_label) == (5, "5/5")
    assert first.corrections == [
        "task 3: recorded unsolved (no_plan), corrected to solved. "
        f"{ANNOTATION['why']} ({ANNOTATION['by']}, {ANNOTATION['time']})"
    ]
    assert (second.solved, second.corrected, second.corrections) == (2, None, [])
    assert curve.corrected_legend == "corrected (human-confirmed), n=1"


def test_a_later_annotation_of_the_same_task_replaces_the_earlier(*, tmp_path):
    run = _run(tmp_path=tmp_path)
    annotations = tmp_path / "run.annotations.jsonl"
    undone = {**ANNOTATION, "corrected": {"solved": False}, "why": "looked again"}
    annotations.write_text(f"{json.dumps(ANNOTATION)}\n{json.dumps(undone)}\n")
    point = RealRunDashboard(run_dir=run, annotations=annotations).refresh().eval_curve.points[0]
    assert (point.solved, point.corrected) == (4, 4)
    assert len(point.corrections) == 1


def test_without_annotations_the_curve_has_no_corrected_legend(*, tmp_path):
    curve = RealRunDashboard(run_dir=_run(tmp_path=tmp_path)).refresh().eval_curve
    assert curve.corrected_legend is None
    assert all(point.corrected is None for point in curve.points)
    missing = RealRunDashboard(
        run_dir=_run(tmp_path=tmp_path / "x"), annotations=tmp_path / "none.jsonl"
    )
    assert missing.refresh().eval_curve.corrected_legend is None
