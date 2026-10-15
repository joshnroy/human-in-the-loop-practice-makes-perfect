"""The Tossing3D pick benchmark's report.json and page, from a synthetic output dir."""

from pathlib import Path

import pytest

from analysis.tossing3d_pick_benchmark import FIGURE, PickBenchmarkPage
from hitl_pmp.environments.tossing3d.pick_benchmark.report import PickReports
from hitl_pmp.environments.tossing3d.pick_benchmark.states import PickStateSet
from hitl_pmp.environments.tossing3d.pick_benchmark.types import PickResult, PickState


def _setup(*, tmp_path: Path) -> tuple[Path, Path]:
    states = [
        PickState(
            state_id=f"s{i}",
            family=family,
            seed=0,
            layout="barrier",
            scene_fingerprint="scene",
            plain={},
            atoms=[],
            skill=skill,
            in_target=in_target,
        )
        for i, (family, skill, in_target) in enumerate([
            ("start", "PickCube", True),
            ("harvested: in bin near wall", "PickCubeFromBin", True),
            ("tipped_bin", "PickCube", False),
        ])
    ]
    states_path = tmp_path / "states.jsonl"
    PickStateSet.append(path=states_path, states=states)
    out = tmp_path / "out"
    out.mkdir()
    labels = ["success", "no_plan", "no_plan"]
    rows = [
        PickResult(
            case_id=f"{s.state_id}-{s.skill}",
            state_id=s.state_id,
            family=s.family,
            skill=s.skill,
            label=label,
            in_target=s.in_target,
            dry_run_plannable=label == "success",
            pybullet="planned" if label == "success" else "no_plan",
            video=None if label == "success" else f"runs/{s.state_id}/video.mp4",
        ).model_dump_json()
        for s, label in zip(states, labels, strict=True)
    ]
    (out / "results-00.jsonl").write_text("\n".join(rows) + "\n")
    return states_path, out


def test_the_page_reports_counts_and_links_every_failure(*, tmp_path: Path) -> None:
    states_path, out = _setup(tmp_path=tmp_path)
    report = PickReports.build(out=out, states_path=states_path, implementation="current")
    verdicts = {v.skill: v for v in report.skills}
    assert (verdicts["PickCube"].successes, verdicts["PickCube"].total) == (1, 1)
    assert verdicts["PickCube"].outside_operator == 1
    assert (verdicts["PickCubeFromBin"].successes, verdicts["PickCubeFromBin"].total) == (0, 1)
    assert not report.passed
    text = PickBenchmarkPage.render(out=out).read_text()
    assert "0/1" in text and "FAILED" in text and "runs/s1/video.mp4" in text
    assert (out / FIGURE).stat().st_size > 0


def test_a_report_is_refused_while_cases_have_no_result(*, tmp_path: Path) -> None:
    states_path, out = _setup(tmp_path=tmp_path)
    (out / "results-00.jsonl").write_text("")
    with pytest.raises(ValueError, match="3/3 case"):
        PickReports.build(out=out, states_path=states_path, implementation="current")
