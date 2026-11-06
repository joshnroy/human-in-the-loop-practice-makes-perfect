"""The benchmark's report.json and page, built from a synthetic output directory."""

import json
from pathlib import Path

import pytest

from analysis.sweepoffshelf3d_skill_benchmark import FIGURE, PAGE, SWEEP_FIGURE, SkillBenchmarkPage
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.report import BenchmarkReports
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.states import StateSet
from hitl_pmp.environments.sweepoffshelf3d.skill_benchmark.types import (
    BenchmarkCase,
    BenchmarkState,
    CaseResult,
)


def _setup(*, tmp_path: Path, labels: list[str]) -> tuple[Path, Path]:
    states_path = tmp_path / "states.jsonl"
    cases = [
        BenchmarkCase(skill="PickWiper"),
        BenchmarkCase(skill="SweepShelf", cube="cube_0", params=[0.025, 15.0]),
    ]
    states = [
        BenchmarkState(
            state_id=f"s{i}",
            family="start",
            seed=i,
            scene_fingerprint="scene",
            qpos=[0.0],
            qvel=[0.0],
            ctrl=[0.0],
            grip=0.0,
            atoms=[],
            cases=cases,
        )
        for i in range(len(labels))
    ]
    StateSet.append(path=states_path, states=states)
    out = tmp_path / "out"
    out.mkdir()
    rows = []
    for state, label in zip(states, labels, strict=True):
        for case in state.cases:
            rows.append(
                CaseResult(
                    case_id=case.case_id(state_id=state.state_id),
                    state_id=state.state_id,
                    family=state.family,
                    skill=case.skill,
                    params=list(case.params),
                    label=label,
                    pybullet="planned" if case.skill == "PickWiper" else "n/a",
                    video=None if label == "success" else f"runs/{state.state_id}/video.mp4",
                ).model_dump_json()
            )
    (out / "results-00.jsonl").write_text("\n".join(rows) + "\n")
    return states_path, out


def test_the_report_names_the_commit_and_fails_while_any_skill_misses_its_target(
    *, tmp_path: Path
) -> None:
    states_path, out = _setup(tmp_path=tmp_path, labels=["success", "plan_blocked"])
    report = BenchmarkReports.build(out=out, states_path=states_path)
    assert not report.passed
    assert len(report.git_commit) == 40
    verdicts = {v.skill: v for v in report.skills}
    assert (verdicts["PickWiper"].successes, verdicts["PickWiper"].total) == (1, 2)
    assert verdicts["PickWiper"].pybullet_planned == 2
    assert verdicts["SweepShelf"].per_state == {"s0": [1, 1], "s1": [0, 1]}
    saved = json.loads((out / "report.json").read_text())
    assert saved["state_set_sha256"] == StateSet.sha256(path=states_path)

    page = SkillBenchmarkPage.render(out=out)
    assert page.name == PAGE
    text = page.read_text()
    assert "FAILED" in text and "1/2" in text and "plan_blocked" in text
    assert "runs/s1/video.mp4" in text
    assert (out / FIGURE).stat().st_size > 0 and (out / SWEEP_FIGURE).stat().st_size > 0


def test_a_report_is_refused_while_cases_have_no_result(*, tmp_path: Path) -> None:
    states_path, out = _setup(tmp_path=tmp_path, labels=["success"])
    lines = (out / "results-00.jsonl").read_text().splitlines()
    (out / "results-00.jsonl").write_text(lines[0] + "\n")
    with pytest.raises(ValueError, match="1/2 case"):
        BenchmarkReports.build(out=out, states_path=states_path)
