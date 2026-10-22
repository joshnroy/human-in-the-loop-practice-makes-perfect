import hashlib
import json
from pathlib import Path

import pytest

from hitl_pmp.core.practice_costs import SkillStepCosts
from scripts.skill_step_costs import SkillStepCostTable, main

COMMIT = "1df856b04451c64f6264381351ea21205edd3945"


def _case(*, skill: str, label: str, steps: int) -> dict[str, object]:
    return {"skill": skill, "label": label, "controller_steps": steps}


def _report_dir(*, tmp_path: Path) -> Path:
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.json").write_text(
        json.dumps({"domain": "sweepoffshelf3d", "git_commit": COMMIT, "git_dirty": False})
    )
    (report / "results-00.jsonl").write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                _case(skill="PickWiper", label="success", steps=700),
                _case(skill="SweepShelf", label="success", steps=1000),
                _case(skill="SweepShelf", label="execution_failed", steps=3000),
            ]
        )
        + "\n"
    )
    (report / "results-01.jsonl").write_text(
        "\n".join(
            json.dumps(case)
            for case in [
                _case(skill="SweepShelf", label="effects_missing", steps=2000),
                # Refused at planning: no control step ran, so no execution to average.
                _case(skill="SweepShelf", label="no_plan", steps=0),
                _case(skill="PickWiper", label="plan_blocked", steps=0),
            ]
        )
        + "\n"
    )
    return report


def test_the_price_is_the_mean_over_successful_and_failed_executions(*, tmp_path: Path) -> None:
    table = SkillStepCostTable.measure(report_dir=_report_dir(tmp_path=tmp_path))
    assert table.skill_steps == {"PickWiper": 700.0, "SweepShelf": 2000.0}
    assert table.source.cases == {"PickWiper": 1, "SweepShelf": 3}
    assert table.source.planning_refusals_excluded == {"PickWiper": 1, "SweepShelf": 1}


def test_the_table_records_the_commit_and_the_reports_sha256(*, tmp_path: Path) -> None:
    report = _report_dir(tmp_path=tmp_path)
    table = SkillStepCostTable.measure(report_dir=report)
    assert table.source.benchmark_commit == COMMIT
    assert (
        table.source.report_sha256
        == hashlib.sha256((report / "report.json").read_bytes()).hexdigest()
    )


def test_a_dirty_benchmark_is_refused(*, tmp_path: Path) -> None:
    report = _report_dir(tmp_path=tmp_path)
    (report / "report.json").write_text(json.dumps({"git_commit": COMMIT, "git_dirty": True}))
    with pytest.raises(SystemExit, match="dirty"):
        SkillStepCostTable.measure(report_dir=report)


def test_one_command_writes_a_loadable_table(*, tmp_path: Path) -> None:
    out = tmp_path / "costs.json"
    assert main(argv=[str(_report_dir(tmp_path=tmp_path)), "--out", str(out)]) == 0
    assert SkillStepCosts.load(path=out).skill_steps["SweepShelf"] == 2000.0
