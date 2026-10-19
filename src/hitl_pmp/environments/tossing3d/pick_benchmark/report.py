"""The pick benchmark's `report.json`: per-skill verdicts, stamped with the code and the
pick implementation it measured."""

import argparse
from pathlib import Path

from hitl_pmp.config_snapshot import ConfigSnapshot

from .runner import PickShardWorker
from .states import PickStateSet
from .types import PickReport
from .verdicts import PickVerdicts

REPORT_NAME = "report.json"


class PickReports:
    """A static-method container, never instantiated."""

    @staticmethod
    def build(
        *, out: Path, states_path: Path, implementation: str, elapsed_seconds: float = 0.0
    ) -> PickReport:
        states = PickStateSet.load(path=states_path)
        expected = {f"{s.state_id}-{s.skill}" for s in states}
        results = [r for r in PickShardWorker.results(out=out) if r.case_id in expected]
        missing = sorted(expected - {r.case_id for r in results})
        if missing:
            raise ValueError(
                f"{len(missing)}/{len(expected)} case(s) have no result (e.g. {missing[:3]}); "
                "re-run to resume before reporting"
            )
        verdicts = PickVerdicts.judge(results=results)
        snapshot = ConfigSnapshot.collect(args=argparse.Namespace(), fd_exec_path=None)
        report = PickReport(
            pick_implementation=implementation,
            git_commit=snapshot.git_commit,
            git_dirty=snapshot.git_dirty,
            kindergarden_commit=snapshot.kindergarden_commit,
            kinder_models_commit=snapshot.kinder_models_commit,
            state_set=str(states_path),
            state_set_sha256=PickStateSet.sha256(path=states_path),
            passed=PickVerdicts.passed(verdicts=verdicts),
            skills=verdicts,
            num_cases=len(results),
            elapsed_seconds=elapsed_seconds,
        )
        (out / REPORT_NAME).write_text(report.model_dump_json(indent=2) + "\n")
        return report
