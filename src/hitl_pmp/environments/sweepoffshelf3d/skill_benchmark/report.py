"""The benchmark's `report.json`: per-skill verdicts, stamped with the code it measured.

The launch gate reads this file, so it records exactly what a launch must match: this
checkout's commit and whether its tree was dirty, the two KINDER trees' commits, and
the state set's own hash and scene fingerprint.
"""

import argparse
from pathlib import Path

from hitl_pmp.config_snapshot import ConfigSnapshot

from .runner import ShardWorker
from .states import StateSet
from .types import BenchmarkReport
from .verdicts import Verdicts

DOMAIN = "sweepoffshelf3d"
REPORT_NAME = "report.json"


class BenchmarkReports:
    """A static-method container, never instantiated."""

    @staticmethod
    def build(*, out: Path, states_path: Path, elapsed_seconds: float = 0.0) -> BenchmarkReport:
        states = StateSet.load(path=states_path)
        results = ShardWorker.results(out=out)
        expected = {
            case.case_id(state_id=state.state_id) for state in states for case in state.cases
        }
        ran = {result.case_id for result in results}
        missing = sorted(expected - ran)
        if missing:
            raise ValueError(
                f"{len(missing)}/{len(expected)} case(s) have no result (e.g. {missing[:3]}); "
                "re-run to resume before reporting"
            )
        results = [result for result in results if result.case_id in expected]
        verdicts = Verdicts.judge(results=results)
        snapshot = ConfigSnapshot.collect(args=argparse.Namespace(), fd_exec_path=None)
        fingerprints = sorted({state.scene_fingerprint for state in states})
        report = BenchmarkReport(
            domain=DOMAIN,
            git_commit=snapshot.git_commit,
            git_dirty=snapshot.git_dirty,
            kindergarden_commit=snapshot.kindergarden_commit,
            kinder_models_commit=snapshot.kinder_models_commit,
            state_set=str(states_path),
            state_set_sha256=StateSet.sha256(path=states_path),
            scene_fingerprint=";".join(fingerprints),
            passed=Verdicts.passed(verdicts=verdicts),
            skills=verdicts,
            num_cases=len(results),
            num_invalid=sum(result.label == "invalid" for result in results),
            elapsed_seconds=elapsed_seconds,
        )
        (out / REPORT_NAME).write_text(report.model_dump_json(indent=2) + "\n")
        return report
