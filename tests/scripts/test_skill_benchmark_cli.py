"""`scripts/skill_benchmark.py run` writes, by default, where the launch gate looks."""

from pathlib import Path

import pytest

from hitl_pmp.launch_gate import LaunchGate, LaunchIdentity
from scripts.skill_benchmark import SkillBenchmarkCli


def test_a_run_without_out_lands_where_the_gate_reads_this_commits_report(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = LaunchIdentity(
        git_commit="f" * 40, git_dirty=False, kindergarden_commit="k", kinder_models_commit="m"
    )
    monkeypatch.setattr(LaunchGate, "default_root", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(LaunchIdentity, "current", staticmethod(lambda: identity))
    out = SkillBenchmarkCli.default_out()
    assert out / "report.json" == LaunchGate.report_path(
        root=tmp_path, domain="sweepoffshelf3d", commit="f" * 40
    )
