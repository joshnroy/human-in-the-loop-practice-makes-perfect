"""A coder killed by its memory cap says so, and the caps are recorded."""

import json
from pathlib import Path

import pytest

from hitl_pmp.agentic_runtime.sandbox import SandboxSettings
from hitl_pmp.full_agentic.runner import FullAgenticRunner


def endpoint(**overrides: bool) -> str:
    flags = dict(
        early_stop=False,
        practice_exhausted=False,
        finished=False,
        coder_oom_killed=False,
        uncertain=False,
        stopped=False,
        timed_out=False,
        coder_error=False,
    )
    flags.update(overrides)
    return FullAgenticRunner.adaptation_endpoint(**flags)


def test_oom_kill_is_the_endpoint_even_with_an_unresolved_step() -> None:
    # The 2026-10-13 kill left a step in flight, so it was reported only as
    # unresolved_execution; the cap is the cause and must be named first.
    assert endpoint(coder_oom_killed=True, uncertain=True) == "coder_oom_killed"
    assert endpoint(uncertain=True) == "unresolved_execution"
    assert endpoint(coder_oom_killed=True, practice_exhausted=True) == "practice_step_budget"
    assert endpoint(coder_oom_killed=True, finished=True) == "agent_submission"
    assert endpoint() == "model_budget_or_cli_end"


def test_three_perfect_evaluations_is_the_endpoint_when_it_stopped_the_run() -> None:
    # The early stop ends the loop before the budget or any later cause is reached.
    assert endpoint(early_stop=True) == "three_perfect_evaluations"
    assert endpoint(early_stop=True, practice_exhausted=True) == "three_perfect_evaluations"


@pytest.mark.parametrize("oom", [True, False, None])
def test_coder_exit_summary_reads_the_kept_record(*, tmp_path: Path, oom: bool | None) -> None:
    (tmp_path / "coding").mkdir()
    (tmp_path / "coding" / "container_exit.json").write_text(
        json.dumps({
            "reason": {True: "oom_killed", False: "exited", None: "not_inspectable"}[oom],
            "oom_killed": oom,
            "exit_code": 137 if oom else None,
            "configured_limits": {"memory_mb": 16384},
        })
    )
    summary = FullAgenticRunner.coder_exit(output=tmp_path)
    assert summary is not None
    assert summary["oom_killed"] is oom
    assert summary["reason"] == {True: "oom_killed", False: "exited", None: "not_inspectable"}[oom]


def test_no_record_means_no_summary(*, tmp_path: Path) -> None:
    assert FullAgenticRunner.coder_exit(output=tmp_path) is None


def test_cli_overrides_and_protocol_record_both_caps(*, tmp_path: Path) -> None:
    settings = SandboxSettings(image="t", robocode_checkout=tmp_path, memory_mb=4096)
    unchanged = FullAgenticRunner.with_memory_caps(
        settings=settings, coder_memory_mb=None, sandbox_memory_mb=None
    )
    assert (unchanged.coder_memory_mb, unchanged.memory_mb) == (16384, 4096)
    resolved = FullAgenticRunner.with_memory_caps(
        settings=settings, coder_memory_mb=20000, sandbox_memory_mb=9000
    )
    assert (resolved.coder_memory_mb, resolved.memory_mb) == (20000, 9000)
    assert FullAgenticRunner.container_limits(settings=resolved) == {
        "coder": {"memory_mb": 20000, "memory_swap_mb": 20000, "applied": False},
        "execution_sandbox": {"memory_mb": 9000, "memory_swap_mb": 9000, "applied": True},
    }
    pooled = resolved.model_copy(update={"resource_pool_dir": tmp_path})
    assert FullAgenticRunner.container_limits(settings=pooled)["coder"]["applied"] is True
