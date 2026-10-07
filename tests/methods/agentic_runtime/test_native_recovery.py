"""The shared adapter executes upstream retry/resume and carries finite budgets."""

from types import SimpleNamespace

import pytest

from hitl_pmp.agentic_runtime.docker_cli import RobocodeDockerTransport
from hitl_pmp.agentic_runtime.sandbox import SandboxSettings


def test_native_recovery_resumes_with_remaining_budget(*, tmp_path, monkeypatch):
    recovery = pytest.importorskip("robocode.utils.rate_limit")
    from robocode.utils.sandbox_types import GenerationMetrics, SandboxConfig, SandboxResult

    config = SandboxConfig(
        sandbox_dir=tmp_path, max_budget_usd=2, max_turns=0, output_filename="revision.json"
    )
    calls = []
    transport = RobocodeDockerTransport(
        sandbox=SandboxSettings(image="test", robocode_checkout=tmp_path), backend="claude"
    )
    monkeypatch.setattr(transport, "module", lambda **kwargs: recovery)
    results = iter([
        SandboxResult(
            False,
            None,
            "overloaded",
            total_cost_usd=0.4,
            api_error_hit=True,
            generation_metrics=GenerationMetrics(),
        ),
        SandboxResult(
            True,
            tmp_path / "revision.json",
            None,
            total_cost_usd=0.6,
            generation_metrics=GenerationMetrics(),
        ),
    ])

    def run(**kwargs):
        calls.append(kwargs["config"])
        return next(results)

    monkeypatch.setattr(transport, "run", run)
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    launcher = SimpleNamespace(
        _final_commit=lambda _: None,
        _stream_result_to_sandbox_result=lambda stream, *args, **kw: stream,
    )
    result = transport.run_with_recovery(
        config=config, backend=SimpleNamespace(name="claude"), launcher=launcher
    )
    assert result.success
    assert result.total_cost_usd == pytest.approx(0.6)
    assert result.generation_metrics.aborted_cost_usd == pytest.approx(0.4)
    assert len(calls) == 2
    assert calls[1].resume_previous_session
    assert calls[1].max_budget_usd == pytest.approx(1.6)
    assert calls[1].max_turns == 0
