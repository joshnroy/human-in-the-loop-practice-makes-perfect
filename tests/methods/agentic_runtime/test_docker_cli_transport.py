"""Shared Docker launch semantics for coding and strictly tool-free judgments."""

import io
import json
import signal
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hitl_pmp.agentic_runtime import docker_cli
from hitl_pmp.agentic_runtime.docker_cli import RobocodeDockerTransport
from hitl_pmp.agentic_runtime.sandbox import DockerContainer, SandboxSettings


@pytest.fixture
def launch(*, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"removed": [], "setup": [], "killed": []}
    source = tmp_path / "src/robocode/utils/isolated_transport.py"
    source.parent.mkdir(parents=True)
    source.write_text("def verify_namespace(): pass\ndef verify_strict_runtime(): pass\n")
    sandbox = tmp_path / "call/sandbox"
    sandbox.mkdir(parents=True)
    config = SimpleNamespace(
        sandbox_dir=sandbox,
        mcp_tools=(),
        resume_previous_session=False,
        prompt="judge",
    )

    @contextmanager
    def broker(*args: Any) -> Any:
        calls["broker"] = args
        transport = json.loads((args[0] / "transport.json").read_text())
        assert transport == {
            "strict_blackbox": True,
            "listeners": [{"port": 18080, "socket": "/run/hitl-broker/model.sock"}],
        }
        calls["managed_model_policy"] = json.loads((args[0] / "managed-settings.json").read_text())
        yield

    @contextmanager
    def stdin(*args: Any) -> Any:
        yield io.StringIO(config.prompt)

    def parse(*args: Any, **kwargs: Any) -> Any:
        calls["parse"] = kwargs
        if "error" in calls:
            raise calls["error"]
        kwargs["stream_log_path"].write_text(
            json.dumps({
                "type": "assistant",
                "message": {"model": calls.get("actual_model", "claude-opus-5-5")},
            })
            + "\n"
        )
        return SimpleNamespace(is_error=False)

    class Process:
        def __init__(self, command: list[str], **kwargs: Any) -> None:  # noqa: PLR0917
            calls["command"] = command
            calls["process_kwargs"] = kwargs
            self.pid = 12345

        def __enter__(self) -> "Process":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    class Timer:
        def __init__(self, seconds: float, callback: Any) -> None:  # noqa: PLR0917
            calls["deadline"] = seconds
            self.callback = callback

        def start(self) -> None:
            if calls.get("timeout"):
                self.callback()

        def cancel(self) -> None:
            calls["timer_cancelled"] = True

    module = SimpleNamespace(
        MODEL_PORT=18080,
        model_broker=broker,
        load_broker_upstream=lambda _: "provider",
    )
    monkeypatch.setattr(RobocodeDockerTransport, "module", lambda self, **_: module)
    monkeypatch.setattr(
        docker_cli.BrokerAuth, "load_upstream", staticmethod(lambda **kwargs: "provider")
    )
    monkeypatch.setattr(DockerContainer, "client", staticmethod(lambda: ["docker"]))
    monkeypatch.setattr(DockerContainer, "image_id", lambda self: "sha256:" + "a" * 64)
    monkeypatch.setattr(
        DockerContainer, "remove", lambda self, **kwargs: calls["removed"].append(kwargs["name"])
    )
    monkeypatch.setattr(docker_cli.subprocess, "Popen", Process)
    monkeypatch.setattr(docker_cli.threading, "Timer", Timer)
    monkeypatch.setattr(docker_cli.os, "killpg", lambda *args: calls["killed"].append(args))
    backend = SimpleNamespace(
        build_cli_cmd=lambda _: [
            "claude",
            "-p",
            "--tools",
            "Bash,Read,Write",
            "--setting-sources",
            "project",
            "--model",
            "claude-opus-5-5",
        ],
        setup_sandbox_files=lambda *args, **kwargs: calls["setup"].append(kwargs),
        parse_stream=parse,
    )
    launcher = SimpleNamespace(
        _setup_sandbox_dir=lambda _: None,
        _initial_commit=lambda _: None,
        agent_stdin=stdin,
    )
    calls["transport"] = RobocodeDockerTransport(
        sandbox=SandboxSettings(image="same-image", robocode_checkout=tmp_path),
        backend="claude",
        timeout_seconds=90,
    )
    calls["arguments"] = {"config": config, "backend": backend, "launcher": launcher}
    return calls


@pytest.mark.parametrize("tools_disabled", [False, True])
def test_both_calls_share_disconnected_image_and_private_state(
    *, launch: dict[str, Any], tools_disabled: bool
) -> None:
    launch["transport"].run(**launch["arguments"], tools_disabled=tools_disabled)
    command = launch["command"]
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--entrypoint") + 1] == "/opt/robocode-strict/bin/python"
    assert "sha256:" + "a" * 64 in command
    assert "ANTHROPIC_AUTH_TOKEN=local-broker-no-provider-secret" in command
    assert "ANTHROPIC_BASE_URL=http://127.0.0.1:18080" in command
    mounts = [command[i + 1] for i, item in enumerate(command) if item == "--mount"]
    assert any("dst=/home/node/.claude" in item for item in mounts)
    assert any("dst=/etc/claude-code/managed-settings.json,readonly" in item for item in mounts)
    assert launch["managed_model_policy"] == {
        "model": "claude-opus-5-5",
        "availableModels": ["claude-opus-5-5"],
        "enforceAvailableModels": True,
        "env": {"CLAUDE_CODE_SUBAGENT_MODEL": "claude-opus-5-5"},
    }
    assert all(".credentials.json" not in item for item in mounts)
    assert command[command.index("--tools") + 1] == ("" if tools_disabled else "Bash,Read,Write")
    if tools_disabled:
        assert command[command.index("--disallowedTools") + 1] == "*"
        assert "--strict-mcp-config" in command
        assert command[command.index("--mcp-config") + 1] == "/sandbox/empty-mcp.json"
        assert command[command.index("--setting-sources") + 1] == ""
        assert launch["setup"] == []
        empty_mcp = launch["arguments"]["config"].sandbox_dir / "empty-mcp.json"
        assert json.loads(empty_mcp.read_text()) == {"mcpServers": {}}
    else:
        assert len(launch["setup"]) == 1
    assert "--continue" not in command and "--resume" not in command
    assert launch["deadline"] == 90
    assert launch["timer_cancelled"] is True
    assert len(launch["removed"]) == 1
    root = launch["arguments"]["config"].sandbox_dir.parent
    assert json.loads((root / "container_command.json").read_text()) == command
    assert json.loads((root / "model-policy.json").read_text()) == launch["managed_model_policy"]
    assert (root / "container.stderr").read_text() == ""


def test_timeout_kills_client_and_removes_container(*, launch: dict[str, Any]) -> None:
    launch["timeout"] = True
    with pytest.raises(TimeoutError, match="time budget"):
        launch["transport"].run(**launch["arguments"], tools_disabled=True)
    assert launch["killed"] == [(12345, signal.SIGKILL)]
    assert len(launch["removed"]) == 1
    assert launch["timer_cancelled"] is True


def test_parse_failure_still_removes_container(*, launch: dict[str, Any]) -> None:
    launch["error"] = RuntimeError("broken CLI stream")
    with pytest.raises(RuntimeError, match="broken CLI stream"):
        launch["transport"].run(**launch["arguments"], tools_disabled=True)
    assert len(launch["removed"]) == 1
    assert launch["timer_cancelled"] is True
    assert launch["killed"] == [(12345, signal.SIGKILL)]


def test_unexpected_model_is_rejected_before_acceptance(*, launch: dict[str, Any]) -> None:
    launch["actual_model"] = "claude-sonnet-5-5"
    with pytest.raises(RuntimeError, match="Claude model mismatch"):
        launch["transport"].run(**launch["arguments"])
    assert len(launch["removed"]) == 1


def test_result_usage_cannot_hide_a_different_child_model(*, tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text(
        json.dumps({
            "type": "result",
            "modelUsage": {"claude-opus-5-5": {}, "claude-sonnet-5-5": {}},
        })
        + "\n"
    )
    with pytest.raises(RuntimeError, match="Claude model mismatch"):
        RobocodeDockerTransport.validate_claude_models(path=path, expected="claude-opus-5-5")


@pytest.mark.parametrize(
    "field,value", [("mcp_tools", ("simulator",)), ("resume_previous_session", True)]
)
def test_judgments_reject_mcp_or_resumption(
    *, launch: dict[str, Any], field: str, value: Any
) -> None:
    setattr(launch["arguments"]["config"], field, value)
    with pytest.raises(ValueError, match="fresh sessions"):
        launch["transport"].run(**launch["arguments"], tools_disabled=True)
    assert "command" not in launch


@pytest.mark.parametrize("policy", ["optional", "disallowed"])
def test_experiment_delegation_replaces_instruction_and_enforces_denial(
    *, launch: dict[str, Any], policy: str
) -> None:
    transport = launch["transport"]
    transport.subagent_policy = policy
    path = launch["arguments"]["config"].sandbox_dir / "CLAUDE.md"
    path.write_text("Paths rule.\n\nCONTEXT MANAGEMENT: Delegate ALL reading.\n\nPython rule.\n")
    transport.run(**launch["arguments"])
    body = path.read_text()
    assert "Delegate ALL" not in body
    assert "Paths rule." in body and "Python rule." in body
    if policy == "disallowed":
        assert "Subagents are disallowed" in body
        assert launch["managed_model_policy"]["permissions"]["deny"] == ["Agent", "Task"]
        command = launch["command"]
        assert command[command.index("--disallowedTools") + 1 :] == ["Agent", "Task"]
    else:
        assert "Delegation is optional" in body
        assert "do not create nested subagents" in body
        assert "permissions" not in launch["managed_model_policy"]


def test_no_deadline_and_native_context_settings(*, launch):
    launch["transport"].timeout_seconds = None
    launch["transport"].run(**launch["arguments"])
    assert "deadline" not in launch
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS=32768" in launch["command"]
    assert "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80" in launch["command"]
    assert not set(("--memory", "--cpus", "--pids-limit")) & set(launch["command"])


def test_evidence_tools_cannot_edit_execute_or_delegate(*, launch):
    evidence = launch["arguments"]["config"].sandbox_dir / "evidence"
    evidence.mkdir()
    launch["transport"].run(**launch["arguments"], read_only_evidence=True)
    command = launch["command"]
    assert command[command.index("--tools") + 1] == "Read,Glob,Grep"
    assert "--max-thinking-tokens" not in command
    assert "--continue" not in command
    assert any("dst=/sandbox/evidence,readonly" in item for item in command)
    assert launch["setup"] == []
    assert set(launch["managed_model_policy"]["permissions"]["deny"]) >= {"Bash", "Edit", "Agent"}


def test_resume_does_not_overwrite_partial_policy_edits(*, launch):
    config = launch["arguments"]["config"]
    config.resume_previous_session = True
    launcher = launch["arguments"]["launcher"]

    def destructive_setup(_):
        pytest.fail("resume restored initial files over revised policies")

    launcher._setup_sandbox_dir = destructive_setup
    launch["transport"].run(**launch["arguments"])


def test_planning_mcp_is_local_pybullet_not_native_environment(*, launch):
    spec = launch["arguments"]["config"].sandbox_dir / "robot_spec.json"
    spec.write_text('{"planning_scene": {}}')
    launch["transport"].run(**launch["arguments"])
    command = launch["command"]
    assert "mcp__robocode-tools__render_state" in command[command.index("--tools") + 1]
    assert command[command.index("--mcp-config") + 1] == "/sandbox/planning-mcp.json"
    configuration = json.loads(spec.with_name("planning-mcp.json").read_text())
    assert configuration["mcpServers"]["robocode-tools"]["args"] == [
        "/opt/hitl-planning/planning_mcp.py"
    ]


def test_evaluator_never_gets_planning_mcp(*, launch):
    spec = launch["arguments"]["config"].sandbox_dir / "robot_spec.json"
    spec.write_text('{"planning_scene": {}}')
    launch["transport"].run(**launch["arguments"], read_only_evidence=True)
    command = launch["command"]
    assert command[command.index("--tools") + 1] == "Read,Glob,Grep"
    assert not spec.with_name("planning-mcp.json").exists()


@pytest.mark.parametrize("api_error", [True, False])
def test_synthetic_api_errors_reach_native_recovery_without_relaxing_model_check(
    *, tmp_path, api_error
):
    path = tmp_path / "stream.jsonl"
    path.write_text(
        json.dumps({
            "type": "assistant",
            "message": {"model": "<synthetic>"},
            "is_api_error_message": api_error,
            "error": "rate_limit",
        })
        + "\n"
    )
    if api_error:
        RobocodeDockerTransport.validate_claude_models(path=path, expected="claude-opus-5-5")
    else:
        with pytest.raises(RuntimeError, match="Claude model mismatch"):
            RobocodeDockerTransport.validate_claude_models(path=path, expected="claude-opus-5-5")
