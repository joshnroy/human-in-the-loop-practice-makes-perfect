"""Every container's memory cap is explicit and its exit record outlives it."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hitl_pmp.agentic_runtime.sandbox import (
    DockerContainer,
    DockerPolicyExecutor,
    SandboxSettings,
)

OOM_INSPECT = {
    "Name": "/hitl-coder-x",
    "State": {
        "Status": "exited",
        "Running": False,
        "OOMKilled": True,
        "ExitCode": 137,
        "Error": "",
        "StartedAt": "2026-10-13T10:00:00Z",
        "FinishedAt": "2026-10-13T10:33:44Z",
        "Pid": 0,
    },
    "HostConfig": {
        "Memory": 16384 * 2**20,
        "MemorySwap": 16384 * 2**20,
        "NanoCpus": 3_000_000_000,
        "PidsLimit": 256,
        "Binds": ["ignored"],
    },
}


@pytest.fixture
def docker(*, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []
    monkeypatch.setattr(DockerContainer, "client", staticmethod(lambda: ["docker"]))
    monkeypatch.setattr(DockerContainer, "image_id", lambda _: "sha256:" + "a" * 64)
    return calls


def fake_run(*, calls: list[list[str]], inspect: Any) -> Any:
    def run(command: list[str], **kwargs: Any) -> Any:  # noqa: PLR0917 - mimics subprocess.run
        calls.append(command)
        if command[1] == "inspect" and "--format" not in command:
            if inspect is None:
                return SimpleNamespace(
                    returncode=1, stdout="[]", stderr="Error: No such object: hitl-coder-x"
                )
            return SimpleNamespace(returncode=0, stdout=json.dumps([inspect]), stderr="")
        return SimpleNamespace(returncode=0, stdout="false", stderr="")

    return run


def test_defaults_are_sixteen_gb_coder_and_eight_gb_sandbox(*, tmp_path: Path) -> None:
    settings = SandboxSettings(image="test", robocode_checkout=tmp_path)
    assert settings.coder_memory_mb == 16384
    assert settings.memory_mb == 8192


@pytest.mark.parametrize("pool", [False, True])
def test_configured_cap_is_used_with_swap_equal_to_memory(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch, pool: bool
) -> None:
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run", fake_run(calls=docker, inspect=None)
    )
    settings = SandboxSettings(
        image="test",
        robocode_checkout=tmp_path,
        memory_mb=5000,
        coder_memory_mb=12000,
        resource_pool_dir=tmp_path / "pool" if pool else None,
    )
    sandbox = DockerContainer(settings=settings)
    coder = DockerContainer(settings=settings, memory_mb=settings.coder_memory_mb)
    for container, expected in ((sandbox, "5000m"), (coder, "12000m")):
        command = container.command(name="n", work_dir=tmp_path, argv=["true"])
        assert command[command.index("--memory") + 1] == expected
        assert command[command.index("--memory-swap") + 1] == expected
        container.remove(name="n")


def test_containers_are_not_auto_removed(*, tmp_path: Path, docker: list[list[str]]) -> None:
    settings = SandboxSettings(image="test", robocode_checkout=tmp_path)
    command = DockerContainer(settings=settings).command(name="n", work_dir=tmp_path, argv=["x"])
    assert "--rm" not in command


def test_exit_record_written_before_removal(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run",
        fake_run(calls=docker, inspect=OOM_INSPECT),
    )
    record = tmp_path / "coding" / "container_exit.json"
    settings = SandboxSettings(image="test", robocode_checkout=tmp_path)
    DockerContainer(settings=settings, memory_mb=16384).remove(name="hitl-coder-x", record=record)
    inspect_index = next(i for i, c in enumerate(docker) if c[1] == "inspect")
    remove_index = next(i for i, c in enumerate(docker) if c[1:3] == ["rm", "--force"])
    assert inspect_index < remove_index
    written = json.loads(record.read_text())
    assert written["name"] == "hitl-coder-x"
    assert written["state"] == {
        "Status": "exited",
        "Running": False,
        "OOMKilled": True,
        "ExitCode": 137,
        "Error": "",
        "StartedAt": "2026-10-13T10:00:00Z",
        "FinishedAt": "2026-10-13T10:33:44Z",
    }
    assert written["oom_killed"] is True
    assert written["reason"] == "oom_killed"
    assert written["configured_limits"] == {
        "memory_mb": 16384,
        "memory_swap_mb": 16384,
        "cpus": 2.0,
        "pids_limit": 256,
    }
    assert written["applied_limits"] == {
        "Memory": 16384 * 2**20,
        "MemorySwap": 16384 * 2**20,
        "NanoCpus": 3_000_000_000,
        "PidsLimit": 256,
    }


def test_missing_container_still_writes_record_and_removes(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run", fake_run(calls=docker, inspect=None)
    )
    record = tmp_path / "container_exit.json"
    DockerContainer(settings=SandboxSettings(image="t", robocode_checkout=tmp_path)).remove(
        name="hitl-coder-x", record=record
    )
    written = json.loads(record.read_text())
    assert written["state"] is None
    assert written["oom_killed"] is None
    assert written["reason"] == "not_inspectable"
    assert "No such object" in written["inspect_error"]
    assert any(c[1:3] == ["rm", "--force"] for c in docker)


def test_inspect_timeout_still_removes(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(command: list[str], **kwargs: Any) -> Any:  # noqa: PLR0917 - mimics subprocess.run
        docker.append(command)
        if command[1] == "inspect":
            raise subprocess.TimeoutExpired(cmd=command, timeout=30)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("hitl_pmp.agentic_runtime.sandbox.subprocess.run", run)
    record = tmp_path / "container_exit.json"
    DockerContainer(settings=SandboxSettings(image="t", robocode_checkout=tmp_path)).remove(
        name="n", record=record
    )
    assert json.loads(record.read_text())["reason"] == "not_inspectable"
    assert any(c[1:3] == ["rm", "--force"] for c in docker)


def test_running_container_is_recorded_as_force_removed(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    running = json.loads(json.dumps(OOM_INSPECT))
    running["State"].update(Running=True, OOMKilled=False, ExitCode=0, Status="running")
    monkeypatch.setattr(
        "hitl_pmp.agentic_runtime.sandbox.subprocess.run",
        fake_run(calls=docker, inspect=running),
    )
    record = tmp_path / "container_exit.json"
    DockerContainer(settings=SandboxSettings(image="t", robocode_checkout=tmp_path)).remove(
        name="n", record=record
    )
    assert json.loads(record.read_text())["reason"] == "force_removed_while_running"


def test_policy_executor_records_every_container(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = SandboxSettings(image="t", robocode_checkout=tmp_path, artifact_dir=tmp_path / "e")
    executor = DockerPolicyExecutor(settings=settings)
    recorded: list[dict[str, Any]] = []
    monkeypatch.setattr(executor, "command", lambda **kwargs: ["true"])
    monkeypatch.setattr(executor.container, "remove", lambda **kwargs: recorded.append(kwargs))
    relay = SimpleNamespace(started=True, handle_request=lambda: None)
    assert executor._run(work=tmp_path, relay=relay) is None
    assert len(recorded) == 1
    name = recorded[0]["name"]
    assert recorded[0]["record"] == tmp_path / "e" / "containers" / f"{name}.json"


def test_later_removal_does_not_erase_an_informative_record(
    *, tmp_path: Path, docker: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The runner's budget stop removes the coder first; the transport's own
    # finally then finds nothing and must not overwrite what was recorded.
    record = tmp_path / "container_exit.json"
    settings = SandboxSettings(image="t", robocode_checkout=tmp_path)
    for inspect in (OOM_INSPECT, None):
        monkeypatch.setattr(
            "hitl_pmp.agentic_runtime.sandbox.subprocess.run",
            fake_run(calls=docker, inspect=inspect),
        )
        DockerContainer(settings=settings).remove(name="hitl-coder-x", record=record)
    assert json.loads(record.read_text())["reason"] == "oom_killed"


def test_uncapped_coder_does_not_claim_a_cap(*, tmp_path: Path) -> None:
    settings = SandboxSettings(image="t", robocode_checkout=tmp_path)
    limits = DockerContainer(settings=settings, memory_mb=16384, capped=False).configured_limits()
    assert limits == {"memory_mb": None, "memory_swap_mb": None, "cpus": None, "pids_limit": None}
