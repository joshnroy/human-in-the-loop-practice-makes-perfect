"""One disconnected Robocode Docker transport for coding and tool-free judgments."""

import importlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import replace  # noqa: TID251 - upstream Robocode objects are dataclasses
from pathlib import Path
from typing import Any

from tenacity import Retrying, retry_if_result, wait_random_exponential

from .broker_auth import BrokerAuth
from .sandbox import DockerContainer, SandboxSettings


class RobocodeDockerTransport:
    """Use the same image, broker, namespace verification, and CLI state mounts."""

    def __init__(
        self,
        *,
        sandbox: SandboxSettings,
        backend: str,
        timeout_seconds: float | None = None,
        subagent_policy: str = "legacy",
    ) -> None:
        self.sandbox = sandbox
        self.backend_name = backend
        self.timeout_seconds = timeout_seconds
        if subagent_policy not in {"legacy", "optional", "disallowed"}:
            raise ValueError("Unknown subagent policy")
        self.subagent_policy = subagent_policy

    def modules(self) -> tuple[Any, Any, Any]:
        """Load only the explicitly configured Robocode source checkout."""
        self.sandbox.transport_source()
        try:
            launcher = self.module(name="robocode.utils.sandbox")
            backends = self.module(name="robocode.utils.backends", package=True)
            self.module(name=f"robocode.utils.backends.{self.backend_name}")
            omega = importlib.import_module("omegaconf")
        except ImportError as exc:
            raise RuntimeError(
                "Install the configured Robocode checkout for Docker CLI calls"
            ) from exc
        return launcher, backends, omega

    def module(self, *, name: str, package: bool = False) -> Any:
        module = importlib.import_module(name)
        relative = Path("src", *name.split("."))
        relative = relative / "__init__.py" if package else relative.with_suffix(".py")
        expected = self.sandbox.robocode_checkout.resolve() / relative
        if module.__file__ is None or Path(module.__file__).resolve() != expected:
            raise RuntimeError("Imported Robocode differs from the configured isolation checkout")
        return module

    def run_with_recovery(
        self,
        *,
        config: Any,
        backend: Any,
        launcher: Any,
        read_only_evidence: bool = False,
    ) -> Any:
        """Use upstream recovery and budget accounting around the shared transport."""
        recovery = self.module(name="robocode.utils.rate_limit")

        def run_attempt(active: Any, active_backend: Any) -> Any:  # noqa: PLR0917
            started = time.monotonic()
            stream = self.run(
                config=active,
                backend=active_backend,
                launcher=launcher,
                read_only_evidence=read_only_evidence,
            )
            launcher._final_commit(active.sandbox_dir)
            return launcher._stream_result_to_sandbox_result(
                stream,
                active.sandbox_dir,
                active.output_filename,
                wall_time_s=time.monotonic() - started,
            )

        active = config
        spent = 0.0
        aborted_tokens = 0
        retries = 0

        def invoke_native() -> Any:
            nonlocal active, spent, aborted_tokens, retries
            result = recovery.run_with_rate_limit_retry(
                docker_config=None,
                local_config=active,
                backend=backend,
                run_sandbox=run_attempt,
            )
            metrics = result.generation_metrics
            previous_cost = float(getattr(metrics, "aborted_cost_usd", 0.0) or 0.0)
            attempt_cost = result.total_cost_usd
            total = previous_cost + float(attempt_cost or 0.0)
            retryable = self.generic_rate_limit(
                result=result, path=config.sandbox_dir.parent / "stream.jsonl"
            )
            # Never replenish a coding invocation's allowance on a retry. Missing
            # spend accounting cannot safely receive another finite allowance.
            budget = config.max_budget_usd
            retryable = retryable and not (
                budget > 0 and (attempt_cost is None or spent + total >= budget)
            )
            if retryable:
                spent += total
                aborted_tokens += int(getattr(metrics, "total_tokens", 0) or 0)
                retries += 1
                active = replace(
                    active,
                    resume_previous_session=True,
                    max_budget_usd=budget - spent if budget else 0.0,
                )
                (config.sandbox_dir.parent / "rate-limit-retry.json").write_text(
                    json.dumps({
                        "retries": retries,
                        "aborted_cost_usd": spent,
                        "remaining_budget_usd": active.max_budget_usd,
                        "last_failure_unix": time.time(),
                    })
                )
                return None
            if retries:
                result = replace(
                    result,
                    generation_metrics=replace(
                        metrics,
                        aborted_cost_usd=previous_cost + spent,
                        aborted_tokens=int(getattr(metrics, "aborted_tokens", 0) or 0)
                        + aborted_tokens,
                        rate_limit_retries=int(getattr(metrics, "rate_limit_retries", 0) or 0)
                        + retries,
                    ),
                )
            return result

        # Confirmed generic 429 only. Other recovery remains upstream. No new
        # timeout/retry cap; exponential jitter avoids hammering a limited account.
        def record_wait(retry_state: Any) -> None:  # noqa: PLR0917 - tenacity callback
            path = config.sandbox_dir.parent / "rate-limit-retry.json"
            status = json.loads(path.read_text())
            status.update(
                status="waiting", retry_at_unix=time.time() + retry_state.next_action.sleep
            )
            path.write_text(json.dumps(status))

        result = Retrying(
            retry=retry_if_result(lambda result: result is None),
            wait=wait_random_exponential(multiplier=30, min=30, max=300),
            before_sleep=record_wait,
        )(invoke_native)
        retry_path = config.sandbox_dir.parent / "rate-limit-retry.json"
        if retry_path.exists():
            retry_status = json.loads(retry_path.read_text())
            retry_status.update(status="finished", finished_unix=time.time())
            retry_path.write_text(json.dumps(retry_status))
        (config.sandbox_dir.parent / "recovery-result.json").write_text(
            json.dumps(
                vars(result),
                default=lambda value: vars(value) if hasattr(value, "__dict__") else str(value),
                indent=2,
            )
            + "\n"
        )
        return result

    def run(
        self,
        *,
        config: Any,
        backend: Any,
        launcher: Any,
        tools_disabled: bool = False,
        read_only_evidence: bool = False,
    ) -> Any:
        """Reuse Robocode's broker, CLI and accounting inside a disconnected Docker image."""
        broker = self.module(name="robocode.utils.model_broker")
        provider = BrokerAuth.load_upstream(
            backend=self.backend_name, broker=broker, timeout_seconds=self.timeout_seconds
        )
        container = DockerContainer(settings=self.sandbox)
        # Initial inputs include editable controllers; never restore them over
        # work produced by the conversation we are resuming.
        if not config.resume_previous_session:
            launcher._setup_sandbox_dir(config)
        command = backend.build_cli_cmd(config)
        if config.resume_previous_session and self.backend_name == "claude":
            session = self.last_session_id(path=config.sandbox_dir.parent / "stream.jsonl")
            marker = config.sandbox_dir.parent / "resume-session.json"
            if session is None and marker.exists():
                session = json.loads(marker.read_text())["session_id"]
            if session:
                command = [arg for arg in command if arg != "--continue"]
                command.extend(["--resume", session])
        if tools_disabled:
            command = self.disable_tools(command=command, config=config)
        if not tools_disabled and not read_only_evidence:
            command = self.configure_planning_tools(command=command, config=config)
        if read_only_evidence:
            command = self.configure_evidence_tools(command=command, config=config)
        command, environment = self.model_client(backend=self.backend_name, command=command)
        if self.backend_name == "claude":
            environment.update({
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(getattr(config, "max_output_tokens", 32768)),
                "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(getattr(config, "autocompact_pct", 80)),
            })
        if not tools_disabled and not read_only_evidence:
            # Coding instructions and file-edit hooks are irrelevant to a judgment.
            backend.setup_sandbox_files(config, docker_python=self.sandbox.container_python)
        if (
            not tools_disabled
            and not read_only_evidence
            and self.backend_name == "claude"
            and self.subagent_policy != "legacy"
        ):
            self.configure_delegation(sandbox_dir=config.sandbox_dir)
            if self.subagent_policy == "disallowed":
                command.extend(["--disallowedTools", "Agent", "Task"])
        launcher._initial_commit(config.sandbox_dir)
        name = f"hitl-coder-{uuid.uuid4().hex}"
        with tempfile.TemporaryDirectory(prefix="hitl-model-broker-") as tmp:
            transport = Path(tmp)
            model_mounts: tuple[tuple[Path, str, bool], ...] = ()
            if self.backend_name == "claude":
                model = command[command.index("--model") + 1]
                policy: dict[str, Any] = {
                    "model": model,
                    "availableModels": [model],
                    "enforceAvailableModels": True,
                    "env": {"CLAUDE_CODE_SUBAGENT_MODEL": model},
                }
                # --model alone does not constrain Agent(model="sonnet"). Managed
                # settings cover child models too and cannot be edited by the
                # disconnected non-root coding process or project settings.
                if read_only_evidence or (
                    not tools_disabled and self.subagent_policy == "disallowed"
                ):
                    policy["permissions"] = {"deny": ["Agent", "Task"]}
                if read_only_evidence:
                    policy["permissions"]["deny"] += ["Bash", "Write", "Edit", "NotebookEdit"]
                policy_path = transport / "managed-settings.json"
                policy_path.write_text(json.dumps(policy, indent=2) + "\n")
                (config.sandbox_dir.parent / "model-policy.json").write_text(
                    json.dumps(policy, indent=2) + "\n"
                )
                model_mounts = ((policy_path, "/etc/claude-code/managed-settings.json", True),)
            shutil.copyfile(self.sandbox.transport_source(), transport / "transport.py")
            (transport / "transport.json").write_text(
                json.dumps({
                    "strict_blackbox": True,
                    "listeners": [
                        {"port": broker.MODEL_PORT, "socket": "/run/hitl-broker/model.sock"}
                    ],
                })
            )
            docker_command = container.command(
                name=name,
                work_dir=config.sandbox_dir,
                writable=True,
                mounts=(
                    (transport, "/run/hitl-broker", True),
                    *model_mounts,
                    *(
                        (((config.sandbox_dir / "evidence"), "/sandbox/evidence", True),)
                        if read_only_evidence and (config.sandbox_dir / "evidence").is_dir()
                        else ()
                    ),
                    *self.cli_state_mounts(
                        sandbox_dir=config.sandbox_dir, backend=self.backend_name
                    ),
                ),
                environment=environment,
                argv=[
                    "/run/hitl-broker/transport.py",
                    "/run/hitl-broker/transport.json",
                    *command,
                ],
            )
            # Native Robocode does not impose these per-container resource caps.
            for flag in (
                ()
                if self.sandbox.resource_pool_dir is not None
                else ("--memory", "--memory-swap", "--cpus", "--pids-limit")
            ):
                if flag in docker_command:
                    index = docker_command.index(flag)
                    del docker_command[index : index + 2]
            docker_command = [
                item.replace(",size=256m", "").replace(",size=128m", "") for item in docker_command
            ]
            (config.sandbox_dir.parent / "container_command.json").write_text(
                json.dumps(docker_command, indent=2)
            )
            with (
                broker.model_broker(
                    transport, provider, config.sandbox_dir.parent / "broker.jsonl"
                ),
                tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr,
                launcher.agent_stdin(backend, config) as stdin,
            ):
                timed_out = threading.Event()
                timer: threading.Timer | None = None
                try:
                    with subprocess.Popen(
                        docker_command,
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                        text=True,
                        start_new_session=True,
                        env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                    ) as process:

                        def expire() -> None:
                            timed_out.set()
                            # End the pipe reader immediately, then remove the actual
                            # container in finally; killing docker alone is insufficient.
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)

                        if self.timeout_seconds is not None:
                            timer = threading.Timer(self.timeout_seconds, expire)
                            timer.start()
                        try:
                            stream = backend.parse_stream(
                                process,
                                stream_log_path=config.sandbox_dir.parent / "stream.jsonl",
                                stderr_file=stderr,
                            )
                        except BaseException:
                            # Popen.__exit__ waits; unblock it before leaving on a
                            # parser error or operator interrupt, then remove Docker.
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                            raise
                    if timed_out.is_set():
                        raise TimeoutError("Disconnected Docker CLI exceeded its time budget")
                finally:
                    if timer is not None:
                        timer.cancel()
                    container.remove(name=name)
                    stderr.seek(0)
                    (config.sandbox_dir.parent / "container.stderr").write_text(stderr.read())
        if self.backend_name == "claude":
            self.validate_claude_models(
                path=config.sandbox_dir.parent / "stream.jsonl",
                expected=command[command.index("--model") + 1],
            )
        return stream

    def configure_delegation(self, *, sandbox_dir: Path) -> None:
        """Replace upstream mandatory delegation with the experiment condition."""
        path = sandbox_dir / "CLAUDE.md"
        original = path.read_text()
        start = original.find("CONTEXT MANAGEMENT:")
        if start < 0:
            # Compatibility with workspaces prepared before markers were retained.
            start = original.find("Work directly on this bounded skill task.")
        end = original.find("\n\n", start)
        if start < 0 or end < 0:
            raise RuntimeError("Expected upstream context-management instructions missing")
        common = (
            "Work directly on this bounded skill task. Keep file reads and outputs focused. "
            "Avoid duplicate reviews of the same code or trajectories. "
        )
        delegation = (
            "Delegation is optional, not required. Use a subagent only for a concrete, "
            "independent question that materially helps the task; do not create nested subagents."
            if self.subagent_policy == "optional"
            else "Subagents are disallowed. Read, reason, edit, and check the work yourself. "
            "Do not invoke Agent/Task or spawn another model CLI or agent process."
        )
        path.write_text(
            original[:start] + "CONTEXT MANAGEMENT: " + common + delegation + original[end:]
        )

    @staticmethod
    def last_session_id(*, path: Path) -> str | None:
        """Read only the top-level invocation ID, never a delegated child ID."""
        session = None
        if path.exists():
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if not event.get("parent_tool_use_id") and event.get("type") in {
                    "system",
                    "result",
                }:
                    candidate = event.get("session_id")
                    if isinstance(candidate, str) and candidate:
                        session = candidate
        return session

    @staticmethod
    def generic_rate_limit(*, result: Any, path: Path) -> bool:
        """Require terminal CLI rate-limit evidence, even if partial output exists."""
        messages = [] if result.success else [str(result.error or "").lower()]
        if path.exists():
            # Only inspect events since the most recent CLI initialization;
            # old failures in an appended recovery transcript are not new errors.
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event.get("type") == "system" and event.get("subtype") == "init":
                    messages = [] if result.success else [str(result.error or "").lower()]
                if event.get("type") == "result" and not event.get("is_error", False):
                    messages = []
                    continue
                if event.get("is_api_error_message") is True or event.get("type") == "result":
                    messages.append(json.dumps(event).lower())
        return any(
            "rate_limit_error" in message
            or "exceed your account's rate limit" in message
            or "429" in message
            and ("rate" in message or "too many requests" in message)
            for message in messages
        )

    @staticmethod
    def validate_claude_models(*, path: Path, expected: str) -> None:
        """Reject responses from another model before accepting generated code."""
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                models: set[str] = set()
                if event.get("type") == "assistant":
                    actual = event.get("message", {}).get("model")
                    synthetic_api_error = (
                        actual == "<synthetic>"
                        and event.get("is_api_error_message") is True
                        and bool(event.get("error"))
                    )
                    if isinstance(actual, str) and not synthetic_api_error:
                        models.add(actual)
                elif event.get("type") == "result":
                    models.update(event.get("modelUsage", {}))
                if models - {expected}:
                    raise RuntimeError(
                        f"Claude model mismatch: expected {expected}, observed {sorted(models)}"
                    )

    @staticmethod
    def configure_planning_tools(*, command: list[str], config: Any) -> list[str]:
        """Expose local PyBullet previews only when a planning specification exists."""
        specification = config.sandbox_dir / "robot_spec.json"
        if not specification.is_file() or "planning_scene" not in json.loads(
            specification.read_text()
        ):
            return command
        if config.mcp_tools or "--mcp-config" in command:
            raise ValueError("Do not mix native environment MCP with PyBullet-only tools")
        command = list(command)
        command[command.index("--tools") + 1] += (
            ",mcp__robocode-tools__render_state,mcp__robocode-tools__render_policy"
        )
        mcp = config.sandbox_dir / "planning-mcp.json"
        mcp.write_text(
            json.dumps({
                "mcpServers": {
                    "robocode-tools": {
                        "command": "/opt/robocode-strict/bin/python",
                        "args": ["/opt/hitl-planning/planning_mcp.py"],
                    }
                }
            })
        )
        return [*command, "--strict-mcp-config", "--mcp-config", "/sandbox/planning-mcp.json"]

    @staticmethod
    def configure_evidence_tools(*, command: list[str], config: Any) -> list[str]:
        """Permit observation inspection without execution, edits or delegation."""
        command = list(command)
        if config.mcp_tools:
            raise ValueError("State evaluation does not expose simulator MCP tools")
        command[command.index("--tools") + 1] = "Read,Glob,Grep"
        if "--setting-sources" in command:
            command[command.index("--setting-sources") + 1] = ""
        mcp = config.sandbox_dir / "empty-mcp.json"
        mcp.write_text('{"mcpServers": {}}')
        return [
            *command,
            "--disallowedTools",
            "Bash",
            "Write",
            "Edit",
            "NotebookEdit",
            "Agent",
            "Task",
            "--strict-mcp-config",
            "--mcp-config",
            "/sandbox/empty-mcp.json",
        ]

    @staticmethod
    def disable_tools(*, command: list[str], config: Any) -> list[str]:
        """Disable built-in and MCP tools without changing the upstream CLI launcher."""
        if config.mcp_tools or config.resume_previous_session:
            raise ValueError("Judgments require fresh sessions with no MCP tools")
        command = list(command)
        if "--tools" not in command or any(
            argument in command for argument in ("--continue", "--resume", "--mcp-config")
        ):
            raise ValueError("Unexpected Claude CLI tool/session configuration")
        command[command.index("--tools") + 1] = ""
        if "--setting-sources" in command:
            command[command.index("--setting-sources") + 1] = ""
        # No inherited project, account, plugin, or managed MCP configuration.
        mcp = config.sandbox_dir / "empty-mcp.json"
        mcp.write_text('{"mcpServers": {}}')
        return [
            *command,
            "--disallowedTools",
            "*",
            "--strict-mcp-config",
            "--mcp-config",
            "/sandbox/empty-mcp.json",
            "--max-thinking-tokens",
            "0",
        ]

    @staticmethod
    def cli_state_mounts(*, sandbox_dir: Path, backend: str) -> tuple[tuple[Path, str, bool], ...]:
        """Give the CLI writable state without mounting operator configuration.

        A nested sessions bind alone makes Docker create a root-owned CLI home
        on the otherwise user-owned home tmpfs. Mount its parent explicitly so
        startup can write SQLite state and PATH aliases as the unprivileged user.
        Keep session logs at Robocode's existing path for budget accounting.
        """
        if backend not in {"codex", "claude"}:
            raise ValueError("The disconnected Docker broker supports claude or codex")
        state = sandbox_dir.parent / "cli-state" / backend
        sessions = sandbox_dir / ".agent_sessions" / backend
        state.mkdir(parents=True, exist_ok=True)
        sessions.mkdir(parents=True, exist_ok=True)
        root = f"/home/node/.{backend}"
        target = f"{root}/sessions" if backend == "codex" else f"{root}/projects"
        return ((state, root, False), (sessions, target, False))

    @staticmethod
    def model_client(*, backend: str, command: list[str]) -> tuple[list[str], dict[str, str]]:
        """The CLI sees an inert token and only the fixed host inference broker."""
        command = list(command)
        token = "local-broker-no-provider-secret"
        environment = {"ROBOCODE_MODEL_TOKEN": token, "UV_OFFLINE": "1", "PIP_NO_INDEX": "1"}
        if backend == "claude":
            environment.update({
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:18080",
                "ANTHROPIC_AUTH_TOKEN": token,
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            })
        elif backend == "codex":
            for key, value in {
                "model_provider": "robocode",
                "model_providers.robocode.name": "Robocode isolated broker",
                "model_providers.robocode.base_url": "http://127.0.0.1:18080/v1",
                "model_providers.robocode.wire_api": "responses",
                "model_providers.robocode.env_key": "ROBOCODE_MODEL_TOKEN",
                "model_providers.robocode.supports_websockets": False,
                "features.responses_websockets": False,
                "features.responses_websockets_v2": False,
            }.items():
                command[-1:-1] = ["--config", f"{key}={json.dumps(value)}"]
        else:
            raise ValueError("The disconnected Docker broker supports claude or codex")
        return command, environment
