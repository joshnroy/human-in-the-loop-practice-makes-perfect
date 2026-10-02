"""Optional adapter to Robocode's audited disconnected coding-agent launcher."""

import importlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .sandbox import DockerContainer, SandboxSettings
from .types import GeneratedLibrary, RevisionProposal


class CodingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sandbox: SandboxSettings
    artifact_dir: Path
    backend: str
    model: str
    max_budget_usd: float = Field(gt=0)
    max_turns: int = Field(default=20, gt=0)
    reasoning_effort: str = ""
    system_prompt: str = ""
    timeout_seconds: float = Field(default=900, gt=0)


class RobocodeCodingAgent:
    """Revise policy files offline; no environment reset or simulator access.

    The broker still permits model inference. Robocode owns that fixed-provider
    exception, credentials, namespace verification and source-clean image checks.
    End-of-session evidence is mounted as data; this adapter does not provide an
    environment server during learning, so learner tool calls cannot reset or
    advance the persistent robot behind the practice loop's accounting.
    """

    def __init__(
        self, *, settings: CodingSettings, input_files: dict[str, str] | None = None
    ) -> None:
        self.settings = settings
        self.input_files = dict(input_files or {})
        for name in self.input_files:
            if (
                Path(name).is_absolute()
                or ".." in Path(name).parts
                or Path(name).as_posix() != name
                or name in {"generated_library.json", "revision.json", "."}
            ):
                raise ValueError("Coding context paths must remain inside the sandbox")
        if settings.backend not in {"claude", "codex"}:
            raise ValueError("The disconnected Robocode broker supports claude or codex")

    def generate(self, *, prompt: str) -> GeneratedLibrary:
        result = self._run(
            prompt=prompt, files=dict(self.input_files), output_filename="generated_library.json"
        )
        return GeneratedLibrary.model_validate_json(result)

    def revise(
        self, *, library: GeneratedLibrary, evidence: list[dict[str, Any]], prompt: str
    ) -> RevisionProposal:
        prepared_evidence = json.loads(json.dumps(evidence, allow_nan=False))
        trajectories: dict[str, str] = {}
        for index, item in enumerate(prepared_evidence):
            path = item.get("trajectory_path")
            if path is not None:
                destination = f"trajectories/{index:06d}.jsonl"
                trajectories[destination] = Path(path).read_text()
                item["trajectory_path"] = destination
        files = {
            **library.controllers,
            **trajectories,
            "library.json": library.model_dump_json(indent=2),
            "evidence.json": json.dumps(prepared_evidence, allow_nan=False),
        }
        if self.input_files.keys() & files.keys():
            raise ValueError("Coding context collides with generated policy or evidence files")
        files.update(self.input_files)
        result = self._run(prompt=prompt, files=files, output_filename="revision.json")
        proposal = RevisionProposal.model_validate_json(result)
        if not set(proposal.controllers).issubset(library.controllers):
            raise ValueError("Coding agent attempted to change the frozen option library")
        return proposal

    def _run(self, *, prompt: str, files: dict[str, str], output_filename: str) -> str:
        self.settings.sandbox.transport_source()
        try:
            launcher = importlib.import_module("robocode.utils.sandbox")
            backends = importlib.import_module("robocode.utils.backends")
            omega = importlib.import_module("omegaconf")
        except ImportError as exc:
            raise RuntimeError(
                "Install the configured Robocode checkout to run the coding learner"
            ) from exc
        expected = (
            self.settings.sandbox.robocode_checkout.resolve() / "src/robocode/utils/sandbox.py"
        )
        if launcher.__file__ is None or Path(launcher.__file__).resolve() != expected:
            raise RuntimeError("Imported Robocode differs from the configured isolation checkout")
        run_dir = self.settings.artifact_dir / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        with tempfile.TemporaryDirectory(prefix="hitl-learner-inputs-") as tmp:
            source_dir = Path(tmp)
            initial_files: dict[str, Path] = {}
            for name, source in files.items():
                path = source_dir / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
                initial_files[name] = path
            config = launcher.SandboxConfig(
                sandbox_dir=run_dir / "sandbox",
                init_files=initial_files,
                prompt=prompt,
                system_prompt=self.settings.system_prompt,
                output_filename=output_filename,
                model=self.settings.model,
                effort=self.settings.reasoning_effort,
                max_budget_usd=self.settings.max_budget_usd,
                max_turns=self.settings.max_turns,
                blackbox=True,
                mcp_tools=(),
            )
            backend = backends.create_backend(
                omega.DictConfig({
                    "backend": self.settings.backend,
                    "model": self.settings.model,
                    "reasoning_effort": self.settings.reasoning_effort,
                })
            )
            result = self._run_docker(config=config, backend=backend, launcher=launcher)
        if not result.success or result.output_file is None:
            raise RuntimeError(f"Disconnected coding learner failed: {result.error}")
        output = Path(result.output_file)
        sandbox_root = (run_dir / "sandbox").resolve()
        if output.is_symlink() or not output.resolve().is_relative_to(sandbox_root):
            raise ValueError("Generated output escaped its sandbox")
        if output.stat().st_size > 8_000_000:
            raise ValueError("Generated source artifact exceeds the size limit")
        return output.read_text()

    def _run_docker(self, *, config: Any, backend: Any, launcher: Any) -> Any:
        """Reuse Robocode's broker, CLI and accounting inside a disconnected Docker image."""
        broker = importlib.import_module("robocode.utils.model_broker")
        provider = broker.load_broker_upstream(self.settings.backend)
        container = DockerContainer(settings=self.settings.sandbox)
        launcher._setup_sandbox_dir(config)
        command = backend.build_cli_cmd(config)
        command, environment = self.model_client(backend=self.settings.backend, command=command)
        backend.setup_sandbox_files(config, docker_python=self.settings.sandbox.container_python)
        launcher._initial_commit(config.sandbox_dir)
        name = f"hitl-coder-{uuid.uuid4().hex}"
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="hitl-model-broker-") as tmp:
            transport = Path(tmp)
            shutil.copyfile(self.settings.sandbox.transport_source(), transport / "transport.py")
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
                    *self.cli_state_mounts(
                        sandbox_dir=config.sandbox_dir, backend=self.settings.backend
                    ),
                ),
                environment=environment,
                argv=[
                    "/run/hitl-broker/transport.py",
                    "/run/hitl-broker/transport.json",
                    *command,
                ],
            )
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
                timer = threading.Timer(
                    self.settings.timeout_seconds, lambda: container.remove(name=name)
                )
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
                        timer.start()
                        stream = backend.parse_stream(
                            process,
                            stream_log_path=config.sandbox_dir.parent / "stream.jsonl",
                            stderr_file=stderr,
                        )
                finally:
                    timer.cancel()
                    container.remove(name=name)
                    stderr.seek(0)
                    (config.sandbox_dir.parent / "container.stderr").write_text(stderr.read())
        launcher._final_commit(config.sandbox_dir)
        return launcher._stream_result_to_sandbox_result(
            stream,
            config.sandbox_dir,
            config.output_filename,
            wall_time_s=time.monotonic() - started,
        )

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
