"""The coding agent: one Claude Code session per run, continued at every refit.

`RobocodeCoder` drives it through the agentic runtime's `RobocodeDockerTransport`, the
same disconnected container, host-owned model broker and session-resume logic the
Full-Agentic method uses: the first refit starts the session, every later refit passes
`resume_previous_session`, which resumes the session id recorded in `stream.jsonl`.
"""

import abc
import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from hitl_pmp.agentic_runtime.docker_cli import RobocodeDockerTransport
from hitl_pmp.agentic_runtime.sandbox import SandboxSettings

from .types import CoderResult, CompactionEvent


class Coder(BaseModel, abc.ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def run(self, *, prompt: str, system_prompt: str, resume: bool) -> CoderResult:
        """One invocation in the persistent workspace; `resume` continues the session."""


class StreamLog:
    """What one invocation appended to the CLI's `stream.jsonl`. A static container."""

    @staticmethod
    def summarize(*, lines: list[str]) -> dict[str, Any]:
        session_id = None
        compactions: list[CompactionEvent] = []
        context_tokens = None
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("parent_tool_use_id"):
                continue
            if event.get("type") in {"system", "result"} and event.get("session_id"):
                session_id = event["session_id"]
            if event.get("type") == "system" and event.get("subtype") == "compact_boundary":
                metadata = event.get("compact_metadata") or {}
                compactions.append(
                    CompactionEvent(
                        trigger=metadata.get("trigger"),
                        pre_tokens=metadata.get("pre_tokens"),
                        post_tokens=metadata.get("post_tokens"),
                    )
                )
            if event.get("type") == "assistant":
                usage = (event.get("message") or {}).get("usage") or {}
                if usage:
                    context_tokens = (
                        int(usage.get("input_tokens", 0) or 0)
                        + int(usage.get("cache_read_input_tokens", 0) or 0)
                        + int(usage.get("cache_creation_input_tokens", 0) or 0)
                    )
        return {
            "session_id": session_id,
            "compactions": compactions,
            "final_context_tokens": context_tokens,
        }


class EffortBackend:
    """Robocode's Claude backend with `--effort` added: upstream's builder has no
    effort argument, so without this the requested effort would silently not apply."""

    def __init__(self, *, backend: Any, effort: str) -> None:
        self._backend = backend
        self._effort = effort

    def build_cli_cmd(self, *args: Any, **kwargs: Any) -> list[str]:
        return [*self._backend.build_cli_cmd(*args, **kwargs), "--effort", self._effort]

    def __getattr__(self, name: str) -> Any:  # noqa: PLR0917 (dunder)
        return getattr(self._backend, name)


class RobocodeCoder(Coder):
    settings: SandboxSettings
    sandbox_dir: Path
    model: str
    effort: str
    timeout_seconds: float | None = None
    max_budget_usd: float = 0.0

    def run(self, *, prompt: str, system_prompt: str, resume: bool) -> CoderResult:
        source_root = str(self.settings.robocode_checkout.resolve() / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        transport = RobocodeDockerTransport(
            sandbox=self.settings,
            backend="claude",
            timeout_seconds=self.timeout_seconds,
            subagent_policy="legacy",
        )
        launcher, backends, omega = transport.modules()
        config = launcher.SandboxConfig(
            sandbox_dir=self.sandbox_dir,
            init_files={},
            prompt=prompt,
            system_prompt=system_prompt,
            output_filename="CLAUDE.md",
            model=self.model,
            max_budget_usd=self.max_budget_usd,
            max_turns=0,
            max_output_tokens=32768,
            autocompact_pct=80,
            blackbox=True,
            mcp_tools=(),
            resume_previous_session=resume,
        )
        backend = EffortBackend(
            backend=backends.create_backend(
                omega.DictConfig(dict(backend="claude", model=self.model, effort=self.effort))
            ),
            effort=self.effort,
        )
        stream = self.sandbox_dir.parent / "stream.jsonl"
        offset = len(stream.read_text().splitlines()) if stream.exists() else 0
        started = time.monotonic()
        try:
            result = transport.run_with_recovery(config=config, backend=backend, launcher=launcher)
        except Exception as error:  # noqa: BLE001 -- reported, the run falls back
            summary = StreamLog.summarize(
                lines=stream.read_text().splitlines()[offset:] if stream.exists() else []
            )
            return CoderResult(
                success=False,
                error=f"{type(error).__name__}: {error}"[:4000],
                resumed=resume,
                wall_seconds=time.monotonic() - started,
                **summary,
            )
        summary = StreamLog.summarize(lines=stream.read_text().splitlines()[offset:])
        metrics = result.generation_metrics
        return CoderResult(
            success=bool(result.success),
            error=result.error,
            resumed=resume,
            wall_seconds=time.monotonic() - started,
            cost_usd=result.total_cost_usd,
            input_tokens=int(getattr(metrics, "input_tokens", 0) or 0),
            output_tokens=int(getattr(metrics, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(metrics, "cache_read_tokens", 0) or 0),
            cache_creation_tokens=int(getattr(metrics, "cache_creation_tokens", 0) or 0),
            num_turns=int(getattr(metrics, "num_turns", 0) or 0),
            rate_limit_retries=int(getattr(metrics, "rate_limit_retries", 0) or 0),
            **summary,
        )
