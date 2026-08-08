import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from hitl_pmp.methods.pure_agent.agent_backend import AgentBackend
from hitl_pmp.methods.pure_agent.types import AgentReply

# Where `prpl-agent-utils` streams the CLI's own JSON messages, one per line. Read only
# when a query raised -- see `recover_metadata`.
STREAM_LOG = Path(".agent_logs") / "stream.jsonl"


class ClaudeCodeAgentBackend(AgentBackend):
    """The real backend: `prpl-agent-utils`' `ClaudeCodeAgent`, driving the Claude Code
    CLI once per environment step.

    **The one module in this package that touches `prpl_agent_utils`, and the import is
    lazy** -- the same discipline `environments/tossing3d/kinder_backend.py` applies to
    KINDER, and for the same reason. `hitl_pmp/cli.py` imports every method-CLI at module
    import time, so a top-level third-party import here would make `--help` fail on any
    machine without the optional dependency. CI is exactly such a machine. Everything
    above this file talks to `AgentBackend` in plain types.

    **`tools` defaults to none, and that is the design rather than a hardening measure.**
    This agent is a policy: it is handed an observation and must reply with one line of
    JSON. It has nothing to read, nothing to write, and no reason to run a command, so
    giving it tools would only buy latency and a class of failure (a turn spent exploring
    an empty sandbox) that has no upside. It also makes each call a single assistant turn,
    which is what keeps a per-step agent affordable at all -- measured at ~2.4-4.2 s and
    well under a cent per call on Opus.

    **`use_docker` defaults to False, which is a deviation and is stated as one.** The
    package's own default is True and that is its safe mode. It is off here because this
    machine's Docker socket is not accessible to this user (`permission denied ...
    /var/run/docker.sock`, verified), so `True` cannot run at all; and because the
    argument for the container is containment of an agent that writes files and runs
    commands, which this one does not do -- with no tools, the only thing the CLI can
    emit is text. Turn it back on wherever Docker is available.

    **The conversation persists across queries and is cleared by `reset()`.**
    `ClaudeCodeAgent` resumes via `--continue` against a session stored under the sandbox,
    so query k+1 sees query k. That is the entire online-learning channel within a period
    or an episode. Both come from constructing the agent ONCE and holding it, which is why
    this is a stateful object rather than a function."""

    sandbox_dir: Path
    model: str = "opus"
    use_docker: bool = False
    tools: str = ""
    # 0 disables the CLI's own budget stop. Deliberate: the stop emits a `result` message
    # with no `result` field, which `prpl_agent_utils._parse_stream` treats as fatal, so a
    # per-step policy would lose the very decision it just paid for. A per-step query is
    # one short turn, so the cap protects against nothing here and costs a decision when
    # it fires.
    max_budget_usd_per_query: float = 0.0
    system_prompt: str = ""

    # Built on first use, not at construction: constructing a backend must not create a
    # sandbox directory or reach for credentials, since `--help` and a config snapshot
    # both construct one and neither should touch the filesystem.
    _agent: Any = PrivateAttr(default=None)

    def get_agent(self) -> Any:
        """The underlying `ClaudeCodeAgent`, constructed once and reused.

        Raises with an actionable message rather than an `ImportError` traceback if the
        dependency is missing, because "install prpl-agent-utils" is the whole fix and a
        stack trace buries it."""
        if self._agent is not None:
            return self._agent
        if importlib.util.find_spec("prpl_agent_utils") is None:
            raise RuntimeError(
                "--method pure-agent needs `prpl_agent_utils`, which is not installed. "
                "It is an optional dependency with no dependencies of its own: "
                "`pip install <prpl-mono>/prpl-agent-utils`. Replaying a recorded run "
                "(--pure-agent-replay) never needs it."
            )
        agents = importlib.import_module("prpl_agent_utils.agents")
        self._agent = agents.ClaudeCodeAgent(
            self.sandbox_dir,
            model=self.model,
            use_docker=self.use_docker,
            system_prompt=self.system_prompt,
            max_budget_usd_per_query=self.max_budget_usd_per_query,
            tools=self.tools,
        )
        return self._agent

    def query(self, *, prompt: str) -> AgentReply:
        """One paid call. A query that RAISES still counts as a call.

        `prpl_agent_utils` raises `RuntimeError("Agent CLI produced no result")` whenever
        the CLI exits without a parseable final `result` -- a transport failure, a budget
        stop, a killed process. Letting that propagate would abort a run several hours and
        several thousand calls in, and would lose the cost of the call that failed. So it
        is caught, the cost is recovered from the stream log the package itself wrote, and
        an empty reply is returned: the caller then records a malformed decision and takes
        a no-op step, which is a visible, counted outcome rather than a crash."""
        try:
            response = self.get_agent().query(prompt)
        except RuntimeError as exc:
            metadata = self.recover_metadata()
            metadata["query_error"] = f"{type(exc).__name__}: {exc}"
            return AgentReply(text="", metadata=metadata)
        # Converted at this boundary rather than passed through: `AgentResponse` is a
        # `dataclass`, which ruff TID251 bans in this project, and nothing above this file
        # should have a type from an optional dependency in its signature.
        return AgentReply(text=response.text, metadata=dict(response.metadata))

    def reset(self) -> None:
        """Start a new CLI conversation. The sandbox's files are kept (there are none
        that matter here); only `--continue` stops being passed.

        A no-op before the first query, so an episode boundary at the very start of a run
        costs nothing."""
        if self._agent is not None:
            self._agent.reset()

    def recover_metadata(self) -> dict[str, Any]:
        """The last `result` message in the sandbox's stream log, as metadata.

        Reading the artifact the package already writes, rather than reimplementing its
        parser: the log is appended per query, so the last `result` line is this query's.
        Returns `{}` if there is no log or nothing parseable in it, which leaves the
        call's cost `None` -- honestly unknown, and counted by
        `PureAgentLedger.num_calls_missing_cost` rather than silently read as free."""
        path = self.sandbox_dir / STREAM_LOG
        if not path.is_file():
            return {}
        latest: dict[str, Any] = {}
        for line in path.read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("type") == "result":
                latest = message
        if not latest:
            return {}
        return {
            "is_error": latest.get("is_error", True),
            "num_turns": latest.get("num_turns"),
            "total_cost_usd": latest.get("total_cost_usd"),
            "stop_reason": latest.get("subtype"),
        }

    def describe(self) -> str:
        docker = "docker" if self.use_docker else "host"
        return f"claude-code-{self.model}-{docker}"
