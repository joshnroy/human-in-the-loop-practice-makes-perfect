import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from hitl_pmp.methods.pure_agent.agent_backend import AgentBackend
from hitl_pmp.methods.pure_agent.types import AgentReply

# The file the agent is told to write, and the only one read back out of the sandbox.
POLICY_FILENAME = "policy.py"
# Where `prpl-agent-utils` streams the CLI's own JSON messages, one per line. Read only
# when a query raised -- see `recover_metadata`.
STREAM_LOG = Path(".agent_logs") / "stream.jsonl"


class ClaudeCodeAgentBackend(AgentBackend):
    """The real backend: `prpl-agent-utils`' `ClaudeCodeAgent`, which runs the Claude Code
    CLI inside a Docker sandbox whose only writable host path is the agent's own directory.

    **The one module in this package that touches `prpl_agent_utils`, and the import is
    lazy** -- the same discipline `environments/tossing3d/kinder_backend.py` applies to
    KINDER, and for the same reason. `hitl_pmp/cli.py` imports every method-CLI at module
    import time, so a top-level third-party import here would make `--help` fail on any
    machine without the optional dependency. CI is exactly such a machine. Everything
    above this file talks to `AgentBackend` in plain types.

    **Docker is on by default and that is deliberate.** `use_docker=True` is the package
    default and its safe mode: the container's firewall restricts network access to the
    Anthropic API, GitHub and PyPI, and the agent cannot write outside its sandbox at all.
    `use_docker=False` keeps only a `PreToolUse` hook blocking writes elsewhere, while
    shell commands can still *read* the host filesystem. The flag exists because a machine
    without Docker should fail loudly rather than silently downgrade, not because turning
    it off is a reasonable default.

    **The sandbox and the conversation both persist across queries.** That is what makes
    the revise loop work at all: round k's `policy.py` is there to be edited in round k+1,
    and the agent sees the feedback for the code it just wrote without being re-told the
    problem. Both come from constructing the agent ONCE and holding it, which is why this
    is a stateful object rather than a function."""

    sandbox_dir: Path
    model: str = "sonnet"
    use_docker: bool = True
    # 1.0 is `prpl-agent-utils`' own notebook value (its constructor default is 5.0). The
    # notebook runs 3 rounds at this cap and reports a few tens of cents; a run here is
    # `--num-cycles + 1` rounds, so the cap is what bounds the worst case.
    max_budget_usd_per_query: float = 1.0
    system_prompt: str = ""

    # Built on first use, not at construction: constructing a backend must not create a
    # sandbox directory or reach for credentials, since a `--help` or a config snapshot
    # constructs one and neither should touch the filesystem.
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
                "--method pure-agent-author needs `prpl_agent_utils`, which is not "
                "installed. It is an optional dependency with no dependencies of its own: "
                "`pip install <prpl-mono>/prpl-agent-utils`. Replaying a recorded "
                "transcript (--method pure-agent) never needs it."
            )
        agents = importlib.import_module("prpl_agent_utils.agents")
        self._agent = agents.ClaudeCodeAgent(
            self.sandbox_dir,
            model=self.model,
            use_docker=self.use_docker,
            system_prompt=self.system_prompt,
            max_budget_usd_per_query=self.max_budget_usd_per_query,
        )
        return self._agent

    def query(self, *, prompt: str) -> AgentReply:
        """One paid query. A query that RAISES still counts as a round.

        **Measured upstream behaviour, not a hypothetical.** When the CLI stops on its own
        `--max-budget-usd` cap it emits a final `result` message with
        `subtype: "error_max_budget_usd"` and **no `result` field**, then exits 1.
        `prpl_agent_utils._parse_stream` requires that field, so it raises
        `RuntimeError("Agent CLI produced no result")` -- discarding a round whose
        `policy.py` was already written to the sandbox and whose exact cost the CLI just
        reported. Observed on the first smoke run of this backend: a $0.886 query that
        produced a 5.6 KB policy, thrown away.

        Letting that raise would be wrong twice over: it loses work already paid for, and
        it makes the spend this baseline is supposed to report unknowable. So the failure
        is caught, the file is read as usual (it is on disk regardless of how the CLI
        exited), and the cost is recovered from the stream log the package itself wrote.
        `query_error` in the metadata is how a round that took this path stays visible
        rather than passing for a clean one."""
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

    def recover_metadata(self) -> dict[str, Any]:
        """The last `result` message in the sandbox's stream log, as metadata.

        Reading the artifact the package already writes, rather than reimplementing its
        parser: the log is appended per query, so the last `result` line is this query's.
        Returns `{}` if there is no log or nothing parseable in it, which leaves the
        round's cost `None` -- honestly unknown, and counted by
        `AuthoringTranscript.num_rounds_missing_cost` rather than silently read as free."""
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

    def policy_source(self) -> str | None:
        path = self.sandbox_dir / POLICY_FILENAME
        if not path.is_file():
            return None
        return path.read_text()

    def describe(self) -> str:
        docker = "docker" if self.use_docker else "host"
        return f"claude-code-{self.model}-{docker}"
