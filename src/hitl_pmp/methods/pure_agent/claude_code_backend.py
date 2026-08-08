import importlib
import importlib.util
import json
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr

from hitl_pmp.methods.pure_agent.agent_backend import AgentBackend
from hitl_pmp.methods.pure_agent.types import AgentReply

# Where `prpl-agent-utils` streams the CLI's own JSON messages, one per line. Read only
# when a query raised -- see `recover_from_stream_log`.
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
    # The CLI's own per-query stop. **On rather than off, and the known defect is handled
    # rather than avoided.** The stop emits a `result` message with
    # `subtype: "error_max_budget_usd"` and no `result` field, which
    # `prpl_agent_utils._parse_stream` treats as fatal -- so naively enabling it discards
    # the very decision the query just paid for. `query` below recovers both the cost and
    # the agent's last assistant text from the stream log, so a capped call that had
    # already answered still yields its answer.
    #
    # **2.00, set from measurement rather than from taste, and 0.50 was measurably wrong.**
    # A decision's cost is dominated by the conversation it resumes into, so it climbs
    # with the period: over one 50-step practice period the baseline went $0.017 -> $0.075
    # and the tail reached **$0.597** -- already past 0.50, on a period a third of the
    # default length. A cap that fires routinely is not a guard, it is a source of no-ops
    # dressed up as one.
    #
    # This is the per-call guard and it is deliberately loose: its only job is to stop one
    # pathological call, not to shape normal cost. `PureAgentMethod.max_total_cost_usd` is
    # the run-level ceiling and the one that actually protects the allowance, which is why
    # that one is required on the CLI and this one is merely defaulted.
    max_budget_usd_per_query: float = 2.00
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
        """One paid call. A query that RAISES still counts as a call, and is salvaged.

        `prpl_agent_utils` raises `RuntimeError("Agent CLI produced no result")` whenever
        the CLI exits without a parseable final `result` -- a transport failure, a killed
        process, and **every `--max-budget-usd` stop**, which reports its own cost and then
        exits in a shape the package treats as fatal.

        Letting that propagate would be wrong three times over: it aborts a run thousands of
        calls in, it discards a decision already paid for, and it makes the spend this
        baseline exists to report unknowable. So it is caught and the stream log the package
        itself already wrote is read back for both halves -- the cost from the `result`
        message, the answer from the last assistant text. `query_error` in the metadata is
        how a call that took this path stays visible rather than passing for a clean one."""
        try:
            response = self.get_agent().query(prompt)
        except RuntimeError as exc:
            text, metadata = self.recover_from_stream_log()
            metadata["query_error"] = f"{type(exc).__name__}: {exc}"
            return AgentReply(text=text, metadata=metadata)
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

    def recover_from_stream_log(self) -> tuple[str, dict[str, Any]]:
        """`(the agent's last assistant text, metadata)` from the sandbox's stream log.

        Reading the artifact the package already writes, rather than reimplementing its
        parser: the log is appended per query, so the last `result` line and the last
        assistant text after it are this query's.

        **Recovering the text is what makes a budget cap usable.** A capped query that
        already emitted its one line of JSON has done everything it was asked to do; the
        CLI merely exits in a shape the package cannot parse. Reading it back turns what
        would be a discarded, paid-for decision into the decision it actually was. When the
        cap fires before any answer, the text is empty and the caller counts a malformed
        decision -- honest, and visible in the ledger.

        Returns `("", {})` if there is no log or nothing parseable in it, which leaves the
        call's cost `None` -- honestly unknown, and counted by
        `PureAgentLedger.num_calls_missing_cost` rather than silently read as free."""
        path = self.sandbox_dir / STREAM_LOG
        if not path.is_file():
            return "", {}
        # A `result` line CLOSES a call, so the assistant text belonging to it is whatever
        # was seen since the previous one. Both are tracked, and the closed pair is kept
        # together: reading the last text and the last result independently would pair this
        # call's answer with the previous call's cost the moment a stream ends without a
        # result, which is exactly the case this function exists for.
        text_since_last_result = ""
        closed_text = ""
        closed_result: dict[str, Any] = {}
        for line in path.read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get("type") == "result":
                closed_text, closed_result = text_since_last_result, message
                text_since_last_result = ""
            elif message.get("type") == "assistant":
                for block in message.get("message", {}).get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_since_last_result = str(block.get("text", ""))
        if text_since_last_result:
            # Text with no result after it: the CLI died mid-stream. The answer is real,
            # the cost is genuinely unknown, and saying so is better than attributing the
            # previous call's cost to this one.
            return text_since_last_result, {}
        if not closed_result:
            return "", {}
        return closed_text, {
            "is_error": closed_result.get("is_error", True),
            "num_turns": closed_result.get("num_turns"),
            "total_cost_usd": closed_result.get("total_cost_usd"),
            "stop_reason": closed_result.get("subtype"),
        }

    def describe(self) -> str:
        docker = "docker" if self.use_docker else "host"
        return f"claude-code-{self.model}-{docker}"
