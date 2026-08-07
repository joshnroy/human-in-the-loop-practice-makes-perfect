import abc

from pydantic import BaseModel, PrivateAttr

from hitl_pmp.methods.pure_agent.types import AgentReply


class AgentBackend(BaseModel, abc.ABC):
    """Where an authored `policy.py` comes from: something that takes a prompt and
    afterwards has a file to read.

    Two implementations, and the split is what makes this method testable at all.
    `ScriptedAgentBackend` (below) is deterministic, needs no network, no Docker and no
    API key, and so exercises the entire authoring path in CI at zero cost.
    `ClaudeCodeAgentBackend` (its own module, gated on `importlib.util.find_spec`) is the
    real one, and is the only part of this method that spends money.

    The two-call shape -- `query` then `policy_source` -- rather than a single
    `query() -> str` mirrors what the real agent actually does: it does not *return*
    code, it writes a file into a sandbox that persists between queries, and the final
    message it returns is prose about what it did. Collapsing the two would mean parsing
    code out of that prose, which is the fragile thing this interface exists to avoid."""

    @abc.abstractmethod
    def query(self, *, prompt: str) -> AgentReply:
        """Run one prompt to completion. The conversation persists to the next call, so
        a later query sees the score for the code it just wrote without being re-told
        the problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def policy_source(self) -> str | None:
        """The current contents of `policy.py` in the agent's workspace, or `None` if
        there is no such file. Read *after* a query, never parsed out of its text."""
        raise NotImplementedError

    def describe(self) -> str:
        """A short identifier for this backend, recorded so a transcript says what
        produced it -- a replay of a scripted transcript and a replay of a real one are
        very different evidence."""
        return type(self).__name__


class ScriptedAgentBackend(AgentBackend):
    """A deterministic stand-in that hands back a fixed sequence of `policy.py` contents,
    one per query, and records the prompts it was given.

    **This is not a mock of the agent, it is a mock of the API.** Everything downstream of
    `policy_source()` -- loading agent-written code, probing it, running it against the
    observation contract, recovering from a file that does not import -- is the real code
    path, exercised for real. What it removes is only the network call, which is the one
    part that cannot run in CI and the one part that costs money.

    `sources` may contain `None`, which stands for "the agent wrote no file this round",
    the failure mode a real agent hits when it spends its whole budget reasoning. Running
    off the end of `sources` raises rather than repeating the last entry: a test that
    queries more times than it scripted is a test whose cadence has drifted from the
    harness's, and silently reusing an entry would hide that."""

    sources: tuple[str | None, ...]

    _index: int = PrivateAttr(default=0)
    _prompts: list[str] = PrivateAttr(default_factory=list)

    def query(self, *, prompt: str) -> AgentReply:
        if self._index >= len(self.sources):
            raise RuntimeError(
                f"ScriptedAgentBackend was queried {self._index + 1} times but only "
                f"{len(self.sources)} sources were scripted. Extend `sources` rather than "
                "letting the last one repeat -- a repeat would silently model an agent "
                "that declined to revise."
            )
        self._prompts.append(prompt)
        self._index += 1
        return AgentReply(text=f"scripted round {self._index - 1}")

    def policy_source(self) -> str | None:
        """Whatever the most recent query "wrote". Before any query there is no file, the
        same as a fresh sandbox."""
        if self._index == 0:
            return None
        return self.sources[self._index - 1]

    def prompts_seen(self) -> list[str]:
        """Every prompt this backend has been given, in order -- a copy, so a caller
        holding it cannot have it grow underneath them."""
        return list(self._prompts)
