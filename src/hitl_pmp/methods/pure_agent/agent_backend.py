import abc
import json
from typing import Any

from pydantic import BaseModel, PrivateAttr

from hitl_pmp.methods.pure_agent.types import AgentReply


class AgentBackend(BaseModel, abc.ABC):
    """A conversation with an agent: send a prompt, get a reply, and the reply may depend
    on everything sent before it.

    **`reset()` is the whole interface's reason for existing.** A `query` alone would be
    enough for a stateless policy; what this baseline needs is a conversation that
    accumulates *and* a way to say "forget everything, this is a new episode". Those two
    together are what make the practice/evaluation firewall expressible at all -- see
    `PureAgentMethod` for how the two backends and their resets are arranged.

    Two implementations plus a replay. `ScriptedAgentBackend` (below) is deterministic,
    needs no network and no optional dependency, and so exercises the whole acting path
    in CI at zero cost. `ClaudeCodeAgentBackend` (its own module, gated on
    `importlib.util.find_spec`) is the real one, and is the only part of this method that
    spends anything. `ReplayAgentBackend` reads back a recorded run."""

    @abc.abstractmethod
    def query(self, *, prompt: str) -> AgentReply:
        """Run one prompt to completion. Whatever has been sent since the last `reset()`
        is still in the conversation."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Start a new conversation. Everything said before is gone from the agent's
        view.

        Called at every evaluation episode boundary and at every practice period
        boundary. Not merely tidiness in the first case: without it, the agent would
        carry what it saw on test task k into test task k+1, which is learning across
        held-out tasks and is exactly what the firewall exists to prevent."""
        raise NotImplementedError

    def describe(self) -> str:
        """A short identifier, recorded so a run says what produced it -- a scripted run
        and a real one are very different evidence."""
        return type(self).__name__


class ScriptedAgentBackend(AgentBackend):
    """A deterministic stand-in that hands back a fixed cycle of replies and records
    every prompt it was given.

    **This is not a mock of the agent, it is a mock of the transport.** Everything
    downstream of `query()` -- parsing a reply into a decision, rejecting an out-of-range
    index, executing the chosen ground skill, tallying what it achieved, resetting at an
    episode boundary -- is the real code path, exercised for real. What it removes is only
    the network call, which is the one part that cannot run in CI and the one part that
    costs anything.

    `replies` CYCLES rather than running out, and that is deliberate here where the
    authoring arm's equivalent deliberately raised. A run makes one call per environment
    step, and the step count is a property of the domain and the horizon rather than
    something a test states up front; requiring a test to script 1,380 replies exactly
    would make every test a transcription exercise and would break on any change to the
    horizon. `prompts_seen()` is how a test asserts on the *count*, which is the thing
    worth asserting -- see the firewall tests.

    `reset_count` is public because the firewall is partly a claim about resets, and a
    claim a test cannot see is a claim that decays."""

    replies: tuple[str, ...] = ('{"skill_index": 0, "params": []}',)
    # Answers for the prompts that are not decision points. A run sends three kinds of
    # prompt (an opening, a decision, a digest request) and only one of them wants a JSON
    # decision back, so a test that needs a specific note out of the digest query has to
    # be able to answer *that* prompt without disturbing the decision cycle. Keyed by a
    # substring of the prompt; the first matching entry wins, and a match does not
    # advance `replies`.
    replies_by_marker: dict[str, str] = {}

    _index: int = PrivateAttr(default=0)
    _prompts: list[str] = PrivateAttr(default_factory=list)
    _reset_count: int = PrivateAttr(default=0)

    def query(self, *, prompt: str) -> AgentReply:
        self._prompts.append(prompt)
        for marker, reply in self.replies_by_marker.items():
            if marker in prompt:
                return AgentReply(text=reply, metadata={"total_cost_usd": 0.0, "num_turns": 1})
        reply = self.replies[self._index % len(self.replies)]
        self._index += 1
        return AgentReply(text=reply, metadata={"total_cost_usd": 0.0, "num_turns": 1})

    def reset(self) -> None:
        self._reset_count += 1

    def prompts_seen(self) -> list[str]:
        """Every prompt this backend has been given, in order -- a copy, so a caller
        holding it cannot have it grow underneath them."""
        return list(self._prompts)

    def reset_count(self) -> int:
        """How many times a new conversation was started on this backend."""
        return self._reset_count


class FirstApplicableAgentBackend(ScriptedAgentBackend):
    """A stand-in that always returns a LEGAL decision: the first applicable ground skill,
    with `param_dim` copies of `param_value`.

    **Fixed replies cannot do this, and that is why this exists.** A scripted
    `{"skill_index": 0, "params": [3.0]}` is a *parse failure* at every decision point
    whose first applicable skill happens to take no parameters, so a run scripted that way
    malforms every decision, executes nothing, and reports no practice outcomes -- which
    silently turns every downstream assertion into a test of the malformed path. This one
    reads the applicable set out of the prompt it was just handed, so its decisions are
    always executable and the tests above it are exercising the path they claim to.

    It is a genuinely terrible policy and makes no claim otherwise. What it is good for is
    being a *deterministic* one: it needs no network, no optional dependency and no seed,
    and it drives the whole method -- grounding, execution, add-effect tallying, the
    firewall -- for real."""

    param_value: float = 0.0

    def query(self, *, prompt: str) -> AgentReply:
        skills = self.applicable_skills(prompt=prompt)
        if not skills:
            # An opening or a digest request: no observation in it, so there is no
            # decision to make. Fall back to the scripted reply, which is what those
            # prompts want anyway -- and which records the prompt itself, so this branch
            # must not record it too.
            return super().query(prompt=prompt)
        self._prompts.append(prompt)
        params = [self.param_value] * int(skills[0].get("param_dim", 0))
        return AgentReply(
            text=json.dumps({"skill_index": 0, "params": params}),
            metadata={"total_cost_usd": 0.0, "num_turns": 1},
        )

    @staticmethod
    def applicable_skills(*, prompt: str) -> list[dict[str, Any]]:
        """The `skills` list of the observation this prompt carries, or `[]` if it carries
        none. Parsed back out of the prompt rather than passed in beside it, because that
        is exactly what the real agent has to do -- a stand-in given a private side channel
        would be exercising a path production does not have."""
        marker = "observation = "
        index = prompt.rfind(marker)
        if index < 0:
            return []
        try:
            observation = json.loads(prompt[index + len(marker) :].strip())
        except json.JSONDecodeError:
            return []
        skills = observation.get("skills", []) if isinstance(observation, dict) else []
        return skills if isinstance(skills, list) else []
