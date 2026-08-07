from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AuthoringTranscript(BaseModel):
    """Everything one authoring run produced: the ordered rounds, and the derived
    totals a report has to quote.

    **This is the run artifact record-then-replay turns on.** Authoring is
    nondeterministic -- the agent is queried, and the same prompt does not give the same
    file twice -- so a measured run must never author. The transcript is written once by
    an authoring run and read back by every replay, which then makes no API call at all
    and produces a byte-stable `stats.json`.

    It carries **every** round rather than only the final policy, and that is the
    load-bearing part. A replay holding only the last round would evaluate a fully-revised
    policy at every checkpoint, flattening the learning curve into a horizontal line that
    is indistinguishable from a method that converged immediately. `policy_sources()`
    below is the sequence a replay walks, one entry per round, in order."""

    rounds: list[AuthoringRound] = Field(default_factory=list)
    # Every call into the authored policy, over both phases. NOT a cost of this arm --
    # this arm queries the agent once per round, never per decision -- but the price of
    # the *be-the-policy* variant the blog proposes and the notebook refuses, which would
    # make exactly one API call per decision. Recorded here so that arm can be priced
    # from a run that has already happened rather than by building it. See
    # `PureAgentMethod.num_decisions`.
    num_decisions: int = 0
    # Decisions the authored policy returned but that could not be executed: an
    # out-of-range skill index, a parameter vector of the wrong width, a non-finite
    # parameter, or a raised exception. Each became a no-op. Reported beside
    # `num_decisions` because a policy that "ran" while malforming half its decisions is
    # a very different object from one that ran cleanly, and the success rate alone
    # cannot tell them apart.
    num_malformed_decisions: int = 0

    def policy_sources(self) -> tuple[str | None, ...]:
        """The per-round `policy.py` contents, in order -- exactly what a replay consumes.

        `None` for a round the agent failed to deliver a file for. Kept in the sequence
        rather than dropped: the round happened, it left the previous policy in effect,
        and a replay has to reproduce that same gap to reproduce the same curve."""
        return tuple(round_.policy_source for round_ in self.rounds)

    def total_cost_usd(self) -> float:
        """What this authoring run actually spent, summed over rounds.

        Rounds whose backend reported no cost (the scripted backend, and any real query
        whose `result` message omitted the field) contribute 0.0, so this is a lower
        bound rather than an estimate -- state it as one. `num_rounds_missing_cost`
        below is how a reader tells a genuine zero from an unreported one."""
        return sum(round_.total_cost_usd or 0.0 for round_ in self.rounds)

    def num_rounds_missing_cost(self) -> int:
        """Rounds that reported no cost at all, so `total_cost_usd` under-counts by an
        unknown amount rather than by zero."""
        return sum(1 for round_ in self.rounds if round_.total_cost_usd is None)

    def num_failed_rounds(self) -> int:
        """Rounds whose authored file could not be loaded and used."""
        return sum(1 for round_ in self.rounds if round_.load_error is not None)


class AuthoringRound(BaseModel):
    """One query to the agent and what came back: the prompt sent, the file found in the
    sandbox afterwards, whether it loaded, and what the query cost.

    **One query per round, never a repair loop inside one.** A failed load does not
    re-query immediately; it leaves the previous policy in effect and makes the *next*
    round's prompt an error prompt. That is the notebook's own shape, and it is what
    keeps a round a 1:1 match for a `PracticeLoop` cycle -- a variable number of queries
    per round would make the authored-policy sequence no longer indexable by round, which
    is exactly what replay indexes by.

    Frozen: a round is a record of something that already happened."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    prompt: str
    # The `policy.py` the sandbox held after the query, verbatim. `None` when the agent
    # wrote no file at all -- distinct from a file that exists and does not import, which
    # is a non-None source with a non-None `load_error`.
    policy_source: str | None
    # `None` when the source loaded and passed its probe call. Otherwise the rendered
    # exception, which is also what the next round's prompt quotes back.
    load_error: str | None = None
    # The agent's final message. Kept because it is often the only place the agent says
    # what it believed it was doing, which is the thing a reader most wants when the
    # authored code is surprising.
    agent_text: str = ""
    # Straight from `AgentResponse.metadata`. `None` rather than 0.0 when the backend
    # reported nothing, so an unreported cost never silently reads as a free query.
    total_cost_usd: float | None = None
    num_turns: int | None = None
    num_tool_calls: int | None = None


class AgentReply(BaseModel):
    """One backend query's result, in this project's own shape.

    Deliberately not `prpl_agent_utils.AgentResponse` itself, for two reasons. That type
    is a `dataclass`, which ruff `TID251` bans here; and depending on it directly would
    put an optional third-party import in the path of `ScriptedAgentBackend`, which
    exists precisely so the whole method is exercisable in CI with no such dependency
    installed. The real backend converts at its own boundary."""

    model_config = ConfigDict(frozen=True)

    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillChoice(BaseModel):
    """What an authored policy returns: which of the *applicable* ground skills to
    execute, and the continuous parameters to execute it with.

    `skill_index` indexes the `skills` list of the observation the policy was just
    handed -- not the domain's lifted-skill list. Indexing the applicable set is what
    makes an authored policy unable to select a skill whose preconditions do not hold,
    so the one failure mode it cannot have is the one that would silently degrade into
    a no-op at the raw-action layer and look like a bad choice instead of an illegal
    one.

    Frozen and validated here rather than trusted: this is the one place agent-authored
    output crosses into the harness."""

    model_config = ConfigDict(frozen=True)

    skill_index: int
    params: tuple[float, ...] = ()
