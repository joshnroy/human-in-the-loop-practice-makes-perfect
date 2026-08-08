from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentCallRecord(BaseModel):
    """One query to the agent and everything it cost, appended to the run's ledger the
    moment it returns.

    **The ledger is written incrementally, one JSON line per call, never assembled at
    the end.** A run of this baseline is thousands of sequential agent calls over
    several hours; assembling the record in memory and writing it once means a crash at
    hour four reports nothing about the money already spent. Spend is a hard deliverable
    here, so it is journalled.

    **`cost_usd` is subscription allowance, not an invoice.** The Claude Code CLI reports
    `total_cost_usd` for every query, and `prpl_agent_utils` passes it straight through;
    on this project the CLI authenticates against a Claude subscription
    (`claude_auth.py` brokers `~/.claude/.credentials.json`, and no `ANTHROPIC_API_KEY`
    is set), so nothing here is billed at these figures. They are the API-equivalent
    price of the tokens, which is the only quantity the CLI reports and the right one for
    comparing arms against each other -- but a total must always be labelled as
    allowance, never presented as money owed.

    `None` rather than `0.0` when the CLI reported no cost, so an unreported query never
    silently reads as a free one. `PureAgentLedger.num_calls_missing_cost` is how a
    reader tells the two apart.

    Frozen: a record is a record of something that already happened."""

    model_config = ConfigDict(frozen=True)

    # Which side of the firewall this call happened on. The single most important field
    # in the record: it is what makes "did anything measured during evaluation reach the
    # agent?" answerable from the artifact rather than only from reading the code.
    phase: AgentPhase
    # What the call was for. Orthogonal to `phase` on purpose -- the digest call is a
    # PRACTICE call, because it runs on the practice conversation and can therefore only
    # carry what practice saw, and folding it into a third phase would hide that.
    #
    # Required rather than defaulted to DECISION, which the file's top-down ordering
    # would otherwise make impossible anyway: a default VALUE is evaluated when the class
    # body runs, so `AgentCallKind.DECISION` here would need the enum defined above this
    # class. Requiring it is the better answer regardless -- every call site knows which
    # kind it is making, and a defaulted one would silently mislabel a new call site.
    kind: AgentCallKind
    # The practice cycle this call belongs to. 0 during the evaluation sweep that runs
    # before any practice, so a sweep and the cycle it follows share an index.
    cycle_index: int
    # Which test task within the sweep, for an evaluation call; `None` during practice,
    # where there is no episode structure inside a period.
    episode_index: int | None = None
    # Step within the episode (evaluation) or within the practice period (practice).
    step_index: int
    # SHA-256 over the canonical JSON of the observation this decision was made against.
    # Not the observation itself: the ledger is one line per call over thousands of
    # calls, and a Tossing Room observation is ~2 KB. The digest is what a replay checks
    # itself against -- see `ReplayAgentBackend`. Empty for a call that is not a decision
    # (an opening, a digest request), which has no observation behind it.
    observation_digest: str = ""
    # What the agent replied, verbatim but capped. Kept because it is the only place the
    # agent says why it chose what it chose, and a run whose decisions look strange is
    # exactly when someone wants to read them.
    reply_text: str = ""
    # The parsed decision, or `None` when the reply could not be parsed into one. A
    # `None` here with a `parse_error` beside it is the malformed-decision failure mode
    # this baseline is allowed to have and must report, not hide.
    skill_index: int | None = None
    params: tuple[float, ...] = ()
    parse_error: str | None = None
    # Wall-clock seconds this one call took, including CLI process startup.
    seconds: float = 0.0
    total_cost_usd: float | None = None
    num_turns: int | None = None
    num_tool_calls: int | None = None
    stop_reason: str | None = None
    # Set when the backend's query itself raised but the call was salvaged. Recorded so a
    # call that was cut off stays distinguishable from one that ran to completion.
    query_error: str | None = None


class PureAgentLedger(BaseModel):
    """Every `AgentCallRecord` of one run, plus the totals a report has to quote.

    Read back from the run's `agent_calls.jsonl` by `analysis/`; never assembled during
    the run itself (see `AgentCallRecord` for why the file is journalled instead)."""

    records: list[AgentCallRecord] = Field(default_factory=list)

    def total_cost_usd(self) -> float:
        """Subscription allowance consumed, summed over calls. Calls that reported no
        cost contribute 0.0, so this is a LOWER BOUND rather than an estimate -- state it
        as one, and quote `num_calls_missing_cost` beside it."""
        return sum(record.total_cost_usd or 0.0 for record in self.records)

    def num_calls_missing_cost(self) -> int:
        """Calls that reported no cost at all, so `total_cost_usd` under-counts by an
        unknown amount rather than by zero."""
        return sum(1 for record in self.records if record.total_cost_usd is None)

    def num_calls(self, *, phase: AgentPhase | None = None) -> int:
        """How many agent calls this run made, optionally on one side of the firewall."""
        if phase is None:
            return len(self.records)
        return sum(1 for record in self.records if record.phase is phase)

    def num_decisions(self, *, phase: AgentPhase | None = None) -> int:
        """Calls that were actual decision points, i.e. one environment step each.
        Distinct from `num_calls`, which also counts the per-episode opening and the
        per-cycle digest -- overhead that is real spend but is not a step."""
        return sum(
            1
            for record in self.records
            if record.kind is AgentCallKind.DECISION and (phase is None or record.phase is phase)
        )

    def num_malformed_decisions(self) -> int:
        """Replies that could not be parsed into a decision, over the whole run. Each
        became a no-op step. Reported beside the success rate because a run that acted
        while malforming half its decisions is a very different object from one that
        acted cleanly, and the success rate alone cannot tell them apart."""
        return sum(1 for record in self.records if record.parse_error is not None)


class AgentPhase(str, enum.Enum):
    """Which side of the practice/evaluation firewall an agent call happened on.

    `(str, Enum)` so a member compares equal to its own wire string and lands in the
    ledger as a readable word rather than an integer nobody can interpret later.

    The two members are not two kinds of prompt, they are two kinds of *permission*.
    `PRACTICE` may be told outcomes and may accumulate across the period. `EVALUATION`
    may be told nothing that measurement produced, and everything it accumulates is
    discarded at the end of its episode."""

    PRACTICE = "practice"
    EVALUATION = "evaluation"

    def __str__(self) -> str:
        return self.value


class AgentCallKind(str, enum.Enum):
    """What one agent call was for.

    Separate from `AgentPhase` so the spend table can say how much of a run's cost was
    decisions and how much was the overhead of starting each episode's conversation --
    two numbers that scale with completely different things (`DECISION` with the horizon,
    `OPENING` with the number of episodes)."""

    # The first prompt of a practice period or an evaluation episode: the contract, the
    # symbolic layer, and the agent's own notes. One per period and one per test task.
    OPENING = "opening"
    # One environment step. This is the number that makes this baseline expensive.
    DECISION = "decision"
    # The end-of-period request to write down what was learned. One per cycle, on the
    # practice conversation. The only channel from practice to evaluation.
    DIGEST = "digest"

    def __str__(self) -> str:
        return self.value


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
    """What the agent returns at one decision point: which of the *applicable* ground
    skills to execute, and the continuous parameters to execute it with.

    `skill_index` indexes the `skills` list of the observation the agent was just handed
    -- not the domain's lifted-skill list. Indexing the applicable set is what makes the
    agent unable to select a skill whose preconditions do not hold, so the one failure
    mode it cannot have is the one that would silently degrade into a no-op at the raw
    action layer and look like a bad choice instead of an illegal one.

    Frozen and validated here rather than trusted: this is the one place agent-authored
    output crosses into the harness."""

    model_config = ConfigDict(frozen=True)

    skill_index: int
    params: tuple[float, ...] = ()
