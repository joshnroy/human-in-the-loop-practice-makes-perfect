from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PracticeSessionEnd(BaseModel):
    """Why a session ended; actions include human resets but exclude STOP."""

    cycle_index: int
    reason: Literal["planner_stop", "interaction_complete", "session_action_cap"]
    actions_executed: int
    action_limit: int


class EvaluationBreakdown(BaseModel):
    """Which of one evaluation sweep's test tasks were solved -- the per-task
    detail behind a single (transitions, solved, total) entry in
    Metrics.evaluations.

    Recorded alongside that aggregate rather than replacing it: the aggregate is
    what Figure 4 plots and what every archived stats.json already contains, so
    it stays the primary record and this is strictly additive. The motivating
    question was "the arm ends at 95% -- *which* 5% is it failing?", which the
    aggregate can never answer, and which mattered because a domain can have
    task families that fail for structurally different reasons (in Tossing Room,
    a missed throw is merely expensive for TRASH but terminal for RECYCLING).

    Safe to correlate across sweeps by task_index: PracticeLoop draws the test
    set once, up front, so index i is the same Task at every checkpoint of a
    run."""

    num_online_transitions: int
    outcomes: tuple[TaskOutcome, ...]

    def num_solved(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.solved)


class TaskOutcome(BaseModel):
    """One test task's result in one evaluation sweep.

    goal is Goal.describe()'s rendering rather than the Goal itself: it is
    stable across processes, groups tasks into families by construction (every
    Tossing Room RECYCLING task renders identically), and keeps Metrics free of
    any dependency on core.problem -- Metrics stays domain-agnostic, and the one
    caller that has both (practice_loop.py) does the rendering."""

    task_index: int
    goal: str
    solved: bool
