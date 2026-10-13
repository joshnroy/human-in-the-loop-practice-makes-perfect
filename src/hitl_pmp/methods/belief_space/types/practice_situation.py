"""What the method tells an LLM practice planner beyond the belief state."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import PracticeSchedule


class PracticeHistory(BaseModel):
    """A compact, already-scoped slice of the session log.

    `scope` names how the method cut it: the last N decisions for a stateless
    planner or the first call of a resumed session, or only what happened since the
    previous call for a resumed session. `skill_totals` is cumulative regardless."""

    model_config = ConfigDict(frozen=True)

    scope: str
    events: tuple[dict[str, Any], ...]
    skill_totals: dict[str, dict[str, int]] = Field(default_factory=dict)


class PracticeSituation(BaseModel):
    """Session bookkeeping the belief state does not carry, handed to one decision."""

    model_config = ConfigDict(frozen=True)

    session_index: int = Field(ge=0)
    decision_index: int = Field(ge=1)
    session_key: str = Field(min_length=1)
    remaining_session_actions: int | None = Field(default=None, ge=0)
    schedule: PracticeSchedule | None = None
    configured_costs: dict[str, Any] = Field(default_factory=dict)
    # Candidate id -> the human-facing name of a human reset (`reset_cube_far` /
    # `reset_cube_and_bin_near`), for the groundings a person would execute.
    human_names: dict[str, str] = Field(default_factory=dict)
    history: PracticeHistory
