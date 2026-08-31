"""Recorded expectimax search events."""

from typing import Any

from pydantic import BaseModel, Field

from hitl_pmp.core.log_timing import LogTiming


class SearchTrace(BaseModel):
    """Actual search events, recorded without additional model calls or RNG draws."""

    events: list[dict[str, Any]] = Field(default_factory=list)

    def record(self, *, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields, **LogTiming.fields()})
