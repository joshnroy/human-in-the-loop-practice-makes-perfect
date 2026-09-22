"""Compact root-level diagnostics for one expectimax search."""

from typing import Any

from pydantic import BaseModel, Field

from hitl_pmp.core.log_timing import LogTiming


class SearchTrace(BaseModel):
    """Root values and aggregate counts needed for diagnostics and video replay.

    Expectimax retains compact root summaries. Determinized search can additionally
    emit opt-in flat node/edge events without serializing nested belief payloads.
    """

    events: list[dict[str, Any]] = Field(default_factory=list)

    def record(self, *, event: str, **fields: Any) -> None:
        self.events.append({"event": event, **fields, **LogTiming.fields()})

    def close(self) -> None:
        """Retain the old lifecycle interface; compact summaries need no writer."""
