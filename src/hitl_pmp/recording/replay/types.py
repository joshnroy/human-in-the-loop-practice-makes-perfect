from typing import Literal

from pydantic import BaseModel, ConfigDict

from hitl_pmp.recording.run_record.types import Phase, PlainState


class ReplayStatus(BaseModel):
    """What the overlay reports around one frame."""

    model_config = ConfigDict(frozen=True)

    environment: str
    phase: Phase
    speedup: int
    method: str | None = None
    checkpoint: int | None = None
    task_index: int | None = None
    cycle: int | None = None
    actor: Literal["robot", "human"] | None = None
    skill: str | None = None
    start_step: int | None = None
    end_step: int | None = None
    history: tuple[str, ...] = ()
    banner: str | None = None
    banner_tone: Literal["human", "success", "failure"] | None = None


class ReplaySummary(BaseModel):
    frames_written: int
    ticks_shown: int
    ticks_recorded: int
    # The state the renderer reports after the last frame it showed.
    final_state: PlainState | None
