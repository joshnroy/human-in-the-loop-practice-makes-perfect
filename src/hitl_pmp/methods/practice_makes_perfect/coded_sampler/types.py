"""Data for the coding-agent sampler: what a refit produced and what a call returned."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CodedSamplerMode(str, Enum):
    """What the generated module per skill exposes.

    `SCORER` replaces the MLP's `predict_proba`: `score(sampler_inputs) -> list[float]`,
    and the existing argmax-over-candidates wrapper decides. `SAMPLER` replaces the
    whole draw: `sample(state, grounding, rng) -> list[float]`."""

    SCORER = "scorer"
    SAMPLER = "sampler"


class TrajectoryEvent(BaseModel):
    """One line of the run's practice history, in the order things happened."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    kind: str
    # The practice cycle the event belongs to, 0-based: the refit that ends cycle `c`
    # is refit `c`.
    cycle: int = Field(ge=0)
    data: dict[str, Any]


class CodeVersion(BaseModel):
    """One frozen snapshot of the generated modules, taken after a refit."""

    model_config = ConfigDict(frozen=True)

    # 0 is "no code yet"; refit k leaves version k + 1.
    index: int = Field(ge=0)
    cycle: int | None
    snapshot_dir: Path | None
    git_commit: str | None
    # skill name -> module path inside `snapshot_dir`, for modules that exist.
    modules: dict[str, str] = Field(default_factory=dict)


class CompactionEvent(BaseModel):
    """One context compaction the CLI reported in its stream."""

    model_config = ConfigDict(frozen=True)

    trigger: str | None
    pre_tokens: int | None
    post_tokens: int | None


class CoderResult(BaseModel):
    """What one coding-agent invocation reported."""

    success: bool
    error: str | None = None
    session_id: str | None = None
    resumed: bool = False
    wall_seconds: float = 0.0
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    num_turns: int = 0
    # Context size at the last assistant turn: input + cache read + cache creation.
    final_context_tokens: int | None = None
    compactions: list[CompactionEvent] = Field(default_factory=list)
    rate_limit_retries: int = 0
    # The CLI stopped this invocation at its dollar budget (`error_max_budget_usd`),
    # so whatever it left in the workspace may be half-written.
    budget_exhausted: bool = False


class CodeCallOutcome(BaseModel):
    """One sandboxed call of a generated function."""

    ok: bool
    value: Any = None
    error: str | None = None
    latency_seconds: float = 0.0
