"""Search data: cluster identity and observed outcome, never simulator state."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClusterState(BaseModel):
    """Outcome is retained even when success and failure end in the same cluster."""

    model_config = ConfigDict(frozen=True)

    cluster_id: str
    outcome: Literal["success", "failure"] | None = None


class OptionAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    option_id: str


class SearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int = Field(default=3, ge=0)
    observation_probability_weight: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    linear_cost_lambda: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    num_particles: int = Field(default=256, ge=1)
    seed: int = 0
