"""The always-ask wrapper's forced full reset, as the belief-space model sees it."""

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.problem.tasks.types import GroundAtom


class ForcedStartReset(BaseModel):
    """After every skill with continuous parameters the human resets the scene to a
    freshly sampled start, at `cost`. A single symbolic destination: every sampled
    start abstracts to the same atoms on the domains that use it, so the destination
    distribution is one point at the model's level."""

    model_config = ConfigDict(frozen=True)

    start_atoms: frozenset[GroundAtom]
    cost: float = Field(ge=0.0, allow_inf_nan=False)
