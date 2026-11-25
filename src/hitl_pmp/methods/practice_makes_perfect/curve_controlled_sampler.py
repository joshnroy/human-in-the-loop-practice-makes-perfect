"""`--curve-controlled-sampler` (Josh, 2026-11-25): an explicit oracle control, for
testing the planners against a skill whose learning curve is known in advance.

For skill k after n_k of its own practice tries, a setting is drawn uniformly from the
skill's measured success region with probability alpha(n_k) = min(1, start + rate n_k),
and uniformly from its measured failure region otherwise -- the chained toy's
`TwoRoutesEnvironment.success_probability` (start 0.1, rate 0.15), realised through
the measured scene instead of a coin. Test draws use the same alpha at the current n_k,
so deployed success follows the same curve. When on, it replaces the learned sampler
and its epsilon exploration for every skill it has regions for; never a default."""

import numpy as np
from pydantic import BaseModel, Field

from hitl_pmp.core.method.types import KnownOutcomeRegions


class CurveControlledSampler(BaseModel):
    regions: dict[str, KnownOutcomeRegions]
    start: float = Field(default=0.1, ge=0.0, le=1.0)
    rate: float = Field(default=0.15, ge=0.0)
    # Skill name -> practice tries so far. A field, not private state, so the deployment
    # snapshot (which carries the method's fields) evaluates at the current n.
    tries: dict[str, int] = Field(default_factory=dict)

    def controls(self, *, name: str) -> bool:
        return name in self.regions

    def alpha(self, *, n: int) -> float:
        return min(1.0, self.start + self.rate * n)

    def success_probability(self, *, name: str) -> float:
        return self.alpha(n=self.tries.get(name, 0))

    def record_try(self, *, name: str) -> None:
        self.tries[name] = self.tries.get(name, 0) + 1

    def draw(self, *, name: str, rng: np.random.Generator) -> np.ndarray:
        """One setting: S with probability alpha(n), else F; uniform within the box."""
        region = self.regions[name]
        box = (
            region.success
            if rng.uniform() < self.success_probability(name=name)
            else region.failure
        )
        return np.array([rng.uniform(low, high) for low, high in box])
