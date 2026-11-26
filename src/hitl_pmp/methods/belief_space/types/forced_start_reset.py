"""The always-ask wrapper's forced full reset, as the belief-space model sees it."""

from pydantic import BaseModel, ConfigDict, Field

from hitl_pmp.core.method.types import Skill
from hitl_pmp.core.problem.tasks.types import GroundAtom


class ForcedStartReset(BaseModel):
    """After every skill with continuous parameters -- or, when `trigger_skills` names
    some, after those alone, as the wrapper does -- the human resets the scene to a
    freshly sampled start, at `cost`. A single symbolic destination: every sampled
    start abstracts to the same atoms on the domains that use it, so the destination
    distribution is one point at the model's level."""

    model_config = ConfigDict(frozen=True)

    start_atoms: frozenset[GroundAtom]
    cost: float = Field(ge=0.0, allow_inf_nan=False)
    trigger_skills: frozenset[str] | None = None

    def follows(self, *, skill: Skill) -> bool:
        """Whether the reset follows an execution of `skill`, mirroring
        `AlwaysAskReset.observe_dispatch`."""
        return skill.param_dim > 0 and (
            self.trigger_skills is None or skill.name in self.trigger_skills
        )
