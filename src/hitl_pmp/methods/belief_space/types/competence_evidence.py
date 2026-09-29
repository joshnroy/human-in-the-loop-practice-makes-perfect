"""Which practice outcomes condition a skill's competence posterior."""

from enum import Enum

from hitl_pmp.core.method.types import SamplerConsultation


class CompetenceEvidence(Enum):
    """Filter on a sampler-driven attempt's `SamplerConsultation`.

    Only competence conditioning is filtered. Every attempt still enters the
    sampler's training data, and so still advances the training clock, and every
    attempt still updates the cost filter. A skill with no sampler (`NO_SAMPLER`:
    pick, open gripper, and the resets, which report no consultation at all) is a
    fixed controller whose every attempt is evidence, so every setting admits it.
    """

    ALL = "all"
    NON_EPSILON = "non_epsilon"
    INFORMED = "informed"

    def admits(self, *, consultation: SamplerConsultation) -> bool:
        if consultation is SamplerConsultation.NO_SAMPLER or self is CompetenceEvidence.ALL:
            return True
        if self is CompetenceEvidence.NON_EPSILON:
            return consultation is not SamplerConsultation.EPSILON_RANDOM
        return consultation is SamplerConsultation.INFORMED
