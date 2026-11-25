"""Which practice outcomes condition a skill's competence posterior."""

from enum import Enum

from hitl_pmp.core.method.types import SamplerConsultation


class CompetenceEvidence(Enum):
    """Filter on a sampler-driven attempt's `SamplerConsultation`.

    Only competence conditioning is filtered. Every attempt still enters the
    sampler's training data and still updates the cost filter; it advances the
    training clock too, unless `Tossing3DPomdpMethod.pomdp_clock_on_conditioned_only`
    keeps a filtered-out attempt off the clock as well. A skill with no sampler (`NO_SAMPLER`:
    pick, open gripper, and the resets, which report no consultation at all) is a
    fixed controller whose every attempt is evidence, so every setting admits it.
    """

    ALL = "all"
    NON_EPSILON = "non_epsilon"
    INFORMED = "informed"
    # Informed picks, and uninformative ones once the sampler's fit has seen a success:
    # only a uniform draw made with no success yet to learn from is kept out, as
    # `EesMethod.competence_from_informed_only` keeps it out (2026-11-24). INFORMED
    # also keeps out a never-failed skill's picks, whose one-class fit ties at 1.0.
    SAMPLER_SUCCEEDED = "sampler_succeeded"

    def admits(
        self, *, consultation: SamplerConsultation, sampler_has_success: bool = False
    ) -> bool:
        if consultation is SamplerConsultation.NO_SAMPLER or self is CompetenceEvidence.ALL:
            return True
        if self is CompetenceEvidence.NON_EPSILON:
            return consultation is not SamplerConsultation.EPSILON_RANDOM
        if self is CompetenceEvidence.SAMPLER_SUCCEEDED and sampler_has_success:
            return consultation is not SamplerConsultation.EPSILON_RANDOM
        return consultation is SamplerConsultation.INFORMED
