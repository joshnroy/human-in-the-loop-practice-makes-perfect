"""Canonical ordering of exchangeable imagined Bernoulli evidence."""

import weakref

from .competence_inference import BayesianSkillBelief
from .types.belief_state import Tossing3DBeliefState
from .types.skill_belief import SkillBelief


class SweepCanonicalLikelihood:
    """Factor only a search segment with frozen latent values and cost evidence.

    Imagined Sweep transitions do not advance time, refit, resample, or observe
    costs. Their competence updates multiply c or (1-c) into one frozen root
    posterior. Counts are sufficient within that segment. Canonical success-
    then-failure ordering removes floating-point history dependence without
    rounding distributions or merging distinct root posteriors.
    """

    def __init__(self, *, state: Tossing3DBeliefState) -> None:
        self.posteriors: dict[tuple[str, int, int], BayesianSkillBelief] = {}
        self.members: dict[
            tuple[str, int], tuple[weakref.ReferenceType[BayesianSkillBelief], int, int]
        ] = {}
        for name, belief in state.skill_beliefs.items():
            if isinstance(belief, BayesianSkillBelief):
                self._record(name=name, successes=0, failures=0, belief=belief)

    def _record(
        self,
        *,
        name: str,
        successes: int,
        failures: int,
        belief: BayesianSkillBelief,
    ) -> BayesianSkillBelief:
        self.posteriors[name, successes, failures] = belief
        self.members[name, id(belief)] = (weakref.ref(belief), successes, failures)
        return belief

    def condition(
        self,
        *,
        name: str,
        belief: SkillBelief,
        success: bool,
    ) -> BayesianSkillBelief | None:
        member = self.members.get((name, id(belief)))
        if member is None or member[0]() is not belief:
            # A different posterior, refit, or cost observation is not part of
            # this exchangeable segment, even if its counters happen to match.
            return None
        return self._posterior(
            name=name,
            successes=member[1] + int(success),
            failures=member[2] + int(not success),
        )

    def _posterior(self, *, name: str, successes: int, failures: int) -> BayesianSkillBelief:
        cached = self.posteriors.get((name, successes, failures))
        if cached is not None:
            return cached
        if failures:
            previous = self._posterior(name=name, successes=successes, failures=failures - 1)
            belief = previous.condition_outcome(success=False)
        else:
            previous = self._posterior(name=name, successes=successes - 1, failures=0)
            belief = previous.condition_outcome(success=True)
        return self._record(name=name, successes=successes, failures=failures, belief=belief)
