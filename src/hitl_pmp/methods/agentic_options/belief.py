"""Existing Tossing3D inference behind a replaceable learner-belief interface."""

from typing import Protocol, runtime_checkable

import numpy as np

from hitl_pmp.methods.belief_space.competence_inference import BayesianSkillBelief
from hitl_pmp.methods.belief_space.tossing3d_constants import TOSS_SKILL
from hitl_pmp.methods.belief_space.tossing3d_model import Tossing3DPracticeModel
from hitl_pmp.methods.belief_space.tossing3d_observation_model import refit_belief_state
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.particle_filter_belief import ParticleFilterBelief
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta


@runtime_checkable
class LearnerBelief(Protocol):
    """The search sees predicted learning; actual learner data remains outside it."""

    def success_probability(self, *, belief: Tossing3DBeliefState, skill: str) -> float: ...

    def estimated_cost(
        self, *, belief: Tossing3DBeliefState, skill: str, configured_cost: float
    ) -> float: ...

    def observe(
        self,
        *,
        belief: Tossing3DBeliefState,
        skill: str,
        success: bool,
        observed_cost: float | None = None,
    ) -> Tossing3DBeliefState: ...

    def advance_session(self, *, belief: Tossing3DBeliefState) -> Tossing3DBeliefState: ...

    def stop_value(self, *, belief: Tossing3DBeliefState, cost: float) -> float: ...

    def sample_thetas(
        self, *, belief: Tossing3DBeliefState, count: int
    ) -> list[Tossing3DTheta]: ...

    def evaluate(self, *, theta: Tossing3DTheta) -> float: ...

    def sample_values(self, *, belief: Tossing3DBeliefState, count: int) -> np.ndarray: ...

    def objective(self, *, value: float, cost: float) -> float: ...


class TossingLearnerBelief:
    """Preserve the posterior, training clock, refit forecast and deployment value.

    Code policies have no epsilon sampler: every resolved execution is evidence.
    The legacy sampler metadata is retained as bookkeeping, not used to invent a
    random/greedy mixture for a code controller that never selected such a mode.
    """

    def __init__(self, *, linear_cost_lambda: float | None = None, seed: int = 0) -> None:
        self.value_model = Tossing3DPracticeModel(linear_cost_lambda=linear_cost_lambda, seed=seed)

    def success_probability(self, *, belief: Tossing3DBeliefState, skill: str) -> float:
        return belief.skill_beliefs[skill].mean_competence()

    def estimated_cost(
        self, *, belief: Tossing3DBeliefState, skill: str, configured_cost: float
    ) -> float:
        posterior = belief.skill_beliefs[skill]
        if isinstance(posterior, (ParticleFilterBelief, BayesianSkillBelief)):
            return posterior.mean_cost()
        return configured_cost

    def observe(
        self,
        *,
        belief: Tossing3DBeliefState,
        skill: str,
        success: bool,
        observed_cost: float | None = None,
    ) -> Tossing3DBeliefState:
        posteriors = dict(belief.skill_beliefs)
        posterior = posteriors[skill]
        posteriors[skill] = (
            posterior.condition_execution(success=success, observed_cost=observed_cost)
            if observed_cost is not None
            and isinstance(posterior, (ParticleFilterBelief, BayesianSkillBelief))
            else posterior.condition_outcome(success=success)
        )
        pending = dict(belief.pending_examples)
        pending[skill] = pending.get(skill, 0) + 1
        training = dict(belief.sampler_training)
        if skill == TOSS_SKILL and skill in training:
            training[skill] = training[skill].observe(success=success)
        return belief.model_copy(
            update={
                "skill_beliefs": posteriors,
                "pending_examples": pending,
                "sampler_training": training,
            }
        )

    def advance_session(self, *, belief: Tossing3DBeliefState) -> Tossing3DBeliefState:
        return refit_belief_state(state=belief, advance_cycle=True)

    def stop_value(self, *, belief: Tossing3DBeliefState, cost: float) -> float:
        return self.value_model.J(belief_state=belief, summed_cost=cost, num_samples=1)

    def sample_thetas(self, *, belief: Tossing3DBeliefState, count: int) -> list[Tossing3DTheta]:
        return self.value_model.sample_thetas_from_belief(belief_state=belief, num_samples=count)

    def evaluate(self, *, theta: Tossing3DTheta) -> float:
        return self.value_model.evaluate_policy(sampled_theta=theta)

    def sample_values(self, *, belief: Tossing3DBeliefState, count: int) -> np.ndarray:
        return self.value_model.sample_policy_values_from_belief(
            belief_state=belief, num_samples=count
        )

    def objective(self, *, value: float, cost: float) -> float:
        return self.value_model.G(policy_value=value, summed_cost=cost)
