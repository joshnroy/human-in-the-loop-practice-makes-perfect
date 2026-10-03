"""Skill-chain transitions adapted to the unchanged belief-space search."""

from collections import Counter
from typing import Literal

import numpy as np

from hitl_pmp.agentic_runtime.types import RuntimeManifest
from hitl_pmp.methods.belief_space.types.belief_state import Tossing3DBeliefState
from hitl_pmp.methods.belief_space.types.theta import Tossing3DTheta

from .belief import LearnerBelief
from .types import ClusterState, OptionAction


class SkillChainModel:
    """Physical edges come from the chain; success rates come from the learner.

    A generated edge distribution is an explicit initial modeling assumption.
    Once observations exist for a conditional edge, empirical counts replace it.
    Unmodeled destinations are not folded into a failure self-loop.
    """

    def __init__(self, *, manifest: RuntimeManifest, learner_belief: LearnerBelief) -> None:
        self.manifest = manifest
        self.learner_belief = learner_belief
        self.options = {option.option_id: option for option in manifest.options}
        self.counts: dict[tuple[str, str, str], Counter[str]] = {}
        self.revision = 0
        self.blocked_actions: set[tuple[str, str]] = set()

    def record_destination(
        self, *, source: str, option_id: str, outcome: str, destination: str
    ) -> None:
        clusters = {cluster.cluster_id for cluster in self.manifest.clusters}
        if destination not in clusters or source not in clusters or option_id not in self.options:
            raise ValueError("Observed transition references an unknown cluster or option")
        if outcome not in ("success", "failure"):
            raise ValueError("Unresolved outcomes cannot update the skill-chain")
        self.counts.setdefault((source, option_id, outcome), Counter())[destination] += 1
        self.revision += 1

    def invalidate_code_revision(self) -> None:
        # Code may change conditional failure effects. Old physical evidence must
        # not quietly override the frozen prior graph for a newly revised policy.
        self.counts.clear()
        self.blocked_actions.clear()
        self.revision += 1

    def get_valid_actions(self, *, environment_state: ClusterState) -> list[OptionAction]:
        available = {
            edge.option_id
            for edge in self.manifest.edges
            if edge.source == environment_state.cluster_id
        }
        return [
            OptionAction(option_id=option.option_id)
            for option in self.manifest.options
            if option.option_id in available
            and (environment_state.cluster_id, option.option_id) not in self.blocked_actions
        ]

    def transition_outcomes(
        self,
        *,
        environment_state: ClusterState,
        practice_action: OptionAction,
        belief_state: Tossing3DBeliefState,
    ) -> list[tuple[ClusterState, float, float]]:
        option = self.options[practice_action.option_id]
        success_probability = (
            1.0
            if option.human_destination is not None
            else self.learner_belief.success_probability(
                belief=belief_state, skill=option.belief_skill
            )
        )
        cost = self.learner_belief.estimated_cost(
            belief=belief_state, skill=option.belief_skill, configured_cost=option.cost
        )
        result = []
        outcomes: tuple[tuple[Literal["success", "failure"], float], ...] = (
            ("success", success_probability),
            ("failure", 1.0 - success_probability),
        )
        for outcome, probability in outcomes:
            if probability == 0:
                continue
            key = environment_state.cluster_id, option.option_id, outcome
            counts = self.counts.get(key)
            destinations = (
                [(destination, count / counts.total()) for destination, count in counts.items()]
                if counts
                else [
                    (edge.destination, edge.probability)
                    for edge in self.manifest.edges
                    if (edge.source, edge.option_id, edge.outcome) == key
                ]
            )
            if not destinations:
                raise ValueError(f"Skill-chain lacks conditional destinations for {key}")
            for destination, conditional_probability in destinations:
                result.append((
                    ClusterState(cluster_id=destination, outcome=outcome),
                    cost,
                    probability * conditional_probability,
                ))
        return result

    def compute_next_belief_state(
        self,
        *,
        belief_state: Tossing3DBeliefState,
        environment_state: ClusterState,
        potential_next_environment_state: ClusterState,
        practice_action: OptionAction,
    ) -> Tossing3DBeliefState:
        del environment_state
        outcome = potential_next_environment_state.outcome
        if outcome is None:
            raise ValueError("A successor must retain its outcome for belief conditioning")
        return self.learner_belief.observe(
            belief=belief_state,
            skill=self.options[practice_action.option_id].belief_skill,
            success=outcome == "success",
        )

    def search_cache_key(
        self,
        *,
        environment_state: ClusterState,
        summed_cost: float,
        belief_state: Tossing3DBeliefState,
        horizon: int | None,
    ) -> object:
        return (
            self.manifest.revision,
            self.revision,
            environment_state.cluster_id,
            summed_cost,
            horizon,
            belief_state.model_dump_json(),
        )

    def J(
        self, *, belief_state: Tossing3DBeliefState, summed_cost: float, num_samples: int
    ) -> float:
        del num_samples
        return self.learner_belief.stop_value(belief=belief_state, cost=summed_cost)

    def G(self, *, policy_value: float, summed_cost: float) -> float:
        return self.learner_belief.objective(value=policy_value, cost=summed_cost)

    def sample_theta_from_belief(self, *, belief_state: Tossing3DBeliefState) -> Tossing3DTheta:
        return self.sample_thetas_from_belief(belief_state=belief_state, num_samples=1)[0]

    def sample_thetas_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> list[Tossing3DTheta]:
        return self.learner_belief.sample_thetas(belief=belief_state, count=num_samples)

    def sample_policy_values_from_belief(
        self, *, belief_state: Tossing3DBeliefState, num_samples: int
    ) -> np.ndarray:
        return self.learner_belief.sample_values(belief=belief_state, count=num_samples)

    def evaluate_policy(self, *, sampled_theta: Tossing3DTheta) -> float:
        return self.learner_belief.evaluate(theta=sampled_theta)

    def sample_next_states(
        self,
        *,
        environment_state: ClusterState,
        practice_action: OptionAction,
        belief_state: Tossing3DBeliefState,
    ) -> list[tuple[ClusterState, float]]:
        return [
            (state, cost)
            for state, cost, _ in self.transition_outcomes(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
        ]

    def transition_probability(
        self,
        *,
        potential_next_environment_state: ClusterState,
        sampled_cost: float,
        environment_state: ClusterState,
        practice_action: OptionAction,
        belief_state: Tossing3DBeliefState,
    ) -> float:
        return sum(
            probability
            for state, cost, probability in self.transition_outcomes(
                environment_state=environment_state,
                practice_action=practice_action,
                belief_state=belief_state,
            )
            if state == potential_next_environment_state and cost == sampled_cost
        )
