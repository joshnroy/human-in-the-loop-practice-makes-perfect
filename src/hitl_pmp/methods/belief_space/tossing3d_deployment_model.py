"""Deployment-policy value calculation for Tossing3D."""

import math

import numpy as np
from numpy.typing import NDArray

from .types.skill_belief import SkillBelief


class DeploymentPolicyExpectation:
    """Integrate the canonical deployment policy over independent skill beliefs."""

    @staticmethod
    def evaluate(
        *, pick: SkillBelief, toss: SkillBelief, opened: SkillBelief, horizon: int
    ) -> float:
        assert horizon >= 0
        if horizon < 2:
            return 0.0
        # After n failed picks and m failed opens, Pick -> Toss finishes in
        # 2 + 2*n + m actions. Distribute the m open failures over the n
        # recovery episodes. Integrating each positive term avoids cancellation
        # and keeps repeated trials conditioned on the same latent competence.
        ready_probability = pick.mean_competence()
        for failed_picks in range(1, (horizon - 2) // 2 + 1):
            pick_probability = pick.competence_outcome_probability(
                successes=1, failures=failed_picks
            )
            for failed_opens in range(horizon - 2 - 2 * failed_picks + 1):
                ready_probability += (
                    math.comb(failed_picks + failed_opens - 1, failed_opens)
                    * pick_probability
                    * opened.competence_outcome_probability(
                        successes=failed_picks, failures=failed_opens
                    )
                )
        return toss.mean_competence() * ready_probability


def evaluate_deployment_policies(
    *,
    toss_competences: NDArray[np.float64],
    pick_competences: NDArray[np.float64],
    open_competences: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.float64]:
    """Evaluate a batch of sampled deployment models together."""
    ready_values = np.zeros_like(toss_competences)
    holding_values = np.zeros_like(toss_competences)
    closed_gripper_values = np.zeros_like(toss_competences)
    for _ in range(horizon):
        previous_ready = ready_values
        previous_holding = holding_values
        previous_closed_gripper = closed_gripper_values
        holding_values = toss_competences
        ready_values = (
            pick_competences * previous_holding + (1.0 - pick_competences) * previous_closed_gripper
        )
        closed_gripper_values = (
            open_competences * previous_ready + (1.0 - open_competences) * previous_closed_gripper
        )
    return ready_values


def evaluate_deployment_policy(
    *, toss_competence: float, pick_competence: float, open_competence: float, horizon: int
) -> float:
    """Solve the canonical deployment MDP; human reset is unavailable at test."""
    ready_value = holding_value = closed_gripper_value = 0.0
    for _ in range(horizon):
        previous_ready = ready_value
        previous_holding = holding_value
        previous_closed_gripper = closed_gripper_value
        holding_value = toss_competence
        ready_value = (
            pick_competence * previous_holding + (1.0 - pick_competence) * previous_closed_gripper
        )
        closed_gripper_value = (
            open_competence * previous_ready + (1.0 - open_competence) * previous_closed_gripper
        )
    return ready_value
