"""Deployment-policy value calculation for Tossing3D."""

import numpy as np
from numpy.typing import NDArray


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
