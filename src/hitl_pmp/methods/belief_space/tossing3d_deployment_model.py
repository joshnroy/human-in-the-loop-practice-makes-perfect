"""Deployment-policy value calculation for Tossing3D."""


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
