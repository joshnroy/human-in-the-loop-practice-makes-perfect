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

    @staticmethod
    def evaluate_with_cost(
        *,
        pick: SkillBelief,
        toss: SkillBelief,
        opened: SkillBelief,
        horizon: int,
        pick_cost: float,
        toss_cost: float,
        open_cost: float,
    ) -> tuple[float, float]:
        """`evaluate` and the same policy's expected cost per task, exactly.

        Walks every pick/open outcome path to the horizon. A path's probability
        integrates each skill's latent competence once over all of that skill's
        outcomes on the path (`competence_outcome_probability`), as `evaluate` does;
        its cost is the sum of each executed skill's fixed cost."""
        assert horizon >= 0
        costs = {"pick": pick_cost, "toss": toss_cost, "open": open_cost}
        expected_cost = 0.0
        frontier: list[tuple[str, int, int, int, int, float]] = [("ready", 0, 0, 0, 0, 0.0)]
        for steps_left in range(horizon, 0, -1):
            advanced: list[tuple[str, int, int, int, int, float]] = []
            for state, picks_ok, picks_failed, opens_ok, opens_failed, cost in frontier:
                if state == "holding":
                    probability = pick.competence_outcome_probability(
                        successes=picks_ok, failures=picks_failed
                    ) * opened.competence_outcome_probability(
                        successes=opens_ok, failures=opens_failed
                    )
                    expected_cost += probability * (cost + costs["toss"])
                elif state == "ready":
                    charged = cost + costs["pick"]
                    advanced.append((
                        "holding",
                        picks_ok + 1,
                        picks_failed,
                        opens_ok,
                        opens_failed,
                        charged,
                    ))
                    advanced.append((
                        "closed",
                        picks_ok,
                        picks_failed + 1,
                        opens_ok,
                        opens_failed,
                        charged,
                    ))
                else:
                    charged = cost + costs["open"]
                    advanced.append((
                        "ready",
                        picks_ok,
                        picks_failed,
                        opens_ok + 1,
                        opens_failed,
                        charged,
                    ))
                    advanced.append((
                        "closed",
                        picks_ok,
                        picks_failed,
                        opens_ok,
                        opens_failed + 1,
                        charged,
                    ))
            frontier = advanced
            del steps_left
        for _, picks_ok, picks_failed, opens_ok, opens_failed, cost in frontier:
            expected_cost += (
                pick.competence_outcome_probability(successes=picks_ok, failures=picks_failed)
                * opened.competence_outcome_probability(successes=opens_ok, failures=opens_failed)
                * cost
            )
        value = DeploymentPolicyExpectation.evaluate(
            pick=pick, toss=toss, opened=opened, horizon=horizon
        )
        return value, expected_cost


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


def evaluate_deployment_policy_costs(
    *,
    pick_competences: NDArray[np.float64],
    open_competences: NDArray[np.float64],
    pick_costs: NDArray[np.float64],
    toss_costs: NDArray[np.float64],
    open_costs: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.float64]:
    """Expected cost of one task under `evaluate_deployment_policies`' own policy, batched.

    The same recursion with each executed skill charged its cost: a toss ends the task
    whether it scores or not, a failed pick leaves the gripper closed for OpenGripper,
    and nothing more is charged once the horizon runs out."""
    ready_costs = np.zeros_like(pick_competences)
    holding_costs = np.zeros_like(pick_competences)
    closed_gripper_costs = np.zeros_like(pick_competences)
    for _ in range(horizon):
        previous_ready = ready_costs
        previous_holding = holding_costs
        previous_closed_gripper = closed_gripper_costs
        holding_costs = toss_costs
        ready_costs = pick_costs + (
            pick_competences * previous_holding + (1.0 - pick_competences) * previous_closed_gripper
        )
        closed_gripper_costs = open_costs + (
            open_competences * previous_ready + (1.0 - open_competences) * previous_closed_gripper
        )
    return ready_costs
