"""Deployment-policy value calculation for Tossing3D."""

import math
from collections.abc import Mapping
from functools import cache
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .deployment_value_iteration import DeploymentOption, DeploymentValueIteration
from .tossing3d_constants import OPEN_GRIPPER_SKILL, PICK_SKILL, TOSS_SKILL
from .types.skill_belief import SkillBelief


class DeploymentPolicyExpectation:
    """Integrate `Tossing3DDeployment`'s value over independent skill beliefs, in
    closed form: an exact expectation, where a sample mean's noise would be much
    larger than the cost differences the planner compares stopping values by."""

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


class Tossing3DDeployment:
    """The held-out solve probability for Tossing3D, given each skill's competence:
    pick, toss, and open the gripper to try again after a failed pick. A
    static-method container.

    Each state has one action, so nothing is chosen, and unlike the shelf and the rack
    a failure moves the state: a failed pick leaves the gripper closed on nothing, a
    failed open leaves it closed, and a toss ends the episode either way. No human
    reset exists at test time.
    """

    START = "ready"
    TERMINAL_VALUES: ClassVar[dict[str, float]] = {"success": 1.0, "failure": 0.0}

    @staticmethod
    @cache
    def transitions() -> dict[str, tuple[DeploymentOption, ...]]:
        return {
            "ready": ((PICK_SKILL, "holding", "closed_gripper"),),
            "holding": ((TOSS_SKILL, "success", "failure"),),
            "closed_gripper": ((OPEN_GRIPPER_SKILL, "ready", "closed_gripper"),),
        }

    @staticmethod
    def values(
        *, competences: Mapping[str, NDArray[np.float64]], horizon: int
    ) -> NDArray[np.float64]:
        """Batch value at the start state (cube on the ground, gripper open)."""
        return DeploymentValueIteration.values(
            transitions=Tossing3DDeployment.transitions(),
            terminal_values=Tossing3DDeployment.TERMINAL_VALUES,
            competences=competences,
            horizon=horizon,
            start=Tossing3DDeployment.START,
        )
