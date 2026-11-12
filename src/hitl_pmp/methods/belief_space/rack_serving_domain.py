"""Rack-specific belief routing and an abstract, human-free deployment value.

This mirrors the shelf domain's analytical convention: successful actions apply
their symbolic effects and failures leave the deployment state unchanged. It is
not a model of settling, adhesion, spills, or automatic clock advancement during
robot motion. Practice uses the provider's full operators and learned failure
effects; actual evaluation still uses the live rack backend's verdict.
"""

from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hitl_pmp.core.method.skill_provider import SkillProvider

from .practice_domain import PracticeDomain, SampleSkills
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief

RACK_NUM_FOOD = 5
RACK_ROBOT_SKILLS = (
    "PickScraper",
    "GatherAndSweep",
    "ReturnScraper",
    "PickFood",
    "PlaceFoodOnTray",
    "ReturnFoodToRack",
    # Last, so the six older skills keep their prior seeds and Monte Carlo draws.
    "Sweep",
)
# Two ways to empty the rack with the scraper held, both all-or-nothing in this model.
RACK_SWEEP_SKILLS = ("GatherAndSweep", "Sweep")
RACK_HUMAN_SKILLS = ("human_batch_return", "human_tool_recovery")
RACK_EXPECTATION_SAMPLES = 4096
RACK_EXPECTATION_SEED = 0
RackDeploymentState = tuple[int, str, bool]
# Mean control steps per evaluation execution, (on success, on failure), from every
# evaluation task of run tag rack-prior2x2-b406ad8-20261106 (all four arms, 735 tasks),
# read from each task's recording.jsonl `action` records: unlike actions.jsonl, those
# include the final, goal-reaching action. Success is the record's own `success` flag.
# Executions (success/failure): PickScraper 776/5, GatherAndSweep 246/368,
# ReturnScraper 447/9, PickFood 1146/32, PlaceFoodOnTray 681/390, ReturnFoodToRack 95/6,
# Sweep 130/65.
RACK_SKILL_STEPS: dict[str, tuple[float, float]] = {
    "PickScraper": (449.0, 275.0),
    "GatherAndSweep": (3916.0, 3436.0),
    "ReturnScraper": (874.0, 113.0),
    "PickFood": (645.0, 434.0),
    "PlaceFoodOnTray": (296.0, 166.0),
    "ReturnFoodToRack": (499.0, 212.0),
    "Sweep": (969.0, 889.0),
}


class RackServingDeployment:
    """Value from five foods on the rack; food is ready from the start.

    State is (rack count, hand contents, scraper home). A held food is
    neither on rack nor on tray; all other food is on the tray. Return-to-rack
    and picking from the tray are legal, not forced recovery choices. The goal
    is every food on the tray, not a home scraper, matching the
    rack task rather than adding a new physical completion requirement.
    """

    @staticmethod
    @cache
    def transitions() -> dict[RackDeploymentState, tuple[tuple[str, RackDeploymentState], ...]]:
        moves = {}
        for on_rack in range(RACK_NUM_FOOD + 1):
            for hand in ("empty", "scraper", "food"):
                if hand == "food" and on_rack == RACK_NUM_FOOD:
                    continue
                for home in (True, False):
                    if hand == "scraper" and home:
                        continue
                    state = (on_rack, hand, home)
                    options: list[tuple[str, RackDeploymentState]] = []
                    if hand == "empty" and home:
                        options.append(("PickScraper", (on_rack, "scraper", False)))
                    if hand == "scraper":
                        options.append(("ReturnScraper", (on_rack, "empty", True)))
                        if on_rack == RACK_NUM_FOOD:
                            options.extend(
                                (name, (0, "scraper", False)) for name in RACK_SWEEP_SKILLS
                            )
                    if hand == "empty":
                        if on_rack:
                            options.append(("PickFood", (on_rack - 1, "food", home)))
                        if on_rack < RACK_NUM_FOOD:
                            options.append(("PickFood", (on_rack, "food", home)))
                    if hand == "food":
                        options.append(("PlaceFoodOnTray", (on_rack, "empty", home)))
                        options.append(("ReturnFoodToRack", (on_rack + 1, "empty", home)))
                    moves[state] = tuple(options)
        return moves

    @staticmethod
    def values(*, competences: dict[str, NDArray[np.float64]], horizon: int) -> NDArray[np.float64]:
        """P(solve) from the start state: `solve` with no cost per step."""
        return RackServingDeployment.solve(
            competences=competences, horizon=horizon, step_weight=0.0
        )[0]

    @staticmethod
    def solve(
        *,
        competences: dict[str, NDArray[np.float64]],
        horizon: int,
        step_weight: float,
        one_bulk_attempt: bool = False,
        bulk_rewards: Mapping[str, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """One finite-horizon DP maximising P(solve) - step_weight * E[control steps].

        Returns that value and the expected control steps of the policy attaining it,
        both batched, at the start state. Each attempt costs RACK_SKILL_STEPS' success
        or failure steps. Stopping short (value 0, no further steps) stays available,
        as it always was; at zero weight this is exactly the P(solve)-only DP.

        With `one_bulk_attempt` a failed Sweep or GatherAndSweep ends the task unsolved
        instead of leaving the state to be retried, as `bulk_transfer_only` evaluation
        does. Every other skill is still retried.

        `bulk_rewards` (two-route rewards; needs `one_bulk_attempt`) names what a
        successful Sweep or GatherAndSweep is worth: its success ends the task with
        that reward in place of 1, so the value is a reward, not a probability.
        """
        if bulk_rewards and not one_bulk_attempt:
            raise ValueError("bulk rewards are terminal values of the one bulk attempt")
        moves = RackServingDeployment.transitions()
        size = len(next(iter(competences.values())))
        solved = {
            state: np.full(size, float(state[0] == 0 and state[1] != "food")) for state in moves
        }
        value = dict(solved)
        steps = {state: np.zeros(size) for state in moves}
        for _ in range(horizon):
            previous, previous_steps = value, steps
            value, steps = {}, {}
            for state, options in moves.items():
                best = solved[state].copy()
                best_steps = np.zeros(size)
                if not best.any():
                    for skill, successor in options:
                        if skill not in competences:
                            continue
                        competence = competences[skill]
                        success_steps, failure_steps = RACK_SKILL_STEPS[skill]
                        ends = one_bulk_attempt and skill in RACK_SWEEP_SKILLS
                        failed = 0.0 if ends else previous[state]
                        failed_steps = 0.0 if ends else previous_steps[state]
                        priced = bool(bulk_rewards) and skill in (bulk_rewards or {})
                        succeeded = (bulk_rewards or {})[skill] if priced else previous[successor]
                        succeeded_steps = 0.0 if priced else previous_steps[successor]
                        candidate = competence * (succeeded - step_weight * success_steps) + (
                            1.0 - competence
                        ) * (failed - step_weight * failure_steps)
                        candidate_steps = competence * (success_steps + succeeded_steps) + (
                            1.0 - competence
                        ) * (failure_steps + failed_steps)
                        better = candidate > best
                        best = np.where(better, candidate, best)
                        best_steps = np.where(better, candidate_steps, best_steps)
                value[state] = best
                steps[state] = best_steps
        start = (RACK_NUM_FOOD, "empty", True)
        return value[start], steps[start]


class RackServingPracticeDomain(PracticeDomain):
    """Nine inferred skills, with provider-owned human costs only during training.

    Beta(1, 3) and fixed-seed Monte Carlo expectation follow the existing shelf
    domain as explicit modeling assumptions, not measured rack competence.
    Human beliefs never enter deployment value and are dropped when not offered.
    No unverified symmetry or belief-merging reduction is enabled.
    """

    deployment_horizon: int = 2 * RACK_NUM_FOOD + 2
    default_search_depth: int = 3
    initial_competence_alpha: float | None = 1.0
    initial_competence_beta: float | None = 3.0
    sampler_skill: str = "GatherAndSweep"
    # Sweep is appended after the human skills, so every older belief keeps its
    # prior-seed slot.
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = tuple(
        (
            name,
            PracticeExampleSource.SAMPLER
            if name in RACK_SWEEP_SKILLS
            else PracticeExampleSource.OUTCOME,
        )
        for name in (*RACK_ROBOT_SKILLS[:-1], *RACK_HUMAN_SKILLS, RACK_ROBOT_SKILLS[-1])
    )
    # `RackServingConfig.sweep_skill=False` removes Sweep from the provider.
    # `bulk_transfer_only=True` removes PlaceFoodOnTray.
    optional_skill_names: frozenset[str] = frozenset({"Sweep", "PlaceFoodOnTray"})
    failure_effect_skills: frozenset[str] = frozenset((*RACK_ROBOT_SKILLS, *RACK_HUMAN_SKILLS))
    human_skill_names: frozenset[str] = frozenset(RACK_HUMAN_SKILLS)
    # Bulk skill name -> the reward of the outcome its success reaches; empty unless
    # the provider prices goals (`RackServingConfig.two_route_rewards`).
    bulk_rewards: tuple[tuple[str, float], ...] = ()

    def with_goal_rewards(self, *, provider: SkillProvider) -> "RackServingPracticeDomain":
        """This domain valuing each bulk skill at the largest reward of a predicate it
        adds that the provider prices (`SkillProvider.goal_rewards`); itself when the
        provider prices nothing."""
        rewards = provider.goal_rewards()
        if not rewards:
            return self
        return self.model_copy(
            update={
                "bulk_rewards": tuple(
                    (
                        skill.name,
                        max(
                            rewards[atom.predicate.name]
                            for atom in skill.add_effects
                            if atom.predicate.name in rewards
                        ),
                    )
                    for skill in provider.skills()
                    if skill.name in RACK_SWEEP_SKILLS
                    and any(atom.predicate.name in rewards for atom in skill.add_effects)
                )
            }
        )

    def robot_skills(self) -> tuple[str, ...]:
        """The modelled robot skills this domain still has beliefs for."""
        believed = dict(self.belief_sources)
        return tuple(name for name in RACK_ROBOT_SKILLS if name in believed)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in self.robot_skills()}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def _solve(
        self, *, competences: dict[str, NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return RackServingDeployment.solve(
            competences=competences,
            horizon=self.deployment_horizon,
            step_weight=self.deployment_cost_lambda * self.robot_step_price,
            # A provider under `bulk_transfer_only` offers no placement, and its
            # evaluation gives a task one bulk attempt.
            one_bulk_attempt="PlaceFoodOnTray" not in dict(self.belief_sources),
            bulk_rewards=dict(self.bulk_rewards) or None,
        )

    def _sampled_competences(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> dict[str, NDArray[np.float64]]:
        return {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0] for name in self.robot_skills()
        }

    def _expectation(
        self, *, beliefs: Mapping[str, SkillBelief]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._solve(
            competences=self._sampled_competences(
                beliefs=beliefs,
                rng=np.random.default_rng(RACK_EXPECTATION_SEED),
                count=RACK_EXPECTATION_SAMPLES,
            )
        )

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(
            self._solve(
                competences={
                    name: np.array([theta[name].competence]) for name in self.robot_skills()
                }
            )[0][0]
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        return self._solve(
            competences=self._sampled_competences(beliefs=beliefs, rng=rng, count=count)
        )[0]

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return float(self._expectation(beliefs=beliefs)[0].mean())

    def expected_deployment_cost(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        """Expected control steps per task, times the run's robot-step price."""
        return float(self._expectation(beliefs=beliefs)[1].mean()) * self.robot_step_price


RACK_SERVING_DOMAIN = RackServingPracticeDomain()
