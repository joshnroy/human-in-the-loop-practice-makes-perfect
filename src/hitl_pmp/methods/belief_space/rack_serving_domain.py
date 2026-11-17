"""Rack-specific belief routing and an abstract, human-free deployment value.

This mirrors the shelf domain's analytical convention: successful actions apply
their symbolic effects and failures leave the deployment state unchanged. It is
not a model of settling, adhesion, spills, or automatic clock advancement during
robot motion. Practice uses the provider's full operators and learned failure
effects; actual evaluation still uses the live rack backend's verdict.
"""

from collections import OrderedDict
from collections.abc import Collection, Mapping
from functools import cache
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from hitl_pmp.core.method.skill_provider import SkillProvider

from .practice_domain import PracticeDomain, SampleSkills
from .rewarded_routes_domain import OPTIMISTIC_QUANTILE
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief
from .types.weighted_hypothesis_belief import WeightedHypothesisBelief

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
# `RackServingConfig.split_gather_and_sweep`: GatherAndSweep as Gather, which reaches
# the measured Gathered, then Sweep5, the bulk attempt from it. Kept out of
# RACK_ROBOT_SKILLS and RACK_SWEEP_SKILLS so a model without them is the old one.
RACK_CHAIN_SKILLS = ("Gather", "Sweep5")
# Every skill that is a task's one bulk attempt under `bulk_transfer_only`.
RACK_BULK_SKILLS = (*RACK_SWEEP_SKILLS, "Sweep5")
RACK_HUMAN_SKILLS = ("human_batch_return", "human_tool_recovery")
# The robot's own batch return (`RackServingConfig.robot_batch_return`): offered in
# practice only, so it is never a deployment transition and not in RACK_ROBOT_SKILLS.
# Its competence is inferred from its outcomes and its price from its observed steps,
# like any robot skill's; a provider that does not offer it drops its belief.
RACK_ROBOT_RESET_SKILLS = ("RobotBatchReturn",)
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
    # `split_gather_and_sweep`, measured 2026-11-17 on the launch scene (middle_three,
    # `scripts/measure_serving_routes.py`, six five-route episodes): Gather reached
    # Gathered in 4 (1797, 1778, 1794, 1710 steps) and missed in 2 (1689, 1686);
    # Sweep5 from Gathered served all five in 3 (389, 383, 383) and four in 1 (413).
    "Gather": (1770.0, 1688.0),
    "Sweep5": (385.0, 413.0),
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
    def transitions(
        *, chain: bool = False
    ) -> dict[RackDeploymentState, tuple[tuple[str, RackDeploymentState], ...]]:
        """With `chain`, one more state, (5, "gathered", False): the scraper held over
        a full, gathered rack, reached by Gather and left only by Sweep5 or by putting
        the scraper back."""
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
                    if chain and state == (RACK_NUM_FOOD, "scraper", False):
                        options.append(("Gather", (RACK_NUM_FOOD, "gathered", False)))
                    moves[state] = tuple(options)
        if chain:
            moves[RACK_NUM_FOOD, "gathered", False] = (
                ("ReturnScraper", (RACK_NUM_FOOD, "empty", True)),
                ("Sweep5", (0, "scraper", False)),
            )
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
        value, steps = RackServingDeployment._initial(competences=competences)
        for _ in range(horizon):
            value, steps, _ = RackServingDeployment._backup(
                competences=competences,
                previous=value,
                previous_steps=steps,
                step_weight=step_weight,
                one_bulk_attempt=one_bulk_attempt,
                bulk_rewards=bulk_rewards,
            )
        start = (RACK_NUM_FOOD, "empty", True)
        return value[start], steps[start]

    @staticmethod
    def plan(
        *,
        competences: dict[str, NDArray[np.float64]],
        horizon: int,
        step_weight: float,
        one_bulk_attempt: bool = False,
        bulk_rewards: Mapping[str, float] | None = None,
    ) -> dict[tuple[RackDeploymentState, int], tuple[str, RackDeploymentState] | None]:
        """The deployment policy `solve` values: for every abstract state and number of
        skills still allowed, the skill to attempt and the state its success reaches, or
        None to stop.

        It is `solve`'s own recursion, read out rather than re-derived: on one
        competence vector, following the plan earns exactly `solve`'s value. Over a
        batch `solve` lets each sample take its own best skill, which no one robot can
        do, so a batch has no single plan with `solve`'s mean value; pass one vector
        (`RackServingPracticeDomain.deployment_plan` passes the belief's mean). On a
        batch this takes each state's largest batch-mean option, which over-values
        waiting, because later choices are still valued per sample.
        """
        if bulk_rewards and not one_bulk_attempt:
            raise ValueError("bulk rewards are terminal values of the one bulk attempt")
        value, steps = RackServingDeployment._initial(competences=competences)
        table: dict[tuple[RackDeploymentState, int], tuple[str, RackDeploymentState] | None] = {}
        for remaining in range(1, horizon + 1):
            value, steps, choices = RackServingDeployment._backup(
                competences=competences,
                previous=value,
                previous_steps=steps,
                step_weight=step_weight,
                one_bulk_attempt=one_bulk_attempt,
                bulk_rewards=bulk_rewards,
            )
            for state, choice in choices.items():
                table[state, remaining] = choice
        return table

    @staticmethod
    def state_of(*, atoms: Collection[tuple[str, tuple[str, ...]]]) -> RackDeploymentState:
        """The abstract deployment state of measured atoms, as (name, object names)."""
        names = [name for name, _ in atoms]
        hand = (
            "scraper"
            if "HoldingScraper" in names
            else "food"
            if "HoldingFood" in names
            else "empty"
        )
        on_rack = names.count("OnRack")
        if hand == "scraper" and on_rack == RACK_NUM_FOOD and "Gathered" in names:
            hand = "gathered"
        return on_rack, hand, "ScraperHome" in names

    @staticmethod
    def _initial(
        *, competences: dict[str, NDArray[np.float64]]
    ) -> tuple[
        dict[RackDeploymentState, NDArray[np.float64]],
        dict[RackDeploymentState, NDArray[np.float64]],
    ]:
        moves = RackServingDeployment.transitions(chain="Gather" in competences)
        size = len(next(iter(competences.values())))
        value = {
            state: np.full(size, float(state[0] == 0 and state[1] != "food")) for state in moves
        }
        return value, {state: np.zeros(size) for state in moves}

    @staticmethod
    def _backup(
        *,
        competences: dict[str, NDArray[np.float64]],
        previous: dict[RackDeploymentState, NDArray[np.float64]],
        previous_steps: dict[RackDeploymentState, NDArray[np.float64]],
        step_weight: float,
        one_bulk_attempt: bool,
        bulk_rewards: Mapping[str, float] | None,
    ) -> tuple[
        dict[RackDeploymentState, NDArray[np.float64]],
        dict[RackDeploymentState, NDArray[np.float64]],
        dict[RackDeploymentState, tuple[str, RackDeploymentState] | None],
    ]:
        """One more allowed skill: each state's value and expected steps per sample, and
        the option whose batch-mean value is largest (None when stopping is)."""
        moves = RackServingDeployment.transitions(chain="Gather" in competences)
        size = len(next(iter(competences.values())))
        value, steps = {}, {}
        choices: dict[RackDeploymentState, tuple[str, RackDeploymentState] | None] = {}
        for state, options in moves.items():
            solved = np.full(size, float(state[0] == 0 and state[1] != "food"))
            best = solved.copy()
            best_steps = np.zeros(size)
            choice, choice_mean = None, float(solved.mean())
            if not best.any():
                for skill, successor in options:
                    if skill not in competences:
                        continue
                    competence = competences[skill]
                    success_steps, failure_steps = RACK_SKILL_STEPS[skill]
                    ends = one_bulk_attempt and skill in RACK_BULK_SKILLS
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
                    if float(candidate.mean()) > choice_mean:
                        choice, choice_mean = (skill, successor), float(candidate.mean())
            value[state] = best
            steps[state] = best_steps
            choices[state] = choice
        return value, steps, choices


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
    # Sweep is appended after the human skills, and the robot's batch return after
    # Sweep, so every older belief keeps its prior-seed slot.
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = tuple(
        (
            name,
            PracticeExampleSource.SAMPLER
            if name in RACK_SWEEP_SKILLS
            else PracticeExampleSource.OUTCOME,
        )
        for name in (
            *RACK_ROBOT_SKILLS[:-1],
            *RACK_HUMAN_SKILLS,
            RACK_ROBOT_SKILLS[-1],
            *RACK_ROBOT_RESET_SKILLS,
        )
    )
    # `RackServingConfig.sweep_skill=False` removes Sweep from the provider.
    # `bulk_transfer_only=True` removes PlaceFoodOnTray. `robot_batch_return=True`
    # is what offers RobotBatchReturn at all.
    optional_skill_names: frozenset[str] = frozenset({
        "Sweep",
        "PlaceFoodOnTray",
        *RACK_ROBOT_RESET_SKILLS,
    })
    failure_effect_skills: frozenset[str] = frozenset((
        *RACK_ROBOT_SKILLS,
        *RACK_HUMAN_SKILLS,
        *RACK_ROBOT_RESET_SKILLS,
    ))
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
        domain = self.with_split(provider=provider)
        return domain.model_copy(
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
                    if skill.name in RACK_BULK_SKILLS
                    and any(atom.predicate.name in rewards for atom in skill.add_effects)
                )
            }
        )

    def with_split(self, *, provider: SkillProvider) -> "RackServingPracticeDomain":
        """With `split_gather_and_sweep`'s Gather and Sweep5 on offer, this domain with a
        belief for each, appended after every other so no prior seed moves, each
        learning its own parameters' sampler as GatherAndSweep learned all three, and
        GatherAndSweep optional, so its belief is dropped; itself otherwise."""
        if not set(RACK_CHAIN_SKILLS) <= {skill.name for skill in provider.skills()}:
            return self
        return self.model_copy(
            update={
                "belief_sources": (
                    *self.belief_sources,
                    *((name, PracticeExampleSource.SAMPLER) for name in RACK_CHAIN_SKILLS),
                ),
                "optional_skill_names": self.optional_skill_names
                | {"GatherAndSweep", *RACK_CHAIN_SKILLS},
                "failure_effect_skills": self.failure_effect_skills | set(RACK_CHAIN_SKILLS),
            }
        )

    def robot_skills(self) -> tuple[str, ...]:
        """The modelled robot skills this domain still has beliefs for."""
        believed = dict(self.belief_sources)
        return tuple(name for name in (*RACK_ROBOT_SKILLS, *RACK_CHAIN_SKILLS) if name in believed)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in self.robot_skills()}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def _deployment(self) -> dict[str, Any]:
        return {
            "horizon": self.deployment_horizon,
            "step_weight": self.deployment_cost_lambda * self.robot_step_price,
            # A provider under `bulk_transfer_only` offers no placement, and its
            # evaluation gives a task one bulk attempt.
            "one_bulk_attempt": "PlaceFoodOnTray" not in dict(self.belief_sources),
            "bulk_rewards": dict(self.bulk_rewards) or None,
        }

    def _solve(
        self, *, competences: dict[str, NDArray[np.float64]]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return RackServingDeployment.solve(competences=competences, **self._deployment())

    def deployment_plan(
        self, *, beliefs: Mapping[str, SkillBelief]
    ) -> dict[tuple[RackDeploymentState, int], tuple[str, RackDeploymentState] | None]:
        """The executable policy of this value model for `beliefs`: `solve`'s recursion
        (`RackServingDeployment.plan`) with the same horizon, step weight, one-attempt
        rule and rewards, at the mean of the very competence samples
        `expected_policy_value` averages over.

        It is not the policy `expected_policy_value` prices, because that value has
        none: it averages, over the samples, the value of each sample's own best
        policy, as if the robot were told its competences before choosing."""
        samples = self._sampled_competences(
            beliefs=beliefs,
            rng=np.random.default_rng(RACK_EXPECTATION_SEED),
            count=RACK_EXPECTATION_SAMPLES,
        )
        return RackServingDeployment.plan(
            competences={name: np.array([float(draws.mean())]) for name, draws in samples.items()},
            **self._deployment(),
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

    def optimistic_policy_value(
        self,
        *,
        beliefs: Mapping[str, SkillBelief],
        tries_left: float,
        routes: Collection[str] | None = None,
    ) -> float:
        """An optimistic bound on the deployment value reachable with `tries_left` more
        practice tries, for the determinized search's open-list order.

        This value model at one competence vector, with no step costs: each robot skill
        at its reach, the OPTIMISTIC_QUANTILE of min(1, competence + rate * tries_left)
        over its belief, as if every try left went to that skill. Without step costs the
        model's value never falls as a competence rises (it maximises over policies whose
        every success moves to a state worth at least the one it leaves, or ends on a
        bulk reward), so at quantile 1 this bounds every split of the tries; the
        quantile trades that guarantee for an order the prior's tail does not flatten,
        as in the routes toy.

        With `routes`, the tries go to those bulk skills only, and a bulk skill not
        named is no way to deploy in this bound (the model's net-of-charges bound asks
        for one route at a time). Skills that are not bulk transfers keep their reach."""
        draws = self._reach_draws(beliefs=beliefs)
        if routes is not None and set(routes) & set(RACK_CHAIN_SKILLS):
            # Gather and Sweep5 are one route: a try of either is a try of the chain.
            routes = (*routes, *RACK_CHAIN_SKILLS)
        reach = tuple(
            (name, self._reach(draws=draws[name], tries_left=tries_left))
            for name in self.robot_skills()
            if routes is None or name not in RACK_BULK_SKILLS or name in routes
        )
        deployment = self._deployment() | {"step_weight": 0.0}
        key = (reach, deployment["horizon"], deployment["one_bulk_attempt"], self.bulk_rewards)
        memo = RackServingPracticeDomain._bound_memo
        if (cached := memo.get(key)) is not None:
            memo.move_to_end(key)
            return cached
        value = float(
            RackServingDeployment.solve(
                competences={name: np.array([r]) for name, r in reach}, **deployment
            )[0][0]
        )
        memo[key] = value
        while len(memo) > RackServingPracticeDomain.BOUND_MEMO_SIZE:
            memo.popitem(last=False)
        return value

    # One open-list bound values the same beliefs at every n <= tries left and for every
    # route, so the draws are taken once for the last beliefs mapping (held, so its
    # identity cannot be reused while it is cached), and the value model's solve is
    # memoised on the reach vector it reads. Bounded LRU.
    BOUND_MEMO_SIZE: ClassVar[int] = 4096
    _bound_memo: ClassVar[OrderedDict[tuple[object, ...], float]] = OrderedDict()
    _last_draws: ClassVar[list[Any]] = [None, None, None]

    def _reach_draws(
        self, *, beliefs: Mapping[str, SkillBelief]
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """Per robot skill, (competences, learning rates, weights) the reach is read
        from: a weighted hypothesis set exactly, any other belief as
        RACK_EXPECTATION_SAMPLES fixed-seed draws of equal weight."""
        last = RackServingPracticeDomain._last_draws
        skills = self.robot_skills()
        if last[0] is beliefs and last[1] == skills:
            return last[2]
        rng = np.random.default_rng(RACK_EXPECTATION_SEED)
        draws = {}
        for name in skills:
            belief = beliefs[name]
            if isinstance(belief, WeightedHypothesisBelief):
                draws[name] = (
                    np.asarray(belief.competences, dtype=np.float64),
                    np.asarray(belief.learning_rates, dtype=np.float64),
                    np.asarray(belief.probabilities, dtype=np.float64),
                )
            else:
                rows = belief.sample(rng=rng, count=RACK_EXPECTATION_SAMPLES)
                draws[name] = (rows[:, 0], rows[:, 1], np.full(len(rows), 1.0 / len(rows)))
        last[:] = [beliefs, skills, draws]
        return draws

    @staticmethod
    def _reach(
        *,
        draws: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
        tries_left: float,
    ) -> float:
        """The OPTIMISTIC_QUANTILE of a skill's competence after `tries_left` tries."""
        competences, rates, weights = draws
        reaches = np.minimum(1.0, competences + rates * tries_left)
        order = np.argsort(reaches, kind="stable")
        mass = np.cumsum(weights[order])
        index = int(np.searchsorted(mass, OPTIMISTIC_QUANTILE - 1e-12))
        return float(reaches[order][min(index, len(order) - 1)])


RACK_SERVING_DOMAIN = RackServingPracticeDomain()
