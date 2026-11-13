"""What the belief-space practice method needs to know about one domain.

The search, the competence inference and the transition structure are domain-free; four
things are not, and a `PracticeDomain` supplies them:

- which skills carry a competence belief, which share one (Tossing3D's four picks share
  `PickCube`'s), and which event advances each one's training clock;
- which skills have continuous parameters chosen by learned samplers;
- which skills are known mechanisms that always succeed (Tossing3D's automatic
  non-human reset) rather than skills whose competence is inferred -- never a human
  skill, whose effects are checked and whose competence is learned like a robot's;
- the value of the deployment policy -- the probability a held-out task is solved,
  with no human -- as a function of the skills' competences.

A domain may also charge what deployment costs. The model sets `deployment_cost_lambda`
(its own linear-cost lambda) and `robot_step_price` (the run's price of one robot step,
in practice-cost units); a domain that models deployment cost then returns, from all
three value methods, E[solved] - deployment_cost_lambda * E[deployment cost per task],
the cost in those same practice-cost units, and reports the cost alone from
`expected_deployment_cost`. Both default to zero, and a domain that does not opt in
ignores them: its value is P(solve) exactly as before.
"""

import abc
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .tossing3d_constants import (
    NON_HUMAN_RESET_SKILL,
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    PICK_SKILLS,
    POUR_BIN_SKILL,
    RESET_OPPOSITE_SIDE_SKILL,
    RESET_SAME_SIDE_SKILL,
    TOSS_SKILL,
)
from .tossing3d_deployment_model import (
    DeploymentPolicyExpectation,
    evaluate_deployment_policies,
    evaluate_deployment_policy,
    evaluate_deployment_policy_costs,
)
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief, SkillHypothesis
from .types.theta import Tossing3DTheta

SampleSkills = Callable[[SkillBelief, int], list[SkillHypothesis]]


class PracticeDomain(BaseModel, abc.ABC):
    model_config = ConfigDict(frozen=True)

    deployment_horizon: int = Field(ge=0)
    sampler_skill: str = ""
    # Belief names in prior-seeding order, each with what advances its clock.
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...]
    # Keep seed slots when an ablation removes a skill from belief_sources.
    prior_seed_order: tuple[str, ...] = ()
    # Robot skills that may be disabled by the provider, unlike required skills.
    optional_skill_names: frozenset[str] = frozenset()
    # Skill name -> the belief it shares.
    shared_beliefs: tuple[tuple[str, str], ...] = ()
    deterministic_skills: frozenset[str] = frozenset()
    # Skills whose failed executions teach the empirical failure-effect model.
    failure_effect_skills: frozenset[str] = frozenset()
    # The human's skills this domain models. A method that does not offer one (the
    # always-ask wrapper, `--no-human-reset`) drops it: `PracticeDomainWithoutSkills`.
    human_skill_names: frozenset[str] = frozenset()
    # The Beta prior on an untrained skill's competence; None keeps InferenceConfig's.
    initial_competence_alpha: float | None = None
    initial_competence_beta: float | None = None
    # (belief name, alpha, beta): a Beta prior for that one skill, over the above.
    competence_prior_overrides: tuple[tuple[str, float, float], ...] = ()
    # Expectimax depth when the run does not set one.
    default_search_depth: int = Field(default=3, ge=0)
    # Reduce expectimax by exact object symmetry and count-keyed beliefs
    # (object_symmetry.py, `Tossing3DPracticeModel.search_cache_key`). Off by default,
    # so a domain that has never been checked for it searches exactly as before.
    reduce_expectimax_search: bool = False
    # Set by Tossing3DPracticeModel, never by a domain itself (see the module docstring).
    deployment_cost_lambda: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    robot_step_price: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    def expected_deployment_cost(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        """E[cost of one evaluation task] in practice-cost units, under the policy the
        value methods assume. Zero for a domain that does not model it."""
        del beliefs
        return 0.0

    def belief_name(self, *, skill_name: str) -> str | None:
        shared = dict(self.shared_beliefs)
        if skill_name in shared:
            return shared[skill_name]
        if skill_name in dict(self.belief_sources) or skill_name in self.deterministic_skills:
            return skill_name
        return None

    @property
    def sampler_skills(self) -> frozenset[str]:
        return frozenset(
            name for name, source in self.belief_sources if source == PracticeExampleSource.SAMPLER
        )

    def example_source(self, *, belief_name: str) -> PracticeExampleSource | None:
        if belief_name in self.deterministic_skills:
            return PracticeExampleSource.OUTCOME
        return dict(self.belief_sources).get(belief_name)

    @abc.abstractmethod
    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]: ...

    @abc.abstractmethod
    def evaluate_theta(self, *, theta: Any) -> float: ...

    @abc.abstractmethod
    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]: ...

    @abc.abstractmethod
    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float: ...


# Mean control steps per evaluation execution, from every evaluation task of the tossing
# env's v10/v10b reset-cost sweeps (kitchen-recycling-20261104-v10-costsweep and
# -v10b-highcost, all arms, actions.jsonl): PickCube 193.1 over 11031 executions,
# MoveToTossLocationAndToss 118.9 over 4780. OpenGripper never runs in evaluation there;
# its 39 is its mean charged practice cost (0.39 at 0.01 per step, 26 executions).
TOSSING3D_DEPLOYMENT_STEPS: tuple[tuple[str, float], ...] = (
    (PICK_SKILL, 193.1),
    (TOSS_SKILL, 118.9),
    (OPEN_GRIPPER_SKILL, 39.0),
)


class Tossing3DPracticeDomain(PracticeDomain):
    """Exactly the model the Tossing3D method has always used."""

    deployment_horizon: int = 4
    # Control steps one deployment execution of each skill costs; charged at
    # `robot_step_price` when deployment is priced. A backend with its own
    # measurements (the real robot) replaces it.
    deployment_steps: tuple[tuple[str, float], ...] = TOSSING3D_DEPLOYMENT_STEPS
    sampler_skill: str = TOSS_SKILL
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = (
        (PICK_SKILL, PracticeExampleSource.OUTCOME),
        (TOSS_SKILL, PracticeExampleSource.SAMPLER),
        (OPEN_GRIPPER_SKILL, PracticeExampleSource.OUTCOME),
        # The human resets are skills like any other: their promised effects are
        # checked and their competence inferred, so a human whose resets fail is
        # learned. Each destination is its own skill with its own belief; the
        # same-side reset keeps the single historical reset's prior-seed slot.
        (RESET_SAME_SIDE_SKILL, PracticeExampleSource.OUTCOME),
        # Append new skills so existing skill priors keep their historical seeds.
        (POUR_BIN_SKILL, PracticeExampleSource.OUTCOME),
        (RESET_OPPOSITE_SIDE_SKILL, PracticeExampleSource.OUTCOME),
    )
    optional_skill_names: frozenset[str] = frozenset({POUR_BIN_SKILL})
    shared_beliefs: tuple[tuple[str, str], ...] = tuple(
        (name, PICK_SKILL) for name in sorted(PICK_SKILLS - {PICK_SKILL})
    )
    deterministic_skills: frozenset[str] = frozenset({NON_HUMAN_RESET_SKILL})
    human_skill_names: frozenset[str] = frozenset({
        RESET_SAME_SIDE_SKILL,
        RESET_OPPOSITE_SIDE_SKILL,
    })
    failure_effect_skills: frozenset[str] = frozenset({
        *PICK_SKILLS,
        TOSS_SKILL,
        OPEN_GRIPPER_SKILL,
        POUR_BIN_SKILL,
        RESET_SAME_SIDE_SKILL,
        RESET_OPPOSITE_SIDE_SKILL,
    })

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        pick = sample_skills(beliefs[PICK_SKILL], count)
        toss = sample_skills(beliefs[TOSS_SKILL], count)
        opened = sample_skills(beliefs[OPEN_GRIPPER_SKILL], count)
        return [
            Tossing3DTheta.model_construct(
                pick=pick[index], toss=toss[index], open_gripper=opened[index]
            )
            for index in range(count)
        ]

    def evaluate_theta(self, *, theta: Any) -> float:
        value = evaluate_deployment_policy(
            toss_competence=theta.toss.competence,
            pick_competence=theta.pick.competence,
            open_competence=theta.open_gripper.competence,
            horizon=self.deployment_horizon,
        )
        if not self._prices_deployment():
            return value
        cost = evaluate_deployment_policy_costs(
            pick_competences=np.array([theta.pick.competence]),
            open_competences=np.array([theta.open_gripper.competence]),
            pick_costs=np.array([self._deployment_cost(skill_name=PICK_SKILL)]),
            toss_costs=np.array([self._deployment_cost(skill_name=TOSS_SKILL)]),
            open_costs=np.array([self._deployment_cost(skill_name=OPEN_GRIPPER_SKILL)]),
            horizon=self.deployment_horizon,
        )[0]
        return value - self.deployment_cost_lambda * float(cost)

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        pick, toss, opened = (
            beliefs[name].sample(rng=rng, count=count)
            for name in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)
        )
        values = evaluate_deployment_policies(
            toss_competences=toss[:, 0],
            pick_competences=pick[:, 0],
            open_competences=opened[:, 0],
            horizon=self.deployment_horizon,
        )
        if not self._prices_deployment():
            return values
        costs = evaluate_deployment_policy_costs(
            pick_competences=pick[:, 0],
            open_competences=opened[:, 0],
            pick_costs=np.full(count, self._deployment_cost(skill_name=PICK_SKILL)),
            toss_costs=np.full(count, self._deployment_cost(skill_name=TOSS_SKILL)),
            open_costs=np.full(count, self._deployment_cost(skill_name=OPEN_GRIPPER_SKILL)),
            horizon=self.deployment_horizon,
        )
        return values - self.deployment_cost_lambda * costs

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        if not self._prices_deployment():
            return DeploymentPolicyExpectation.evaluate(
                pick=beliefs[PICK_SKILL],
                toss=beliefs[TOSS_SKILL],
                opened=beliefs[OPEN_GRIPPER_SKILL],
                horizon=self.deployment_horizon,
            )
        value, cost = self._valued_with_cost(beliefs=beliefs)
        return value - self.deployment_cost_lambda * cost

    def expected_deployment_cost(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        """E[cost of one evaluation task] under the policy P(solve) assumes, in
        practice-cost units: `deployment_steps` at `robot_step_price`."""
        if not self._prices_deployment():
            return 0.0
        return self._valued_with_cost(beliefs=beliefs)[1]

    def _prices_deployment(self) -> bool:
        return self.deployment_cost_lambda > 0.0 and self.robot_step_price > 0.0

    def _deployment_cost(self, *, skill_name: str) -> float:
        return dict(self.deployment_steps)[skill_name] * self.robot_step_price

    def _valued_with_cost(self, *, beliefs: Mapping[str, SkillBelief]) -> tuple[float, float]:
        return DeploymentPolicyExpectation.evaluate_with_cost(
            pick=beliefs[PICK_SKILL],
            toss=beliefs[TOSS_SKILL],
            opened=beliefs[OPEN_GRIPPER_SKILL],
            horizon=self.deployment_horizon,
            pick_cost=self._deployment_cost(skill_name=PICK_SKILL),
            toss_cost=self._deployment_cost(skill_name=TOSS_SKILL),
            open_cost=self._deployment_cost(skill_name=OPEN_GRIPPER_SKILL),
        )


TOSSING3D_DOMAIN = Tossing3DPracticeDomain()

# The real TidyBot charges wall-clock seconds (0.1 per second; its robot step price is
# 0.01), so its deployment costs are its own measurements, in steps: the mean observed
# charge of every practice execution of the 2026-11-06 real runs (final-20261105,
# w4 and w10 step_events.jsonl): PickCube 3.642 over 25, MoveToTossLocationAndToss 2.829
# over 24. OpenGripper never ran on the robot; it keeps the sim's 39.
TOSSING3D_REAL_DEPLOYMENT_STEPS: tuple[tuple[str, float], ...] = (
    (PICK_SKILL, 364.2),
    (TOSS_SKILL, 282.9),
    (OPEN_GRIPPER_SKILL, 39.0),
)
TOSSING3D_REAL_DOMAIN = Tossing3DPracticeDomain(deployment_steps=TOSSING3D_REAL_DEPLOYMENT_STEPS)


class SweepOffShelfDeployment:
    """The held-out solve probability for SweepOffShelf3D, given each robot skill's
    competence: an optimal replanning policy over abstract cube counts with no human,
    where a failed skill leaves the abstract state unchanged. A static-method container.

    State: (cubes still on the shelf, what the hand holds, whether the wiper is home);
    a held cube is counted neither on the shelf nor in the box. Both routes are open --
    pick the wiper and sweep, or carry the cubes one at a time -- and the policy takes
    whichever the competences favour.
    """

    NUM_CUBES = 5

    @staticmethod
    @cache
    def _transitions() -> dict[
        tuple[int, str, bool], tuple[tuple[str, tuple[int, str, bool]], ...]
    ]:
        moves: dict[tuple[int, str, bool], tuple[tuple[str, tuple[int, str, bool]], ...]] = {}
        for shelf in range(SweepOffShelfDeployment.NUM_CUBES + 1):
            for hand in ("empty", "wiper", "cube"):
                for home in (True, False):
                    state = (shelf, hand, home)
                    options: list[tuple[str, tuple[int, str, bool]]] = []
                    if hand == "empty" and home:
                        options.append(("PickWiper", (shelf, "wiper", False)))
                    if hand == "wiper" and shelf:
                        options.append(("SweepShelf", (0, "wiper", False)))
                    if hand == "wiper":
                        options.append(("ReturnWiper", (shelf, "empty", True)))
                    if hand == "empty" and shelf:
                        options.append(("PickCubeFromShelf", (shelf - 1, "cube", home)))
                    if hand == "cube":
                        options.append(("PlaceCubeInBox", (shelf, "empty", home)))
                    moves[state] = tuple(options)
        return moves

    @staticmethod
    def values(*, competences: dict[str, NDArray[np.float64]], horizon: int) -> NDArray[np.float64]:
        """Batch value at the start state (all cubes on the shelf, wiper home)."""
        moves = SweepOffShelfDeployment._transitions()
        size = len(next(iter(competences.values())))
        solved = {
            state: np.full(size, float(state[0] == 0 and state[1] != "cube")) for state in moves
        }
        value = dict(solved)
        for _ in range(horizon):
            previous = value
            value = {}
            for state, options in moves.items():
                best = solved[state].copy()
                if not best.any():
                    for skill, successor in options:
                        p = competences[skill]
                        best = np.maximum(
                            best, p * previous[successor] + (1.0 - p) * previous[state]
                        )
                value[state] = best
        return value[(SweepOffShelfDeployment.NUM_CUBES, "empty", True)]


# The skills J's deployment policy is built from, and the order it draws them in.
SHELF_DEPLOYMENT_SKILLS = (
    "PickWiper",
    "SweepShelf",
    "ReturnWiper",
    "PickCubeFromShelf",
    "PickCubeFromFloor",
    "PlaceCubeInBox",
)
# One belief for both `ReturnCubeToShelf*` operators: one controller, one competence.
# J never reads it: deployment starts with every cube on the shelf and has no human,
# and putting a cube back on the shelf never brings one nearer the box.
RETURN_CUBE_BELIEF = "ReturnCubeToShelf"
RETURN_CUBE_SKILLS = ("ReturnCubeToShelfFromBox", "ReturnCubeToShelfFromFloor")
# Every robot belief.
SHELF_ROBOT_SKILLS = (*SHELF_DEPLOYMENT_SKILLS, RETURN_CUBE_BELIEF)
SHELF_HUMAN_SKILLS = ("reset_cubes", "reset_wiper")
# Fixed so that J compares stopping values under common random numbers.
SHELF_EXPECTATION_SAMPLES = 4096
SHELF_EXPECTATION_SEED = 0


class SweepOffShelfPracticeDomain(PracticeDomain):
    """Every robot and human skill carries its own inferred competence, from the same
    Beta(1, 3) prior: the human's `reset_cubes` succeeds only with some probability p,
    and the robot is not told p. No skill is deterministic here, so every imagined
    attempt -- human ones included -- branches on success and failure."""

    # The evaluation episode's horizon (SweepOffShelf3D's EPISODE_HORIZON).
    deployment_horizon: int = 12
    # Beta(1, 3): an untrained skill works one time in four. Under the default
    # Beta(2, 1) the 12-skill deployment value starts at 0.97, so practice looks
    # worthless before any evidence; Beta(1, 3) starts it at 0.555 (Tossing3D's own
    # prior value is 0.518).
    initial_competence_alpha: float | None = 1.0
    initial_competence_beta: float | None = 3.0
    # Five interchangeable cubes: exact symmetry plus count-keyed beliefs take the
    # search from depth 5 to depth 7 in about the same time per decision.
    default_search_depth: int = 7
    reduce_expectimax_search: bool = True
    sampler_skill: str = "SweepShelf"
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = tuple(
        (
            name,
            PracticeExampleSource.SAMPLER
            if name == "SweepShelf"
            else PracticeExampleSource.OUTCOME,
        )
        for name in (*SHELF_ROBOT_SKILLS, *SHELF_HUMAN_SKILLS)
    )
    shared_beliefs: tuple[tuple[str, str], ...] = tuple(
        (name, RETURN_CUBE_BELIEF) for name in RETURN_CUBE_SKILLS
    )
    failure_effect_skills: frozenset[str] = frozenset({
        *SHELF_DEPLOYMENT_SKILLS,
        *RETURN_CUBE_SKILLS,
        *SHELF_HUMAN_SKILLS,
    })
    human_skill_names: frozenset[str] = frozenset(SHELF_HUMAN_SKILLS)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in SHELF_DEPLOYMENT_SKILLS}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        competences = {name: np.array([hyp.competence]) for name, hyp in theta.items()}
        return float(
            SweepOffShelfDeployment.values(
                competences=competences, horizon=self.deployment_horizon
            )[0]
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        competences = {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in SHELF_DEPLOYMENT_SKILLS
        }
        return SweepOffShelfDeployment.values(
            competences=competences, horizon=self.deployment_horizon
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        rng = np.random.default_rng(SHELF_EXPECTATION_SEED)
        return float(
            self.sample_policy_values(
                beliefs=beliefs, rng=rng, count=SHELF_EXPECTATION_SAMPLES
            ).mean()
        )


SWEEP_OFF_SHELF_DOMAIN = SweepOffShelfPracticeDomain()


class PracticeDomainWithoutSkills:
    """A practice domain with some skills' beliefs removed: no belief, no failure-effect
    model, no shared-belief alias. Their deterministic entries are left alone -- they
    describe a skill only if it is offered. A static-method container."""

    @staticmethod
    def keep_offered(*, domain: PracticeDomain, offered: frozenset[str]) -> PracticeDomain:
        """Drop unavailable human/optional skills; retain required skills for validation."""
        return PracticeDomainWithoutSkills.drop(
            domain=domain,
            skill_names=(domain.human_skill_names | domain.optional_skill_names) - offered,
        )

    @staticmethod
    def drop(*, domain: PracticeDomain, skill_names: frozenset[str]) -> PracticeDomain:
        if not skill_names:
            return domain
        return domain.model_copy(
            update={
                "prior_seed_order": domain.prior_seed_order
                or tuple(name for name, _ in domain.belief_sources),
                "belief_sources": tuple(
                    (name, source)
                    for name, source in domain.belief_sources
                    if name not in skill_names
                ),
                "shared_beliefs": tuple(
                    (name, belief)
                    for name, belief in domain.shared_beliefs
                    if name not in skill_names and belief not in skill_names
                ),
                "failure_effect_skills": domain.failure_effect_skills - skill_names,
            }
        )
