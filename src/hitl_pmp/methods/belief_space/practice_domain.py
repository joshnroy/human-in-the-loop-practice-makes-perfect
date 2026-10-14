"""What the belief-space practice method needs to know about one domain.

The search, the competence inference and the transition structure are domain-free; four
things are not, and a `PracticeDomain` supplies them:

- which skills carry a competence belief, which share one (Tossing3D's four picks share
  `PickCube`'s), and which event advances each one's training clock;
- the one skill whose continuous parameters a learned sampler chooses;
- which skills are known mechanisms that always succeed (Tossing3D's resets) rather
  than skills whose competence is inferred;
- the value of the deployment policy -- the probability a held-out task is solved,
  with no human -- as a function of the skills' competences.
"""

import abc
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from .tossing3d_constants import (
    OPEN_GRIPPER_SKILL,
    PICK_SKILL,
    PICK_SKILLS,
    RESET_SKILL,
    RESET_SKILLS,
    TOSS_SKILL,
)
from .tossing3d_deployment_model import (
    DeploymentPolicyExpectation,
    evaluate_deployment_policies,
    evaluate_deployment_policy,
)
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief, SkillHypothesis
from .types.theta import Tossing3DTheta

SampleSkills = Callable[[SkillBelief, int], list[SkillHypothesis]]


class PracticeDomain(BaseModel, abc.ABC):
    model_config = ConfigDict(frozen=True)

    deployment_horizon: int = Field(ge=0)
    sampler_skill: str
    # Belief names in prior-seeding order, each with what advances its clock.
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...]
    # Skill name -> the belief it shares.
    shared_beliefs: tuple[tuple[str, str], ...] = ()
    deterministic_skills: frozenset[str] = frozenset()
    # Known mechanisms whose search forecast is still weighted by their inferred
    # competence rather than assumed to succeed (Tossing3D's human reset).
    competence_weighted_skills: frozenset[str] = frozenset()
    # Skills whose failed executions teach the empirical failure-effect model.
    failure_effect_skills: frozenset[str] = frozenset()
    # The Beta prior on an untrained skill's competence; None keeps InferenceConfig's.
    initial_competence_alpha: float | None = None
    initial_competence_beta: float | None = None

    def belief_name(self, *, skill_name: str) -> str | None:
        shared = dict(self.shared_beliefs)
        if skill_name in shared:
            return shared[skill_name]
        if skill_name in dict(self.belief_sources) or skill_name in self.deterministic_skills:
            return skill_name
        return None

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


class Tossing3DPracticeDomain(PracticeDomain):
    """Exactly the model the Tossing3D method has always used."""

    deployment_horizon: int = 4
    sampler_skill: str = TOSS_SKILL
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = (
        (PICK_SKILL, PracticeExampleSource.OUTCOME),
        (TOSS_SKILL, PracticeExampleSource.SAMPLER),
        (OPEN_GRIPPER_SKILL, PracticeExampleSource.OUTCOME),
    )
    shared_beliefs: tuple[tuple[str, str], ...] = tuple(
        (name, PICK_SKILL) for name in sorted(PICK_SKILLS - {PICK_SKILL})
    )
    deterministic_skills: frozenset[str] = RESET_SKILLS
    competence_weighted_skills: frozenset[str] = frozenset({RESET_SKILL})
    failure_effect_skills: frozenset[str] = frozenset({
        *PICK_SKILLS,
        TOSS_SKILL,
        OPEN_GRIPPER_SKILL,
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
        return evaluate_deployment_policy(
            toss_competence=theta.toss.competence,
            pick_competence=theta.pick.competence,
            open_competence=theta.open_gripper.competence,
            horizon=self.deployment_horizon,
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        competences = [
            beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in (PICK_SKILL, TOSS_SKILL, OPEN_GRIPPER_SKILL)
        ]
        return evaluate_deployment_policies(
            toss_competences=competences[1],
            pick_competences=competences[0],
            open_competences=competences[2],
            horizon=self.deployment_horizon,
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return DeploymentPolicyExpectation.evaluate(
            pick=beliefs[PICK_SKILL],
            toss=beliefs[TOSS_SKILL],
            opened=beliefs[OPEN_GRIPPER_SKILL],
            horizon=self.deployment_horizon,
        )


TOSSING3D_DOMAIN = Tossing3DPracticeDomain()


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


SHELF_ROBOT_SKILLS = (
    "PickWiper",
    "SweepShelf",
    "ReturnWiper",
    "PickCubeFromShelf",
    "PickCubeFromFloor",
    "PlaceCubeInBox",
)
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
    failure_effect_skills: frozenset[str] = frozenset({*SHELF_ROBOT_SKILLS, *SHELF_HUMAN_SKILLS})

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in SHELF_ROBOT_SKILLS}
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
            name: beliefs[name].sample(rng=rng, count=count)[:, 0] for name in SHELF_ROBOT_SKILLS
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
