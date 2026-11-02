"""SweepTable3D practice beliefs and joint all-five deployment value."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .practice_domain import SampleSkills, SweepOffShelfPracticeDomain
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief

TABLE_DEPLOYMENT_SKILLS = ("PickWiper", "SweepTable")
RETURN_CUBE_BELIEF = "ReturnCubeToTable"
RETURN_CUBE_SKILLS = ("ReturnCubeToTableFromBox", "ReturnCubeToTableFromFloor")
TABLE_ROBOT_SKILLS = (
    *TABLE_DEPLOYMENT_SKILLS,
    "ReturnWiper",
    "PickCubeFromTable",
    "PlaceCubeInBox",
    "PickCubeFromFloor",
    RETURN_CUBE_BELIEF,
)
TABLE_HUMAN_SKILLS = ("reset_cubes", "reset_wiper")


class SweepTablePracticeDomain(SweepOffShelfPracticeDomain):
    """Reuse shelf priors, search configuration and common-random-number integration.

    SweepTable competence is the probability of one binary all-five outcome. Every
    robot and human skill is inferred; the two ball-return operators share a belief.
    Practice uses the generic symbolic transitions and empirical failure effects,
    including partial deliveries and loss of TaskActive on physical ground contact.
    Human resets affect which practice is reachable and its cost, while deployment
    starts from a full table and permits only tool pickup followed by one sweep.
    Per-ball manipulation and tool return can enable practice but do not contribute
    to deployment value: the goal is all five balls settled in the tray, not tool home.
    """

    sampler_skill: str = "SweepTable"
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = tuple(
        (
            name,
            PracticeExampleSource.SAMPLER
            if name == "SweepTable"
            else PracticeExampleSource.OUTCOME,
        )
        for name in (*TABLE_ROBOT_SKILLS, *TABLE_HUMAN_SKILLS)
    )
    shared_beliefs: tuple[tuple[str, str], ...] = tuple(
        (name, RETURN_CUBE_BELIEF) for name in RETURN_CUBE_SKILLS
    )
    failure_effect_skills: frozenset[str] = frozenset({
        *(name for name in TABLE_ROBOT_SKILLS if name != RETURN_CUBE_BELIEF),
        *RETURN_CUBE_SKILLS,
        *TABLE_HUMAN_SKILLS,
    })
    human_skill_names: frozenset[str] = frozenset(TABLE_HUMAN_SKILLS)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in TABLE_DEPLOYMENT_SKILLS}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        competences = {name: np.array([theta[name].competence]) for name in TABLE_DEPLOYMENT_SKILLS}
        return float(
            SweepTableDeployment.values(competences=competences, horizon=self.deployment_horizon)[0]
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        competences = {
            name: beliefs[name].sample(rng=rng, count=count)[:, 0]
            for name in TABLE_DEPLOYMENT_SKILLS
        }
        return SweepTableDeployment.values(competences=competences, horizon=self.deployment_horizon)


class SweepTableDeployment:
    """Probability of tool pickup followed by one settled, all-five sweep outcome.

    Sweep competence already describes the joint event, not a per-ball success
    probability. Failure has zero deployment value: a longer horizon does not buy
    another attempt, reset or individual pick/place delivery. Returning the tool is
    a practice operation, not part of this domain's tray-only deployment goal.
    """

    @staticmethod
    def values(*, competences: dict[str, NDArray[np.float64]], horizon: int) -> NDArray[np.float64]:
        """Batch solve probability from a full table and the tool at home."""
        if horizon < len(TABLE_DEPLOYMENT_SKILLS):
            return np.zeros_like(competences["PickWiper"], dtype=np.float64)
        return competences["PickWiper"] * competences["SweepTable"]


SWEEP_TABLE_DOMAIN = SweepTablePracticeDomain()
