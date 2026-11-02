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

from .practice_domain import PracticeDomain, SampleSkills
from .types.practice_example_source import PracticeExampleSource
from .types.skill_belief import SkillBelief

RACK_NUM_FOOD = 5
RACK_ROBOT_SKILLS = (
    "PickScraper",
    "SweepRack",
    "ReturnScraper",
    "PickFood",
    "PlaceFoodOnTray",
    "ReturnFoodToRack",
    "Wait",
)
RACK_HUMAN_SKILLS = ("human_batch_return", "human_tool_recovery")
RACK_EXPECTATION_SAMPLES = 4096
RACK_EXPECTATION_SEED = 0
RackDeploymentState = tuple[int, str, bool, bool]


class RackServingDeployment:
    """Value from five foods on the rack, with optional initial readiness.

    State is (rack count, hand contents, scraper home, ready). A held food is
    neither on rack nor on tray; all other food is on the tray. Return-to-rack
    and picking from the tray are legal, not forced recovery choices. The goal
    is readiness and every food on the tray, not a home scraper, matching the
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
                    for ready in (True, False):
                        state = (on_rack, hand, home, ready)
                        options: list[tuple[str, RackDeploymentState]] = []
                        if not ready:
                            options.append(("Wait", (on_rack, hand, home, True)))
                        if hand == "empty" and home:
                            options.append(("PickScraper", (on_rack, "scraper", False, ready)))
                        if hand == "scraper":
                            options.append(("ReturnScraper", (on_rack, "empty", True, ready)))
                            if ready and on_rack:
                                options.append(("SweepRack", (0, "scraper", False, True)))
                        if hand == "empty" and ready:
                            if on_rack:
                                options.append(("PickFood", (on_rack - 1, "food", home, True)))
                            if on_rack < RACK_NUM_FOOD:
                                options.append(("PickFood", (on_rack, "food", home, True)))
                        if hand == "food":
                            if ready:
                                options.append(("PlaceFoodOnTray", (on_rack, "empty", home, True)))
                            options.append((
                                "ReturnFoodToRack",
                                (on_rack + 1, "empty", home, ready),
                            ))
                        moves[state] = tuple(options)
        return moves

    @staticmethod
    def values(
        *, competences: dict[str, NDArray[np.float64]], horizon: int, initially_ready: bool
    ) -> NDArray[np.float64]:
        moves = RackServingDeployment.transitions()
        size = len(next(iter(competences.values())))
        solved = {
            state: np.full(size, float(state[3] and state[0] == 0 and state[1] != "food"))
            for state in moves
        }
        value = dict(solved)
        for _ in range(horizon):
            previous = value
            value = {}
            for state, options in moves.items():
                best = solved[state].copy()
                if not best.any():
                    for skill, successor in options:
                        competence = competences[skill]
                        best = np.maximum(
                            best,
                            competence * previous[successor] + (1.0 - competence) * previous[state],
                        )
                value[state] = best
        return value[(RACK_NUM_FOOD, "empty", True, initially_ready)]


class RackServingPracticeDomain(PracticeDomain):
    """Nine inferred skills, with provider-owned human costs only during training.

    Beta(1, 3) and fixed-seed Monte Carlo expectation follow the existing shelf
    domain as explicit modeling assumptions, not measured rack competence.
    Wait is inferred too: a cutoff can prevent its promised readiness effect.
    Human beliefs never enter deployment value and are dropped when not offered.
    No unverified symmetry or belief-merging reduction is enabled.
    """

    deployment_horizon: int = 2 * RACK_NUM_FOOD + 3
    initially_ready: bool = False
    default_search_depth: int = 3
    initial_competence_alpha: float | None = 1.0
    initial_competence_beta: float | None = 3.0
    sampler_skill: str = "SweepRack"
    belief_sources: tuple[tuple[str, PracticeExampleSource], ...] = tuple(
        (
            name,
            PracticeExampleSource.SAMPLER if name == "SweepRack" else PracticeExampleSource.OUTCOME,
        )
        for name in (*RACK_ROBOT_SKILLS, *RACK_HUMAN_SKILLS)
    )
    failure_effect_skills: frozenset[str] = frozenset((*RACK_ROBOT_SKILLS, *RACK_HUMAN_SKILLS))
    human_skill_names: frozenset[str] = frozenset(RACK_HUMAN_SKILLS)

    def sample_thetas(
        self, *, beliefs: Mapping[str, SkillBelief], sample_skills: SampleSkills, count: int
    ) -> list[Any]:
        samples = {name: sample_skills(beliefs[name], count) for name in RACK_ROBOT_SKILLS}
        return [{name: draws[index] for name, draws in samples.items()} for index in range(count)]

    def evaluate_theta(self, *, theta: Any) -> float:
        return float(
            RackServingDeployment.values(
                competences={
                    name: np.array([theta[name].competence]) for name in RACK_ROBOT_SKILLS
                },
                horizon=self.deployment_horizon,
                initially_ready=self.initially_ready,
            )[0]
        )

    def sample_policy_values(
        self, *, beliefs: Mapping[str, SkillBelief], rng: np.random.Generator, count: int
    ) -> NDArray[np.float64]:
        return RackServingDeployment.values(
            competences={
                name: beliefs[name].sample(rng=rng, count=count)[:, 0] for name in RACK_ROBOT_SKILLS
            },
            horizon=self.deployment_horizon,
            initially_ready=self.initially_ready,
        )

    def expected_policy_value(self, *, beliefs: Mapping[str, SkillBelief]) -> float:
        return float(
            self.sample_policy_values(
                beliefs=beliefs,
                rng=np.random.default_rng(RACK_EXPECTATION_SEED),
                count=RACK_EXPECTATION_SAMPLES,
            ).mean()
        )


RACK_SERVING_DOMAIN = RackServingPracticeDomain()
