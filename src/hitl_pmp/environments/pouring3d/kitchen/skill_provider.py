"""Ordinary skills and samplers; no kitchen-specific learning algorithm."""

from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import model_validator

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Type
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.pouring3d.practice.skill_provider import (
    FrozenPourOracle,
    FrozenPourSkillProvider,
)
from hitl_pmp.environments.pouring3d.practice.skills import RESET_NAME as LEGACY_RESET
from hitl_pmp.environments.pouring3d.practice.skills import PourSkills

from .environment import HUMAN_RESET, KitchenEnvironment
from .predicates import KitchenPredicates as P
from .types import ClosedLoopPourParameters
from .types import KitchenObjects as O


class KitchenSkillProvider(FrozenPourSkillProvider):
    env: KitchenEnvironment
    # The closed-loop Pour learns only its tilt ramp rate (Josh, 2026-11-04); its approach,
    # ceiling, wiggle and return are fixed on `KitchenController`.
    ramp_rate_bounds: tuple[float, float] = ClosedLoopPourParameters.LIMITS

    @model_validator(mode="after")
    def validate_ramp_rate_bounds(self) -> "KitchenSkillProvider":
        stale = {"tilt_bounds", "amplitude_bounds", "duration_bounds"} & self.model_fields_set
        if stale:
            raise ValueError(
                f"The kitchen Pour learns only its ramp rate; {sorted(stale)} do not apply"
            )
        low, high = ClosedLoopPourParameters.LIMITS
        bounds = self.ramp_rate_bounds
        if not np.isfinite(bounds).all() or not low <= bounds[0] <= bounds[1] <= high:
            raise ValueError(f"Ramp rate bounds must be ordered and inside [{low}, {high}] deg/s")
        return self

    @staticmethod
    @lru_cache(maxsize=1)
    def domain(*, count: int) -> tuple[Skill, ...]:
        plannable = P.get(name="PickPlannable")

        def kitchen(s: Skill) -> Skill:  # noqa: PLR0917
            # Each dumpling pick requires that it has not already refused to plan
            # from this pose (see `pick_plannable.py`); the human pour-back
            # deliberately does not. Skills that move dumplings forget the atom; it is
            # re-read on observation.
            pre = s.preconditions
            if s.name.startswith("PickShrimp"):
                pre |= {LiftedAtom(predicate=plannable, variables=s.parameters[:2])}
            ignore = s.ignore_effects
            if s.name.startswith("Pour") or s.name == LEGACY_RESET:
                ignore |= {plannable}
            update: dict[str, Any] = {"preconditions": pre, "ignore_effects": ignore}
            if s.name.startswith("Pour"):
                update["param_dim"] = len(ClosedLoopPourParameters.model_fields)
            return s.model_copy(update=update)

        # The legacy benchmark's no-op Wait does no work here: EES spent nearly all its
        # practice on it and it flooded Curtis's search (Josh, 2026-11-05).
        base = tuple(kitchen(s) for s in PourSkills.build(count=count) if s.name != "Wait")
        return (*base[:-1], KitchenSkillProvider.human_reset(legacy=base[-1]))

    @staticmethod
    def human_reset(*, legacy: Skill) -> Skill:
        """The human pour-back, applicable from any practice state (Josh, 2026-11-06).

        The food-only reset required an empty hand and the jug at home, so a jug held
        after a spill, with its placement refused, was a dead end. Here the human also
        takes the jug or a dumpling from the hand and stands the jug at home. From the
        states the food-only reset covered, its successor is unchanged.
        """
        parameters = {v.name: v for v in legacy.parameters}
        r, p = parameters["r"], parameters["p"]
        slots = tuple(v for v in legacy.parameters if v.name.startswith("s"))

        def atom(name: str, *variables: Variable) -> LiftedAtom:  # noqa: PLR0917
            return LiftedAtom(predicate=P.get(name=name), variables=variables)

        hand_and_jug = {atom("HandEmpty", r), atom("PitcherHome", p)}
        return legacy.model_copy(
            update={
                "name": HUMAN_RESET,
                "preconditions": legacy.preconditions - hand_and_jug,
                "add_effects": legacy.add_effects | hand_and_jug,
                "delete_effects": legacy.delete_effects
                | {atom("HoldingPitcher", r, p)}
                | {atom("HoldingShrimp", r, s) for s in slots},
            }
        )

    def sample_params(self, *, ground_skill: GroundSkill, rng: np.random.Generator) -> np.ndarray:
        if not ground_skill.skill.name.startswith("Pour"):
            return np.zeros(0)
        return np.array([rng.uniform(*self.ramp_rate_bounds)])

    def native_pour_parameters(self, *, params: np.ndarray) -> np.ndarray:
        # Slots 4-5 carried the open-loop wiggle and duration; the kitchen fixes both.
        return np.r_[params, 0.0, 0.0]

    def predicates(self) -> tuple[Predicate, ...]:
        return P.all()

    def types(self) -> tuple[Type, ...]:
        return O.types()

    def human_cube_bin_reset_skill(self) -> GroundSkill | None:
        if self.env.deployment or not self.offer_human_reset:
            return None
        return self.ground(name=HUMAN_RESET)


class KitchenPourOracle(FrozenPourOracle):
    """The fixed-recipe reference with the closed-loop Pour's canonical ramp rate."""

    env: KitchenEnvironment
    # 1 deg/s delivered Pour1 and Pour2 exactly on seeds 0-3 in the 2026-11-04 prototype.
    parameters: tuple[float, ...] = (1.0,)

    def provider(self) -> FrozenPourSkillProvider:
        return KitchenSkillProvider(env=self.env)
