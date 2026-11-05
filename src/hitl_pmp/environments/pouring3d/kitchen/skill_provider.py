"""Ordinary skills and samplers; no kitchen-specific learning algorithm."""

from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import model_validator

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, Type
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
        cold = LiftedAtom(predicate=P.get(name="ColdIngredients"), variables=())
        plannable = P.get(name="PickPlannable")

        def kitchen(s: Skill) -> Skill:  # noqa: PLR0917
            # Each dumpling pick requires that it has not already refused to plan
            # from this pose (see `pick_plannable.py`); the human pour-back
            # deliberately does not. Skills that move dumplings forget the atom; it is
            # re-read on observation.
            pre = s.preconditions | {cold}
            if s.name.startswith("PickShrimp"):
                pre |= {LiftedAtom(predicate=plannable, variables=s.parameters[:2])}
            ignore = s.ignore_effects
            if s.name.startswith("Pour") or s.name == LEGACY_RESET:
                ignore |= {plannable}
            update: dict[str, Any] = {"preconditions": pre, "ignore_effects": ignore}
            if s.name.startswith("Pour"):
                update["param_dim"] = len(ClosedLoopPourParameters.model_fields)
            return s.model_copy(update=update)

        base = tuple(kitchen(s) for s in PourSkills.build(count=count))
        r, k = Variable(name="r", type=O.robot_type), Variable(name="k", type=O.request_type)
        # Hob is unique and needs no new binding type in the shared grounding interface.
        # Using a zero-argument appliance predicate would lose observed object typing;
        # ground() supplies h explicitly below.
        h = Variable(name="h", type=O.hob_type)

        def a(name: str, *vs: Variable) -> LiftedAtom:  # noqa: PLR0917
            return LiftedAtom(predicate=P.get(name=name), variables=vs)

        activation = Skill(
            name="ActivateCooking",
            parameters=(r, k, h),
            preconditions=frozenset({
                a("Goal", k),
                a("ActiveRequest", k),
                a("HandEmpty", r),
                a("HobOff", h),
                cold,
            }),
            add_effects=frozenset({a("HobOn", h)}),
            delete_effects=frozenset({a("HobOff", h), cold}),
            param_dim=0,
        )
        away = Skill(
            name="MoveAway",
            parameters=(r, k, h),
            preconditions=frozenset({
                a("HobOn", h),
                a("Goal", k),
                a("ActiveRequest", k),
                a("HandEmpty", r),
            }),
            add_effects=frozenset({a("RobotAway", r), a("MealStarted", k)}),
            delete_effects=frozenset(),
            param_dim=0,
        )
        return (*base[:-1], activation, away, base[-1].model_copy(update={"name": HUMAN_RESET}))

    def skills(self) -> tuple[Skill, ...]:
        return tuple(
            s
            for s in super().skills()
            if self.env.deployment or s.name not in {"ActivateCooking", "MoveAway"}
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

    def ground(self, *, name: str, request: int | None = None, shrimp: int = 0) -> GroundSkill:
        if name not in {"ActivateCooking", "MoveAway"}:
            return super().ground(name=name, request=request, shrimp=shrimp)
        request = self.env.requested_count if request is None else request
        if not 1 <= request <= 3:
            raise ValueError("Invalid request")
        skill = next(s for s in self.domain(count=3) if s.name == name)
        return GroundSkill(
            skill=skill, objects=(O.robot, self.env.objects().requests()[request - 1], O.hob)
        )

    def encode(self, *, ground_skill: GroundSkill, params: np.ndarray) -> Action:
        if ground_skill.skill.name not in {"ActivateCooking", "MoveAway"}:
            return super().encode(ground_skill=ground_skill, params=params)
        if params.shape != (0,):
            raise ValueError("Kitchen support actions have no continuous parameters")
        requested = int(ground_skill.objects[1].name.split("_")[-1])
        return np.array(
            [8 if ground_skill.skill.name == "ActivateCooking" else 9, 0, requested, 0, 0, 0],
            dtype=float,
        )

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
