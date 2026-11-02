"""Ordinary skills and samplers; no kitchen-specific learning algorithm."""

from functools import lru_cache

import numpy as np
from pydantic import Field

from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Action, Type
from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.pouring3d.practice.skill_provider import FrozenPourSkillProvider
from hitl_pmp.environments.pouring3d.practice.skills import PourSkills

from .environment import HUMAN_RESET, KitchenEnvironment
from .predicates import KitchenPredicates as P
from .types import KitchenObjects as O


class KitchenSkillProvider(FrozenPourSkillProvider):
    env: KitchenEnvironment
    human_reset_practice_cost: float = Field(default=5, gt=0, allow_inf_nan=False)

    @staticmethod
    @lru_cache(maxsize=1)
    def domain(*, count: int) -> tuple[Skill, ...]:
        cold = LiftedAtom(predicate=P.get(name="ColdIngredients"), variables=())
        base = tuple(
            s.model_copy(update={"preconditions": s.preconditions | {cold}})
            for s in PourSkills.build(count=count)
        )
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
            practice_cost=1,
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
            practice_cost=1,
        )
        return (*base[:-1], activation, away, base[-1].model_copy(update={"name": HUMAN_RESET}))

    def skills(self) -> tuple[Skill, ...]:
        return tuple(
            s
            for s in super().skills()
            if self.env.deployment or s.name not in {"ActivateCooking", "MoveAway"}
        )

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
        ground = self.ground(name=HUMAN_RESET)
        return ground.model_copy(
            update={
                "skill": ground.skill.model_copy(
                    update={"practice_cost": self.env.human_interruption_cost}
                )
            }
        )
