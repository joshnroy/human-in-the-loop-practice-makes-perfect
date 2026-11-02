"""Measured appliance/robot predicates; activation is not a cooking simulation."""

from functools import lru_cache

from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.pouring3d.practice.predicates import PourPredicates

from .types import KitchenObjects as O


class KitchenPredicates:
    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Predicate, ...]:
        return (
            *PourPredicates.all(),
            *(
                PourPredicates.make(name=name, types=types)
                for name, types in (
                    ("HobOff", (O.hob_type,)),
                    ("HobOn", (O.hob_type,)),
                    ("ColdIngredients", ()),
                    ("RobotAway", (O.robot_type,)),
                    ("MealStarted", (O.request_type,)),
                )
            ),
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in KitchenPredicates.all() if p.name == name)
