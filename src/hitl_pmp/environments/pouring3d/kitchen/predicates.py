"""The shared pouring predicates plus the observed pick plannability."""

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
                for name, types in (("PickPlannable", (O.robot_type, O.shrimp_type)),)
            ),
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in KitchenPredicates.all() if p.name == name)
