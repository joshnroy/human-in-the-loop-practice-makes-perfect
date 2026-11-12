from functools import lru_cache

from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import TwoRoutesEnvironment


class TwoRoutesPredicates:
    """One unary predicate per state flag: `ViaA`/`ViaB` say which route finished, and
    `AfterA`/`AfterB` whose reset the robot is waiting for."""

    NAMES = ("Ready", "Done", "ViaA", "ViaB", "AfterA", "AfterB")

    @staticmethod
    def holds(*, state: State, objects: tuple[Object, ...], feature_name: str) -> bool:
        return bool(state.get(obj=objects[0], feature_name=feature_name))

    @staticmethod
    def make(*, name: str, feature_name: str) -> Predicate:
        return Predicate(
            name=name,
            types=(TwoRoutesEnvironment.robot_type,),
            holds=lambda state, objects: TwoRoutesPredicates.holds(
                state=state, objects=objects, feature_name=feature_name
            ),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Predicate, ...]:
        return tuple(
            TwoRoutesPredicates.make(name=name, feature_name=feature_name)
            for name, feature_name in zip(
                TwoRoutesPredicates.NAMES,
                TwoRoutesEnvironment.robot_type.feature_names,
                strict=True,
            )
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in TwoRoutesPredicates.all() if p.name == name)
