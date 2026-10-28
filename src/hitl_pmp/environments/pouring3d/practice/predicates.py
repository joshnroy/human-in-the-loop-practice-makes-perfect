"""Pure symbolic lookups; measured facts never come from intended effects."""

from functools import lru_cache

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate

from .types import FrozenPourState
from .types import PourObjects as O


class PourPredicates:
    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        if not isinstance(state, FrozenPourState):
            raise ValueError("Pouring predicates require a measured FrozenPourState")
        return (name, tuple(o.name for o in objects)) in state.abstract_atoms

    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Predicate, ...]:
        specs: dict[str, tuple[Type, ...]] = {
            "HandEmpty": (O.robot_type,),
            "HoldingPitcher": (O.robot_type, O.pitcher_type),
            "HoldingShrimp": (O.robot_type, O.shrimp_type),
            "InPitcher": (O.shrimp_type, O.pitcher_type),
            "InTray": (O.shrimp_type, O.tray_type),
            "OnFloor": (O.shrimp_type,),
            "Settled": (O.shrimp_type,),
            "PitcherHome": (O.pitcher_type,),
            "TrayHome": (O.tray_type,),
            "PayloadOrder": (O.shrimp_type, O.shrimp_type),
            "ActiveRequest": (O.request_type,),
            **{f"Count{n}": (O.request_type,) for n in range(1, 6)},
            "SplitCorrect": (O.request_type,),
            "Goal": (O.request_type,),
            "NeedsReset": (),
            "NoSpills": (),
            "ReadyToPour": (),
        }
        return tuple(PourPredicates.make(name=name, types=types) for name, types in specs.items())

    @staticmethod
    def make(*, name: str, types: tuple[Type, ...]) -> Predicate:
        return Predicate(
            name=name,
            types=types,
            holds=lambda state, objects: PourPredicates.holds(
                state=state, name=name, objects=objects
            ),
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in PourPredicates.all() if p.name == name)
