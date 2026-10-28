"""Pure predicate lookups into an observation's measured physical abstraction."""

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate

from .objects import PouringObjects as O
from .types import PouringState


class PouringPredicates:
    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        if not isinstance(state, PouringState):
            raise ValueError("Pouring predicates require an observed native PouringState")
        return (name, tuple(obj.name for obj in objects)) in state.abstract_atoms

    @staticmethod
    def predicate(*, name: str, types: tuple[Type, ...]) -> Predicate:
        return Predicate(
            name=name,
            types=types,
            holds=lambda state, objects: PouringPredicates.holds(
                state=state, name=name, objects=objects
            ),
        )

    @staticmethod
    def all() -> tuple[Predicate, ...]:
        return (
            PouringPredicates.predicate(name="HandEmpty", types=(O.robot_type,)),
            PouringPredicates.predicate(name="HoldingPitcher", types=(O.robot_type, O.vessel_type)),
            PouringPredicates.predicate(name="HoldingCube", types=(O.robot_type, O.cube_type)),
            PouringPredicates.predicate(name="InPitcher", types=(O.cube_type, O.vessel_type)),
            PouringPredicates.predicate(name="InTray", types=(O.cube_type, O.vessel_type)),
            PouringPredicates.predicate(name="OnFloor", types=(O.cube_type,)),
            PouringPredicates.predicate(name="Settled", types=(O.cube_type,)),
            PouringPredicates.predicate(name="Upright", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="AtHome", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="Empty", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="AllInPitcher", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="AllInTray", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="AnyInTray", types=(O.vessel_type,)),
            PouringPredicates.predicate(name="NoSpills", types=()),
            PouringPredicates.predicate(name="ReadyToPour", types=()),
            PouringPredicates.predicate(name="RequestedCountInTray", types=()),
            PouringPredicates.predicate(name="RequestedRemainderInPitcher", types=()),
            PouringPredicates.predicate(name="Goal", types=()),
        )
