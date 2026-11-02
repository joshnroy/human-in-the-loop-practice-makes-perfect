"""Predicates read the backend's abstraction; CubeOrder is a static chain.

TaskActive permits one sweep. SweepSucceeded is earned only by that sweep's settled
all-five capture; tray occupancy from pick/place cannot establish it. Summaries never
invent TaskActive or success from positions. BallsRestored records explicit recovery.
"""

import itertools

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

from .objects import SweepTable3DObjects as O

AbstractAtom = tuple[str, tuple[str, ...]]


class SweepTable3DAtoms:
    """Shared atom lookup and summary derivation for observed states."""

    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        atoms = getattr(state, "abstract_atoms", None)
        if atoms is None:
            raise ValueError(
                f"cannot evaluate {name} on a state with no abstraction attached; "
                "build states through SweepTable3DEnvironment"
            )
        return (name, tuple(obj.name for obj in objects)) in atoms

    @staticmethod
    def with_summaries(*, atoms: frozenset[AbstractAtom]) -> frozenset[AbstractAtom]:
        """Recompute summaries without changing the backend's TaskActive atom."""
        base = frozenset(
            atom
            for atom in atoms
            if atom[0]
            not in {
                "AnyCubeOnTable",
                "AnyCubeOffTable",
                "WiperAway",
                "ResetBallsNeeded",
                "NoBallHeld",
            }
        )
        on_table = tuple(cube for cube in O.cubes if ("OnTable", (cube.name,)) in base)
        derived: set[AbstractAtom] = set()
        if not any(name == "HoldingCube" for name, _ in base):
            derived.add(("NoBallHeld", ()))
        if on_table:
            derived.add(("AnyCubeOnTable", ()))
        if len(on_table) < len(O.cubes):
            derived.add(("AnyCubeOffTable", ()))
            derived.add(("ResetBallsNeeded", ()))
            base = base - {("BallsRestored", ())}
        elif ("TaskActive", ()) not in base and ("BallsRestored", ()) not in base:
            derived.add(("ResetBallsNeeded", ()))
        if ("WiperHome", (O.wiper.name,)) not in base:
            derived.add(("WiperAway", (O.wiper.name,)))
        return base | derived

    @staticmethod
    def predicate(*, name: str, types: tuple[Type, ...]) -> Predicate:
        return Predicate(
            name=name,
            types=types,
            holds=lambda state, objects: SweepTable3DAtoms.holds(
                state=state, name=name, objects=objects
            ),
        )


ON_TABLE = SweepTable3DAtoms.predicate(name="OnTable", types=(O.cube_type,))
IN_BOX = SweepTable3DAtoms.predicate(name="InBox", types=(O.cube_type, O.box_type))
ON_FLOOR = SweepTable3DAtoms.predicate(name="OnFloor", types=(O.cube_type,))
HOLDING_CUBE = SweepTable3DAtoms.predicate(name="HoldingCube", types=(O.robot_type, O.cube_type))
HOLDING_WIPER = SweepTable3DAtoms.predicate(name="HoldingWiper", types=(O.robot_type, O.wiper_type))
WIPER_HOME = SweepTable3DAtoms.predicate(name="WiperHome", types=(O.wiper_type,))
WIPER_AWAY = SweepTable3DAtoms.predicate(name="WiperAway", types=(O.wiper_type,))
WIPER_ON_FLOOR = SweepTable3DAtoms.predicate(name="WiperOnFloor", types=(O.wiper_type,))
HAND_EMPTY = SweepTable3DAtoms.predicate(name="HandEmpty", types=(O.robot_type,))
ANY_CUBE_ON_TABLE = SweepTable3DAtoms.predicate(name="AnyCubeOnTable", types=())
ANY_CUBE_OFF_TABLE = SweepTable3DAtoms.predicate(name="AnyCubeOffTable", types=())
TASK_ACTIVE = SweepTable3DAtoms.predicate(name="TaskActive", types=())
SWEEP_SUCCEEDED = SweepTable3DAtoms.predicate(name="SweepSucceeded", types=())
BALLS_RESTORED = SweepTable3DAtoms.predicate(name="BallsRestored", types=())
RESET_BALLS_NEEDED = SweepTable3DAtoms.predicate(name="ResetBallsNeeded", types=())
NO_BALL_HELD = SweepTable3DAtoms.predicate(name="NoBallHeld", types=())

_CUBE_CHAIN = frozenset((first.name, second.name) for first, second in itertools.pairwise(O.cubes))
CUBE_ORDER = Predicate(
    name="CubeOrder",
    types=(O.cube_type, O.cube_type),
    holds=lambda state, objects: (objects[0].name, objects[1].name) in _CUBE_CHAIN,
)
DYNAMIC_PREDICATES = (
    ON_TABLE,
    IN_BOX,
    ON_FLOOR,
    HOLDING_CUBE,
    HOLDING_WIPER,
    WIPER_HOME,
    WIPER_AWAY,
    WIPER_ON_FLOOR,
    HAND_EMPTY,
    ANY_CUBE_ON_TABLE,
    ANY_CUBE_OFF_TABLE,
    TASK_ACTIVE,
    SWEEP_SUCCEEDED,
    BALLS_RESTORED,
    RESET_BALLS_NEEDED,
    NO_BALL_HELD,
)


class SweepTable3DPredicates:
    """The domain's observable vocabulary, static facts and fixed goal."""

    @staticmethod
    def all() -> tuple[Predicate, ...]:
        return (*DYNAMIC_PREDICATES, CUBE_ORDER)

    @staticmethod
    def static_atoms() -> frozenset[GroundAtom]:
        return frozenset(
            GroundAtom(predicate=CUBE_ORDER, objects=pair) for pair in itertools.pairwise(O.cubes)
        )

    @staticmethod
    def goal_atoms() -> frozenset[GroundAtom]:
        return frozenset({
            GroundAtom(predicate=SWEEP_SUCCEEDED, objects=()),
            *(GroundAtom(predicate=IN_BOX, objects=(ball, O.box)) for ball in O.balls),
        })
