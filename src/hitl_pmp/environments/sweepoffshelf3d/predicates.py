"""SweepOffShelf3D's predicates.

Each is a lookup into the atom set the backend derived from the live MuJoCo scene when
the state was built (`SweepOffShelf3DState.abstract_atoms`), so `holds` stays a pure
function of the state while the classifiers themselves can read contacts and geometry.
The one exception is `CubeOrder`, a static chain over cube names that lets the sweep
take every cube as a parameter while grounding exactly once.

`OnFloor` and `WiperOnFloor` are observed outcomes only: no operator ever adds them.
A cube or wiper that is neither where it belongs nor held is on the floor; the box rim
and the other shelves are not places it can come to rest.

Three summary atoms exist so that no skill can succeed vacuously. A skill is scored a
success when its promised effects hold afterwards, so a skill whose promises already
hold "succeeds" with nothing done -- a sweep of an empty shelf, or a `reset_cubes` with
every cube already on the shelf (40/40 and 62/62 such "successes" in the cancelled
batch). The operator language is conjunctive, so "some cube is on the shelf" needs an
atom of its own: `AnyCubeOnShelf`, `AnyCubeOffShelf` and `WiperAway` (not `WiperHome`)
are derived from the per-object atoms by `with_summaries`, which the backend applies
to every observed state.

The operators maintain them exactly except in one place: picking a cube off the shelf
cannot know whether it was the last one, so `PickCubeFromShelf` leaves
`AnyCubeOnShelf` as it was. A forecast can therefore hold a sweep applicable after the
last shelf cube was picked by hand; every observed state is classified afresh, so the
real applicability check is exact.
"""

import itertools

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

from .objects import NUM_CUBES
from .objects import SweepOffShelf3DObjects as O
from .types import AbstractAtom


class SweepOffShelf3DAtoms:
    """A static-method container, never instantiated."""

    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        atoms = getattr(state, "abstract_atoms", None)
        if atoms is None:
            raise ValueError(
                f"cannot evaluate {name} on a state with no abstraction attached; "
                "SweepOffShelf3D's classifiers read the live scene, so build states "
                "through SweepOffShelf3DEnvironment"
            )
        return (name, tuple(obj.name for obj in objects)) in atoms

    @staticmethod
    def with_summaries(*, atoms: frozenset[AbstractAtom]) -> frozenset[AbstractAtom]:
        """`atoms` plus the summary atoms they imply (see the module docstring)."""
        on_shelf = {objects for name, objects in atoms if name == "OnShelf"}
        derived: set[AbstractAtom] = set()
        if on_shelf:
            derived.add(("AnyCubeOnShelf", ()))
        if len(on_shelf) < NUM_CUBES:
            derived.add(("AnyCubeOffShelf", ()))
        if ("WiperHome", (O.wiper.name,)) not in atoms:
            derived.add(("WiperAway", (O.wiper.name,)))
        return atoms | derived

    @staticmethod
    def predicate(*, name: str, types: tuple[Type, ...]) -> Predicate:
        return Predicate(
            name=name,
            types=types,
            holds=lambda state, objects: SweepOffShelf3DAtoms.holds(
                state=state, name=name, objects=objects
            ),
        )


ON_SHELF = SweepOffShelf3DAtoms.predicate(name="OnShelf", types=(O.cube_type,))
IN_BOX = SweepOffShelf3DAtoms.predicate(
    name="InBox",
    types=(
        O.cube_type,
        O.box_type,
    ),
)
ON_FLOOR = SweepOffShelf3DAtoms.predicate(name="OnFloor", types=(O.cube_type,))
HAND_EMPTY = SweepOffShelf3DAtoms.predicate(name="HandEmpty", types=(O.robot_type,))
HOLDING_CUBE = SweepOffShelf3DAtoms.predicate(
    name="HoldingCube",
    types=(
        O.robot_type,
        O.cube_type,
    ),
)
HOLDING_WIPER = SweepOffShelf3DAtoms.predicate(
    name="HoldingWiper",
    types=(
        O.robot_type,
        O.wiper_type,
    ),
)
WIPER_HOME = SweepOffShelf3DAtoms.predicate(name="WiperHome", types=(O.wiper_type,))
WIPER_ON_FLOOR = SweepOffShelf3DAtoms.predicate(name="WiperOnFloor", types=(O.wiper_type,))
ANY_CUBE_ON_SHELF = SweepOffShelf3DAtoms.predicate(name="AnyCubeOnShelf", types=())
ANY_CUBE_OFF_SHELF = SweepOffShelf3DAtoms.predicate(name="AnyCubeOffShelf", types=())
WIPER_AWAY = SweepOffShelf3DAtoms.predicate(name="WiperAway", types=(O.wiper_type,))

_CUBE_CHAIN = frozenset((first.name, second.name) for first, second in itertools.pairwise(O.cubes))
CUBE_ORDER = Predicate(
    name="CubeOrder",
    types=(O.cube_type, O.cube_type),
    holds=lambda state, objects: (objects[0].name, objects[1].name) in _CUBE_CHAIN,
)

# The names the backend's classifier emits, in one place.
DYNAMIC_PREDICATES = (
    ON_SHELF,
    IN_BOX,
    ON_FLOOR,
    HAND_EMPTY,
    HOLDING_CUBE,
    HOLDING_WIPER,
    WIPER_HOME,
    WIPER_ON_FLOOR,
    ANY_CUBE_ON_SHELF,
    ANY_CUBE_OFF_SHELF,
    WIPER_AWAY,
)


class SweepOffShelf3DPredicates:
    """A static-method container, never instantiated."""

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
        return frozenset(GroundAtom(predicate=IN_BOX, objects=(cube, O.box)) for cube in O.cubes)
