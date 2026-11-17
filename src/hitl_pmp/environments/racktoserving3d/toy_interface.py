"""The chained toy's symbolic interface, over the 3D Serving scene (Josh, 2026-11-17).

`--env racktoserving3d_toy` is `tworoutes --both-resets --chained-route-b --robot-reset`
with the 3D scene deciding only whether each skill succeeded. Skill for skill:

    RouteA -> sweep3        RouteB -> gather        RouteB2 -> sweep5
    RobotReset -> robot_reset                       human_reset_b -> human_reset

and predicate for predicate:

    Ready -> FullRack   Done -> Served   ViaA -> ServedThree   ViaB -> ServedFive
    AfterA -> AfterSweep3   AfterB -> AfterGather   After -> NeedsReset
    MidB -> Gathered   AfterB2 -> AfterSweep5

Preconditions, add and delete lists and the bookkeeping atoms are the toy's
(`TwoRoutesSkills.chained`), so `RewardedRoutesPracticeDomain.from_provider` reads the
same route chain (sweep5 needs gather) and learns the same skills. The state is the
toy's too: which atoms hold is decided by which skill ran and whether it succeeded,
exactly as `TwoRoutesEnvironment.take_action` decides it. Success alone is measured
(`ServingToyTransitions.succeeded`)."""

from functools import lru_cache
from typing import ClassVar, Protocol

from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.tasks.types import Predicate

from .objects import RackServing3DObjects as Objects
from .predicates import RackServing3DAtoms

SWEEP3 = "sweep3"
GATHER = "gather"
SWEEP5 = "sweep5"
ROBOT_RESET = "robot_reset"
HUMAN_RESET = "human_reset"
# The robot's skills in the toy's order (`TwoRoutesSkills.chained`); a belief's prior
# seed follows its place.
ROBOT_SKILL_NAMES = (SWEEP3, GATHER, ROBOT_RESET, SWEEP5)
# Action IDs of their own, clear of `SKILL_IDS`, so no plain rack skill can be sent.
TOY_SKILL_IDS = {SWEEP3: 10, GATHER: 11, ROBOT_RESET: 12, SWEEP5: 13}
# In the toy's order (`TwoRoutesPredicates.NAMES`).
PREDICATE_NAMES = (
    "FullRack",
    "Served",
    "ServedThree",
    "ServedFive",
    "AfterSweep3",
    "AfterGather",
    "NeedsReset",
    "Gathered",
    "AfterSweep5",
)
# A Sweep that leaves at least this many on the plate served three; sweep5 must
# leave every dumpling there.
SWEEP3_COUNT = 3
SWEEP5_COUNT = len(Objects.food)


class Measured(Protocol):
    """What the scene says after a skill: the only input to its success."""

    @property
    def plate_count(self) -> int: ...

    @property
    def gathered(self) -> bool: ...

    @property
    def full_rack(self) -> bool: ...


class ServingToyPredicates:
    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Predicate, ...]:
        return tuple(
            RackServing3DAtoms.predicate(name=name, types=(Objects.robot_type,))
            for name in PREDICATE_NAMES
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in ServingToyPredicates.all() if p.name == name)


class ServingToySkills:
    # Continuous parameters: sweep3 and sweep5 take d, gather takes (s, delta), as the
    # split controllers' Sweep, Sweep5 and Gather do.
    PARAM_DIMS: ClassVar[dict[str, int]] = {SWEEP3: 1, GATHER: 2, SWEEP5: 1}
    EVERYTHING: ClassVar[tuple[str, ...]] = (
        "NeedsReset",
        "AfterSweep3",
        "AfterGather",
        "AfterSweep5",
        "Gathered",
        "Served",
        "ServedThree",
        "ServedFive",
    )

    @staticmethod
    def build(
        *, name: str, pre: tuple[str, ...], add: tuple[str, ...], delete: tuple[str, ...]
    ) -> Skill:
        robot = Variable(name="robot", type=Objects.robot_type)

        def atoms(*, names: tuple[str, ...]) -> frozenset[LiftedAtom]:
            return frozenset(
                LiftedAtom(predicate=ServingToyPredicates.get(name=n), variables=(robot,))
                for n in names
            )

        return Skill(
            name=name,
            parameters=(robot,),
            preconditions=atoms(names=pre),
            add_effects=atoms(names=add),
            delete_effects=atoms(names=delete),
            param_dim=ServingToySkills.PARAM_DIMS.get(name, 0),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def chained() -> tuple[Skill, ...]:
        """`TwoRoutesSkills.chained`, renamed."""
        build = ServingToySkills.build
        return (
            build(
                name=SWEEP3,
                pre=("FullRack",),
                add=("Served", "ServedThree", "NeedsReset", "AfterSweep3"),
                delete=("FullRack",),
            ),
            build(
                name=GATHER,
                pre=("FullRack",),
                add=("Gathered", "NeedsReset", "AfterGather"),
                delete=("FullRack",),
            ),
            build(
                name=ROBOT_RESET,
                pre=("NeedsReset",),
                add=("FullRack",),
                delete=ServingToySkills.EVERYTHING,
            ),
            build(
                name=SWEEP5,
                pre=("Gathered",),
                add=("Served", "ServedFive", "AfterSweep5"),
                delete=("Gathered", "AfterGather"),
            ),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def human_reset() -> Skill:
        """`TwoRoutesSkills.chained_human_reset`, renamed."""
        return ServingToySkills.build(
            name=HUMAN_RESET,
            pre=("NeedsReset",),
            add=("FullRack",),
            delete=ServingToySkills.EVERYTHING,
        )


class ServingToyTransitions:
    """`TwoRoutesEnvironment.take_action` and `reset_movables` under `chained_route_b`,
    on sets of atom names."""

    NEEDS: ClassVar[dict[str, str]] = {
        SWEEP3: "FullRack",
        GATHER: "FullRack",
        SWEEP5: "Gathered",
        ROBOT_RESET: "NeedsReset",
        HUMAN_RESET: "NeedsReset",
    }

    @staticmethod
    def initial() -> frozenset[str]:
        return frozenset({"FullRack"})

    @staticmethod
    def applicable(*, skill: str, flags: frozenset[str]) -> bool:
        return ServingToyTransitions.NEEDS[skill] in flags

    @staticmethod
    def after(*, skill: str, succeeded: bool, flags: frozenset[str]) -> frozenset[str]:
        """Each end is its own state, whatever the outcome; a reset that failed leaves
        the robot where it was. A gather that succeeded also holds NeedsReset, as the
        toy's MidB holds After (`TwoRoutesEnvironment.chained_end`)."""
        if skill in (ROBOT_RESET, HUMAN_RESET):
            return ServingToyTransitions.initial() if succeeded else flags
        if skill == SWEEP3:
            served = {"Served", "ServedThree"} if succeeded else set()
            return frozenset({"NeedsReset", "AfterSweep3", *served})
        if skill == GATHER:
            return frozenset({"NeedsReset", "AfterGather", *({"Gathered"} if succeeded else ())})
        if skill == SWEEP5:
            served = {"Served", "ServedFive"} if succeeded else set()
            return frozenset({"NeedsReset", "AfterSweep5", *served})
        raise ValueError(f"unknown skill: {skill}")

    @staticmethod
    def succeeded(*, skill: str, measured: Measured) -> bool:
        """sweep3: at least three on the plate; gather: the measured Gathered
        classifier (`RackServing3DAtoms.gathered`); sweep5: all five on the plate; a
        reset: the full rack is back."""
        if skill == SWEEP3:
            return measured.plate_count >= SWEEP3_COUNT
        if skill == SWEEP5:
            return measured.plate_count >= SWEEP5_COUNT
        if skill == GATHER:
            return measured.gathered
        if skill in (ROBOT_RESET, HUMAN_RESET):
            return measured.full_rack
        raise ValueError(f"unknown skill: {skill}")
