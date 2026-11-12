from functools import lru_cache

from hitl_pmp.core.method.types import LiftedAtom, Skill, Variable

from .environment import HUMAN_RESET_B, RESET_A, ROUTE_A, ROUTE_B, TwoRoutesEnvironment
from .predicates import TwoRoutesPredicates


class TwoRoutesSkills:
    """Either route reaches `Done` from `Ready` and marks which one it was; each leaves
    the robot waiting for its own reset, whether or not it succeeded."""

    @staticmethod
    def build(
        *, name: str, pre: tuple[str, ...], add: tuple[str, ...], delete: tuple[str, ...]
    ) -> Skill:
        robot = Variable(name="robot", type=TwoRoutesEnvironment.robot_type)

        def atoms(*, names: tuple[str, ...]) -> frozenset[LiftedAtom]:
            return frozenset(
                LiftedAtom(predicate=TwoRoutesPredicates.get(name=n), variables=(robot,))
                for n in names
            )

        return Skill(
            name=name,
            parameters=(robot,),
            preconditions=atoms(names=pre),
            add_effects=atoms(names=add),
            delete_effects=atoms(names=delete),
            param_dim=0,
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Skill, ...]:
        """The robot's skills, in the environment's action-id order."""
        return (
            TwoRoutesSkills.build(
                name=ROUTE_A, pre=("Ready",), add=("Done", "ViaA", "AfterA"), delete=("Ready",)
            ),
            TwoRoutesSkills.build(
                name=ROUTE_B, pre=("Ready",), add=("Done", "ViaB", "AfterB"), delete=("Ready",)
            ),
            TwoRoutesSkills.build(
                name=RESET_A, pre=("AfterA",), add=("Ready",), delete=("AfterA", "Done", "ViaA")
            ),
        )

    @staticmethod
    @lru_cache(maxsize=1)
    def human_reset() -> Skill:
        """The human's reset after a Route B attempt: offered in practice only."""
        return TwoRoutesSkills.build(
            name=HUMAN_RESET_B, pre=("AfterB",), add=("Ready",), delete=("AfterB", "Done", "ViaB")
        )
