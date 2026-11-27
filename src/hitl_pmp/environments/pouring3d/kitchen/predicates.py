"""The kitchen's own predicate table (Josh, 2026-11-06), read off the shared measurement.

`PourBackend.observe` still measures the legacy vocabulary, which the shrimp benchmark
keeps unchanged; `translate` maps it onto this table. The requests, `Count*`,
`ActiveRequest`, `Goal`, `NeedsReset`, `NoSpills` and `ReadyToPour` go, and `TrayHome`
with them because the skillet is welded to the stove (`KitchenScene.weld_skillet`).
`SplitCorrect(request_n)` becomes the zero-arity `Split{n}`: exactly n dumplings in the
skillet and the rest in the jug, all settled, none held.
"""

from functools import lru_cache

from hitl_pmp.core.problem.tasks.types import Predicate
from hitl_pmp.environments.pouring3d.practice.predicates import PourPredicates

from .types import KitchenObjects as O

Atoms = frozenset[tuple[str, tuple[str, ...]]]
SHARED = (
    "HandEmpty",
    "HoldingPitcher",
    "HoldingShrimp",
    "InPitcher",
    "InTray",
    "OnFloor",
    "OnSurface",
    "Settled",
    "PitcherHome",
    "PayloadOrder",
)
SPLIT_COUNTS = (1, 2, 3)


class KitchenPredicates:
    @staticmethod
    @lru_cache(maxsize=1)
    def all() -> tuple[Predicate, ...]:
        return (
            *(PourPredicates.get(name=name) for name in SHARED),
            *(PourPredicates.make(name=f"Split{n}", types=()) for n in SPLIT_COUNTS),
            PourPredicates.make(name="PickPlannable", types=(O.robot_type, O.shrimp_type)),
        )

    @staticmethod
    def get(*, name: str) -> Predicate:
        return next(p for p in KitchenPredicates.all() if p.name == name)

    @staticmethod
    def translate(*, atoms: Atoms) -> Atoms:
        """The legacy measured atoms in this table's vocabulary."""
        kept = {a for a in atoms if a[0] in (*SHARED, "PickPlannable")}
        kept |= {
            (f"Split{objects[0].removeprefix('request_')}", ())
            for name, objects in atoms
            if name == "SplitCorrect"
        }
        return frozenset(kept)
