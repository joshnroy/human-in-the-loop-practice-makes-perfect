"""The kitchen's skill table (Josh, 2026-11-06).

"Each" dumpling is the three skill parameters s0, s1, s2 bound once by the constant
`PayloadOrder` chain, and the skillet is the parameter t wherever `InTray` appears: a
`Skill` cannot name a fixed object in a precondition (Josh chose this fallback over
framework support, 2026-11-06). Native dispatch and the controllers are the legacy
benchmark's; only the symbolic model differs from `PourSkills`.

A successful `Pour{n}` takes n dumplings out of the jug and into the skillet, and
nothing symbolic says which: over 38 logged successes a Pour1 took dumpling 0, 1, 2 in
3, 5, 7 of 15 and a Pour2 each pair in 4, 3, 4 of 11. So a pour is still judged by
`Split{n}` alone, and declares one success alternative per subset of n for a forecast
to branch over. The skillet in them is a constant, not a parameter, so the pour's
groundings and its PDDL are what they were.
"""

from functools import lru_cache
from itertools import combinations

from hitl_pmp.core.method.types import LiftedAtom, Skill, SuccessAlternative, Variable

from .environment import HUMAN_RESET
from .predicates import SPLIT_COUNTS
from .predicates import KitchenPredicates as P
from .types import ClosedLoopPourParameters
from .types import KitchenObjects as O


class KitchenSkills:
    @staticmethod
    @lru_cache(maxsize=1)
    def build() -> tuple[Skill, ...]:
        r = Variable(name="r", type=O.robot_type)
        p = Variable(name="p", type=O.pitcher_type)
        t = Variable(name="t", type=O.tray_type)
        s = Variable(name="s", type=O.shrimp_type)
        each = tuple(Variable(name=f"s{i}", type=O.shrimp_type) for i in range(3))

        def atom(name: str, *variables: Variable) -> LiftedAtom:  # noqa: PLR0917
            return LiftedAtom(predicate=P.get(name=name), variables=variables)

        def skill(  # noqa: PLR0917
            name: str,
            variables: tuple[Variable, ...],
            pre: set[LiftedAtom],
            add: set[LiftedAtom],
            delete: set[LiftedAtom],
            ignore: tuple[str, ...] = (),
            alternatives: tuple[SuccessAlternative, ...] = (),
        ) -> Skill:
            return Skill(
                name=name,
                parameters=variables,
                preconditions=frozenset(pre),
                add_effects=frozenset(add),
                delete_effects=frozenset(delete),
                param_dim=len(ClosedLoopPourParameters.model_fields)
                if name.startswith("Pour")
                else 0,
                ignore_effects=frozenset(P.get(name=n) for n in ignore),
                success_alternatives=alternatives,
            )

        empty, held, home = (
            atom("HandEmpty", r),
            atom("HoldingPitcher", r, p),
            atom("PitcherHome", p),
        )
        chain = {atom("PayloadOrder", a, b) for a, b in zip(each, each[1:], strict=False)}
        inside = {atom("InPitcher", x, p) for x in each}
        settled = {atom("Settled", x) for x in each}
        splits = tuple(f"Split{n}" for n in SPLIT_COUNTS)
        held_s, plannable = atom("HoldingShrimp", r, s), atom("PickPlannable", r, s)
        picks = tuple(
            skill(
                f"PickShrimpFrom{where}",
                (r, s, t) if where == "Tray" else (r, s),
                {empty, source, plannable},
                {held_s},
                {empty, source},
                ignore=splits,
            )
            for where, source in (
                ("Tray", atom("InTray", s, t)),
                ("Floor", atom("OnFloor", s)),
                ("Surface", atom("OnSurface", s)),
            )
        )
        return (
            skill(
                "PickPitcher",
                (r, p, *each),
                {empty, home, *inside, *settled, *chain},
                {held},
                {empty, home},
            ),
            *(
                skill(
                    f"Pour{n}",
                    (r, p, *each),
                    {held, *inside, *chain},
                    {atom(f"Split{n}")},
                    set(),
                    ignore=(
                        "InPitcher",
                        "InTray",
                        "OnFloor",
                        "OnSurface",
                        "Settled",
                        *splits,
                        "PickPlannable",
                    ),
                    alternatives=tuple(
                        SuccessAlternative(
                            add_effects=frozenset({
                                atom(f"Split{n}"),
                                *(atom("InTray", x, t) for x in leaving),
                            }),
                            delete_effects=frozenset({
                                *(atom("InPitcher", x, p) for x in leaving),
                                *(atom(name) for name in splits if name != f"Split{n}"),
                            }),
                            constant_bindings=((t, O.tray),),
                        )
                        for leaving in combinations(each, n)
                    ),
                )
                for n in SPLIT_COUNTS
            ),
            skill("PlacePitcherAtHome", (r, p), {held}, {empty, home}, {held}),
            *picks,
            skill(
                "DropShrimpIntoPitcher",
                (r, s, p),
                {held_s, home},
                {empty, atom("InPitcher", s, p), atom("Settled", s)},
                {held_s},
                ignore=splits,
            ),
            # The human pour-back, applicable from any practice state (#742): it takes
            # the jug or a dumpling from the hand, stands the jug at home and returns
            # every dumpling to it.
            skill(
                HUMAN_RESET,
                (r, p, t, *each),
                chain,
                {empty, home, *inside, *settled},
                {
                    held,
                    *(atom("HoldingShrimp", r, x) for x in each),
                    *(atom("InTray", x, t) for x in each),
                    *(atom("OnFloor", x) for x in each),
                    *(atom("OnSurface", x) for x in each),
                    *(atom(n) for n in splits),
                },
                ignore=("PickPlannable",),
            ),
        )
