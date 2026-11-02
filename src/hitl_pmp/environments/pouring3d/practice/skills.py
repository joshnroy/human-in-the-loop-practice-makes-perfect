"""Each requested count is a distinct skill backed by the shared pour controller."""

from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable

from .predicates import PourPredicates as P
from .types import PourObjects as O

SKILL_NAMES = (
    "PickPitcher",
    "Pour",
    "PlacePitcherAtHome",
    "PickShrimpFromTray",
    "PickShrimpFromFloor",
    "DropShrimpIntoPitcher",
    "Wait",
    "PickShrimpFromSurface",
)
RESET_NAME = "reset_shrimp"


class PourSkills:
    @staticmethod
    def build(*, count: int) -> tuple[Skill, ...]:
        r = Variable(name="r", type=O.robot_type)
        p = Variable(name="p", type=O.pitcher_type)
        t = Variable(name="t", type=O.tray_type)
        k = Variable(name="k", type=O.request_type)
        s = Variable(name="s", type=O.shrimp_type)
        slots = tuple(Variable(name=f"s{i}", type=O.shrimp_type) for i in range(count))

        def atom(name: str, *variables: Variable) -> LiftedAtom:  # noqa: PLR0917
            return LiftedAtom(predicate=P.get(name=name), variables=variables)

        def skill(  # noqa: PLR0917
            name: str,
            variables: tuple[Variable, ...],
            pre: set[LiftedAtom],
            add: set[LiftedAtom],
            delete: set[LiftedAtom],
            dim: int = 0,
            ignore: tuple[str, ...] = (),
            conditional: frozenset[ConditionalAddEffect] = frozenset(),
        ) -> Skill:
            return Skill(
                name=name,
                parameters=variables,
                preconditions=frozenset(pre),
                add_effects=frozenset(add),
                delete_effects=frozenset(delete),
                param_dim=dim,
                ignore_effects=frozenset(P.get(name=n) for n in ignore),
                conditional_add_effects=conditional,
                practice_cost=1,
            )

        empty, held = atom("HandEmpty", r), atom("HoldingPitcher", r, p)
        home, tray = atom("PitcherHome", p), atom("TrayHome", t)
        split = atom("SplitCorrect", k)
        chain = {atom("PayloadOrder", a, b) for a, b in zip(slots, slots[1:], strict=False)}
        inside = {atom("InPitcher", x, p) for x in slots}
        settled = {atom("Settled", x) for x in slots}
        held_s = atom("HoldingShrimp", r, s)
        stale = ("SplitCorrect", "Goal", "ReadyToPour")
        picks = tuple(
            skill(
                name,
                (r, s, t) if source == "InTray" else (r, s),
                {empty, atom(source, s, t) if source == "InTray" else atom(source, s)},
                {held_s},
                {empty, atom(source, s, t) if source == "InTray" else atom(source, s)},
                ignore=(*stale, "NoSpills"),
            )
            for name, source in (
                ("PickShrimpFromTray", "InTray"),
                ("PickShrimpFromFloor", "OnFloor"),
                ("PickShrimpFromSurface", "OnSurface"),
            )
        )
        return (
            skill(
                "PickPitcher",
                (r, p, t, *slots),
                {empty, home, tray, *inside, *settled, *chain},
                {held},
                {empty, home},
                ignore=stale,
            ),
            *(
                skill(
                    f"Pour{n}",
                    (r, p, t, k, *slots),
                    {held, tray, atom(f"Count{n}", k), *inside, *chain},
                    {split, held, atom("NoSpills"), atom("NeedsReset")},
                    set(),
                    3,
                    ignore=("InPitcher", "InTray", "OnFloor", "OnSurface", "Settled", *stale),
                )
                for n in range(1, count + 1)
            ),
            skill(
                "PlacePitcherAtHome",
                (r, p, k),
                {held},
                {empty, home},
                {held},
                conditional=frozenset({
                    ConditionalAddEffect(
                        conditions=frozenset({split}), add_effects=frozenset({atom("Goal", k)})
                    )
                }),
            ),
            *picks,
            skill(
                "DropShrimpIntoPitcher",
                (r, s, p),
                {held_s, home},
                {empty, atom("InPitcher", s, p), atom("Settled", s)},
                {held_s},
                ignore=stale,
            ),
            skill("Wait", (), set(), set(), set()),
            skill(
                RESET_NAME,
                (r, p, t, *slots),
                {empty, home, tray, atom("NeedsReset"), *chain},
                {*inside, *settled, atom("ReadyToPour"), atom("NoSpills")},
                {atom("NeedsReset")},
                ignore=("InTray", "OnFloor", "OnSurface", *stale),
            ),
        )
