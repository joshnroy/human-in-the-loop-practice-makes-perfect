"""Explicit new floor-sweep operators; native physical goal remains authoritative."""

from functools import partial
from typing import ClassVar

from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate


class SimpleSymbols:
    SCENE_FACTS: ClassVar[tuple[str, ...]] = (
        "HandEmpty",
        "HoldingWiper",
        "WiperAvailable",
        "WiperHome",
        "RobotHome",
        "RobotAway",
        "ClosedEmpty",
    )
    CUBE_FACTS: ClassVar[tuple[str, ...]] = (
        "AtStart",
        "NotAtStart",
        "InGoal",
        "NotInGoal",
        "OnFloor",
    )
    SCENE_TYPE: ClassVar[Type] = Type(name="simple_scene", feature_names=SCENE_FACTS + ("seed",))
    CUBE_TYPE: ClassVar[Type] = Type(name="simple_cube", feature_names=CUBE_FACTS + ("x", "y", "z"))
    SCENE: ClassVar[Object] = Object(name="scene", type=SCENE_TYPE)
    CUBES: ClassVar[tuple[Object, ...]] = ()
    SCENE_VAR: ClassVar[Variable] = Variable(name="scene", type=SCENE_TYPE)
    CUBE_VAR: ClassVar[Variable] = Variable(name="cube", type=CUBE_TYPE)
    TRAINABLE: ClassVar[tuple[str, ...]] = ("PickFloorWiper", "SweepCubeToGoal")

    @staticmethod
    def holds(state: State, objects: tuple[Object, ...], *, feature: str) -> bool:  # noqa: PLR0917
        return state.get(obj=objects[0], feature_name=feature) > 0.5

    @staticmethod
    def objects() -> tuple[Object, ...]:
        return (SimpleSymbols.SCENE, *SimpleSymbols.CUBES)

    @staticmethod
    def atoms(*, names: tuple[str, ...]) -> frozenset[LiftedAtom]:
        return frozenset(
            LiftedAtom(
                predicate=SIMPLE_PREDICATES[name],
                variables=(
                    SimpleSymbols.SCENE_VAR
                    if name in SimpleSymbols.SCENE_FACTS
                    else SimpleSymbols.CUBE_VAR,
                ),
            )
            for name in names
        )

    @staticmethod
    def skills(*, costs: dict[str, float] | None = None) -> tuple[Skill, ...]:
        costs = costs or {}
        result = []
        descriptions = (
            (
                "PickFloorWiper",
                False,
                ("HandEmpty", "WiperAvailable"),
                ("HoldingWiper", "RobotAway"),
                ("HandEmpty", "WiperAvailable", "WiperHome", "RobotHome"),
                2,
            ),
            (
                "SweepCubeToGoal",
                True,
                ("HoldingWiper", "OnFloor", "NotInGoal"),
                ("InGoal", "NotAtStart", "RobotAway"),
                ("NotInGoal", "AtStart", "RobotHome"),
                2,
            ),
            (
                "SweepCubeToStart",
                True,
                ("HoldingWiper", "OnFloor", "NotAtStart"),
                ("AtStart", "NotInGoal", "RobotAway"),
                ("NotAtStart", "InGoal", "RobotHome"),
                0,
            ),
            (
                "PlaceWiperAtStart",
                False,
                ("HoldingWiper",),
                ("HandEmpty", "WiperAvailable", "WiperHome"),
                ("HoldingWiper", "ClosedEmpty"),
                0,
            ),
            ("OpenGripper", False, (), ("HandEmpty",), ("HoldingWiper", "ClosedEmpty"), 0),
            (
                "ReturnRobotToStart",
                False,
                ("HandEmpty", "RobotAway"),
                ("RobotHome",),
                ("RobotAway",),
                0,
            ),
        )
        for name, cube, pre, add, delete, param_dim in descriptions:
            result.append(
                Skill(
                    name=name,
                    parameters=(SimpleSymbols.SCENE_VAR, SimpleSymbols.CUBE_VAR)
                    if cube
                    else (SimpleSymbols.SCENE_VAR,),
                    preconditions=SimpleSymbols.atoms(names=pre),
                    add_effects=SimpleSymbols.atoms(names=add),
                    delete_effects=SimpleSymbols.atoms(names=delete),
                    param_dim=param_dim,
                    practice_cost=costs.get(name, 1.0),
                )
            )
        return tuple(result)

    @staticmethod
    def initial_atoms() -> frozenset[GroundAtom]:
        return frozenset([
            *(
                GroundAtom(predicate=SIMPLE_PREDICATES[n], objects=(SimpleSymbols.SCENE,))
                for n in ("HandEmpty", "WiperAvailable", "WiperHome", "RobotHome")
            ),
            *(
                GroundAtom(predicate=SIMPLE_PREDICATES[n], objects=(cube,))
                for cube in SimpleSymbols.CUBES
                for n in ("AtStart", "NotInGoal", "OnFloor")
            ),
        ])

    @staticmethod
    def goal_atoms() -> frozenset[GroundAtom]:
        return frozenset(
            GroundAtom(predicate=SIMPLE_PREDICATES["InGoal"], objects=(cube,))
            for cube in SimpleSymbols.CUBES
        )

    @staticmethod
    def human_reset(*, cost: float) -> GroundSkill:
        objects = SimpleSymbols.objects()
        variables = tuple(Variable(name=obj.name, type=obj.type) for obj in objects)
        mapping = dict(zip(objects, variables, strict=True))
        adds = SimpleSymbols.initial_atoms()
        deletes = (
            frozenset(
                GroundAtom(predicate=pred, objects=(obj,))
                for obj in objects
                for pred in SIMPLE_PREDICATES.values()
                if pred.types == (obj.type,)
            )
            - adds
        )

        def lift(atoms: frozenset[GroundAtom]) -> frozenset[LiftedAtom]:  # noqa: PLR0917
            return frozenset(
                LiftedAtom(predicate=a.predicate, variables=tuple(mapping[o] for o in a.objects))
                for a in atoms
            )

        return GroundSkill(
            skill=Skill(
                name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
                parameters=variables,
                preconditions=frozenset(),
                add_effects=lift(adds),
                delete_effects=lift(deletes),
                param_dim=0,
                practice_cost=cost,
            ),
            objects=objects,
        )


SimpleSymbols.CUBES = tuple(
    Object(name=f"cube_{i}", type=SimpleSymbols.CUBE_TYPE) for i in range(5)
)

SIMPLE_PREDICATES = {
    name: Predicate(
        name=name,
        types=(
            SimpleSymbols.SCENE_TYPE
            if name in SimpleSymbols.SCENE_FACTS
            else SimpleSymbols.CUBE_TYPE,
        ),
        holds=partial(SimpleSymbols.holds, feature=name),
    )
    for name in (*SimpleSymbols.SCENE_FACTS, *SimpleSymbols.CUBE_FACTS)
}
