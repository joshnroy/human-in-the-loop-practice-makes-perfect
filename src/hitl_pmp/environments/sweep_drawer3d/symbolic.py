"""Observed Sweep facts and planner operators; no simulator or scripted policy here.

The task has exactly five named cubes. One scene object carries a fixed feature
schema, making each cube-specific recovery operation a distinct zero-parameter
skill. Only the three stock controllers expose learnable continuous parameters.
"""

from functools import partial
from typing import ClassVar

from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate


class SweepSymbols:
    FACT_NAMES: ClassVar[tuple[str, ...]] = (
        "HandEmpty",
        "ClosedEmpty",
        "HoldingWiper",
        "WiperHome",
        "DrawerOpen",
        "DrawerClosed",
        "RobotHome",
        "AnyCubeInPile",
        *(
            f"{fact}{i}"
            for i in range(5)
            for fact in ("InPile", "InDrawer", "HoldingCube", "Pickable", "Blocked", "Loose")
        ),
    )
    CONTINUOUS: ClassVar[tuple[str, ...]] = (
        "seed",
        "base_x",
        "base_y",
        "base_yaw",
        "drawer_pos",
        "gripper",
        *(
            f"{name}_{axis}"
            for name in ("wiper", *(f"cube{i}" for i in range(5)))
            for axis in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ),
    )
    SCENE_TYPE: ClassVar[Type] = Type(name="sweep_scene", feature_names=FACT_NAMES + CONTINUOUS)
    SCENE: ClassVar[Object] = Object(name="scene", type=SCENE_TYPE)
    VARIABLE: ClassVar[Variable] = Variable(name="scene", type=SCENE_TYPE)
    TRAINABLE: ClassVar[tuple[str, ...]] = ("OpenDrawer", "PickWiper", "Sweep")

    @staticmethod
    def holds(state: State, objects: tuple[Object, ...], *, feature: str) -> bool:  # noqa: PLR0917 (predicate protocol)
        return state.get(obj=objects[0], feature_name=feature) > 0.5

    @staticmethod
    def predicates() -> tuple[Predicate, ...]:
        return tuple(SWEEP_PREDICATES.values())

    @staticmethod
    def skill(
        *,
        name: str,
        pre: tuple[str, ...],
        add: tuple[str, ...],
        delete: tuple[str, ...] = (),
        param_dim: int = 0,
        cost: float = 1.0,
    ) -> Skill:
        def atoms(*, names: tuple[str, ...]) -> frozenset[LiftedAtom]:
            return frozenset(
                LiftedAtom(predicate=SWEEP_PREDICATES[n], variables=(SweepSymbols.VARIABLE,))
                for n in names
            )

        return Skill(
            name=name,
            parameters=(SweepSymbols.VARIABLE,),
            preconditions=atoms(names=pre),
            add_effects=atoms(names=add),
            delete_effects=atoms(names=delete),
            param_dim=param_dim,
            practice_cost=cost,
        )

    @staticmethod
    def skills(*, costs: dict[str, float] | None = None) -> tuple[Skill, ...]:
        costs = costs or {}
        build = SweepSymbols.skill
        skills = [
            build(
                name="OpenDrawer",
                pre=("HandEmpty", "DrawerClosed"),
                add=("DrawerOpen",),
                delete=("DrawerClosed",),
                param_dim=2,
            ),
            build(
                name="PickWiper",
                pre=("HandEmpty", "WiperHome"),
                add=("HoldingWiper",),
                delete=("HandEmpty", "WiperHome"),
                param_dim=2,
            ),
            build(
                name="Sweep",
                pre=("HoldingWiper", "DrawerOpen", "AnyCubeInPile"),
                add=tuple(f"InDrawer{i}" for i in range(5)),
                delete=("AnyCubeInPile", *(f"InPile{i}" for i in range(5))),
                param_dim=2,
            ),
            build(
                name="OpenGripper",
                pre=("ClosedEmpty",),
                add=("HandEmpty",),
                delete=("ClosedEmpty",),
            ),
            build(
                name="ParkWiper",
                pre=("HoldingWiper",),
                add=("WiperHome", "HandEmpty"),
                delete=("HoldingWiper",),
            ),
            build(
                name="OpenResetDrawer",
                pre=("HandEmpty", "DrawerClosed"),
                add=("DrawerOpen",),
                delete=("DrawerClosed",),
            ),
            build(
                name="CloseDrawer",
                pre=("HandEmpty", "DrawerOpen"),
                add=("DrawerClosed",),
                delete=("DrawerOpen",),
            ),
            build(name="ParkRobot", pre=("HandEmpty",), add=("RobotHome",)),
        ]
        for i in range(5):
            skills.extend((
                build(
                    name=f"PickCube{i}",
                    pre=("HandEmpty", f"Pickable{i}", f"Loose{i}"),
                    add=(f"HoldingCube{i}",),
                    delete=("HandEmpty", f"Pickable{i}", f"InDrawer{i}", f"InPile{i}"),
                ),
                build(
                    name=f"PlaceCube{i}",
                    pre=(f"HoldingCube{i}",),
                    add=("HandEmpty", f"InPile{i}", "AnyCubeInPile"),
                    delete=(f"HoldingCube{i}", f"Loose{i}", f"Blocked{i}"),
                ),
                build(
                    name=f"NudgeCube{i}",
                    pre=("HandEmpty", f"Blocked{i}", f"Loose{i}"),
                    add=(f"Pickable{i}",),
                    delete=(f"Blocked{i}",),
                ),
                build(
                    name=f"PushCube{i}",
                    pre=("HandEmpty", f"Blocked{i}", f"Loose{i}"),
                    add=(f"Pickable{i}",),
                    delete=(f"Blocked{i}",),
                ),
            ))
        return tuple(
            s.model_copy(update={"practice_cost": costs.get(s.name, s.practice_cost)})
            for s in skills
        )

    @staticmethod
    def human_reset(*, cost: float) -> GroundSkill:
        # Fully specified symbolic target; execution must be checked against these
        # same classifiers. Available only through the practice reset interface.
        add = (
            "HandEmpty",
            "WiperHome",
            "DrawerClosed",
            "RobotHome",
            "AnyCubeInPile",
            *(f"InPile{i}" for i in range(5)),
        )
        skill = SweepSymbols.skill(
            name=ASK_FOR_RESET_CUBE_BIN_ONLY_NAME,
            pre=(),
            add=add,
            delete=tuple(n for n in SweepSymbols.FACT_NAMES if n not in add),
            cost=cost,
        )
        return GroundSkill(skill=skill, objects=(SweepSymbols.SCENE,))


SWEEP_PREDICATES = {
    name: Predicate(
        name=name, types=(SweepSymbols.SCENE_TYPE,), holds=partial(SweepSymbols.holds, feature=name)
    )
    for name in SweepSymbols.FACT_NAMES
}
