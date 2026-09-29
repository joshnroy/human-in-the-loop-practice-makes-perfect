"""Observed Sweep facts and planner operators; no simulator or scripted policy here.

The task has exactly five named cubes. One scene object carries a fixed feature
schema, making each cube-specific recovery operation a distinct zero-parameter
skill. Only the three stock controllers expose learnable continuous parameters.
"""

from functools import partial
from itertools import combinations
from typing import ClassVar

from hitl_pmp.core.method.skill_provider import ASK_FOR_RESET_CUBE_BIN_ONLY_NAME
from hitl_pmp.core.method.types import GroundSkill, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import Predicate


class SweepSymbols:
    GROUPS: ClassVar[tuple[tuple[int, ...], ...]] = tuple(
        group for size in (2, 3) for group in combinations(range(5), size)
    )
    FACT_NAMES: ClassVar[tuple[str, ...]] = (
        "HandEmpty",
        "RecoveryHandEmpty",
        "PhysicallyHoldingWiper",
        "ResetDrawerOpen",
        "ResetDrawerClosed",
        "OnTableWiper",
        "InDrawerWiper",
        "ClosedEmpty",
        "RecoveryGripperClosed",
        "RecoveryArmHome",
        "ResetDrawerNotOpen",
        "WiperLow",
        *(f"GroupPickable{''.join(map(str, group))}" for group in GROUPS),
        "HoldingWiper",
        "WiperHome",
        "DrawerOpen",
        "DrawerClosed",
        "DrawerNotOpen",
        "DrawerNotClosed",
        "RobotHome",
        "RobotAway",
        "AnyCubeInPile",
        *(
            f"{fact}{i}"
            for i in range(5)
            for fact in (
                "OnTable",
                "PhysicallyInDrawer",
                "InPile",
                "InDrawer",
                "HoldingCube",
                "Pickable",
                "Blocked",
                "Loose",
                "SweepReachable",
            )
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
                pre=(
                    "HandEmpty",
                    "OnTableWiper",
                    "DrawerClosed",
                    *(f"OnTable{i}" for i in range(5)),
                ),
                add=("DrawerOpen",),
                delete=("DrawerClosed",),
                param_dim=2,
            ),
            build(
                name="PickWiper",
                pre=("HandEmpty", "OnTableWiper", "DrawerOpen", *(f"OnTable{i}" for i in range(5))),
                add=("HoldingWiper",),
                delete=("HandEmpty", "OnTableWiper"),
                param_dim=2,
            ),
            build(
                name="Sweep",
                pre=("HoldingWiper", "DrawerOpen", *(f"OnTable{i}" for i in range(5))),
                add=tuple(f"InDrawer{i}" for i in range(5)),
                delete=tuple(f"OnTable{i}" for i in range(5)),
                param_dim=2,
            ),
            build(
                name="OpenGripper",
                pre=("RecoveryGripperClosed",),
                add=("HandEmpty", "RecoveryHandEmpty"),
                delete=(
                    "ClosedEmpty",
                    "RecoveryGripperClosed",
                    "HoldingWiper",
                    "PhysicallyHoldingWiper",
                    *(f"HoldingCube{i}" for i in range(5)),
                ),
            ),
            build(
                name="ParkWiper",
                pre=("PhysicallyHoldingWiper",),
                add=("WiperHome", "HandEmpty", "RecoveryHandEmpty", "OnTableWiper"),
                delete=("HoldingWiper", "PhysicallyHoldingWiper", "RecoveryGripperClosed"),
            ),
            build(
                name="OpenResetDrawer",
                pre=("RecoveryHandEmpty", "ResetDrawerNotOpen"),
                add=("DrawerOpen", "ResetDrawerOpen", "DrawerNotClosed"),
                delete=("DrawerClosed", "ResetDrawerClosed", "DrawerNotOpen", "ResetDrawerNotOpen"),
            ),
            build(
                name="CloseDrawer",
                pre=("HandEmpty", "DrawerNotClosed"),
                add=("DrawerClosed", "ResetDrawerClosed", "DrawerNotOpen", "ResetDrawerNotOpen"),
                delete=("DrawerOpen", "ResetDrawerOpen", "DrawerNotClosed"),
            ),
            build(
                name="ParkRobot",
                pre=("HandEmpty", "RobotAway"),
                add=("RobotHome",),
                delete=("RobotAway",),
            ),
        ]
        for i in range(5):
            skills.extend((
                build(
                    name=f"PickCube{i}",
                    pre=("RecoveryHandEmpty", f"Loose{i}", f"Pickable{i}"),
                    add=(f"HoldingCube{i}", "RecoveryGripperClosed"),
                    delete=(
                        "HandEmpty",
                        "RecoveryHandEmpty",
                        f"Pickable{i}",
                        f"InDrawer{i}",
                        f"InPile{i}",
                        f"SweepReachable{i}",
                    ),
                ),
                build(
                    name=f"PlaceCube{i}",
                    pre=(f"HoldingCube{i}",),
                    add=(
                        "HandEmpty",
                        "RecoveryHandEmpty",
                        f"OnTable{i}",
                        f"InPile{i}",
                        "AnyCubeInPile",
                        f"SweepReachable{i}",
                    ),
                    delete=(f"HoldingCube{i}", f"Loose{i}", f"Blocked{i}", "RecoveryGripperClosed"),
                ),
                build(
                    name=f"WiggleDrawerCube{i}",
                    pre=(
                        "RecoveryHandEmpty",
                        "ResetDrawerOpen",
                        f"PhysicallyInDrawer{i}",
                        f"Blocked{i}",
                    ),
                    add=(f"Pickable{i}",),
                    delete=(f"Blocked{i}",),
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
        skills.append(
            build(
                name="RecoverWiper",
                pre=("RecoveryHandEmpty", "WiperLow"),
                add=(
                    "PhysicallyHoldingWiper",
                    "HoldingWiper",
                    "RecoveryGripperClosed",
                    "RecoveryArmHome",
                ),
                delete=("HandEmpty", "RecoveryHandEmpty", "WiperLow"),
            )
        )
        for group in SweepSymbols.GROUPS:
            suffix = "".join(map(str, group))
            skills.extend((
                build(
                    name=f"PickGroup{suffix}",
                    pre=(
                        "RecoveryHandEmpty",
                        f"GroupPickable{suffix}",
                        *(f"Loose{i}" for i in group),
                    ),
                    add=("RecoveryGripperClosed", *(f"HoldingCube{i}" for i in group)),
                    delete=(
                        "HandEmpty",
                        "RecoveryHandEmpty",
                        f"GroupPickable{suffix}",
                        *(f"Pickable{i}" for i in group),
                        *(f"InPile{i}" for i in group),
                    ),
                ),
                build(
                    name=f"PlaceGroup{suffix}",
                    pre=tuple(f"HoldingCube{i}" for i in group),
                    add=(
                        "HandEmpty",
                        "RecoveryHandEmpty",
                        "AnyCubeInPile",
                        *(f"InPile{i}" for i in group),
                        *(f"OnTable{i}" for i in group),
                        *(f"SweepReachable{i}" for i in group),
                    ),
                    delete=(
                        "RecoveryGripperClosed",
                        f"GroupPickable{suffix}",
                        *(f"HoldingCube{i}" for i in group),
                        *(f"Loose{i}" for i in group),
                        *(f"Blocked{i}" for i in group),
                    ),
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
            "RecoveryHandEmpty",
            "RecoveryArmHome",
            "OnTableWiper",
            *(f"OnTable{i}" for i in range(5)),
            "WiperHome",
            "ResetDrawerClosed",
            "ResetDrawerNotOpen",
            "DrawerClosed",
            "DrawerNotOpen",
            "RobotHome",
            "AnyCubeInPile",
            *(f"InPile{i}" for i in range(5)),
            *(f"SweepReachable{i}" for i in range(5)),
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
