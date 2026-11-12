"""The kitchen's ground skills and symbolic states without a simulator (not a test module)."""

import itertools
import re

from hitl_pmp.core.method.types import GroundSkill, Skill
from hitl_pmp.core.problem.environment.types import Object
from hitl_pmp.core.problem.tasks.types import GroundAtom
from hitl_pmp.environments.pouring3d.kitchen.predicates import KitchenPredicates as P
from hitl_pmp.environments.pouring3d.kitchen.skills import KitchenSkills
from hitl_pmp.environments.pouring3d.kitchen.types import KitchenObjects as O


class KitchenSymbols:
    DUMPLINGS: tuple[Object, ...] = O(count=3, payload_kind="potsticker").shrimp()
    OBJECTS: tuple[Object, ...] = (O.robot, O.pitcher, O.tray, *DUMPLINGS)

    @staticmethod
    def atom(*, text: str) -> GroundAtom:
        """`Split1()` or `InTray(potsticker_0,tray)`, as the decision logs render atoms."""
        match = re.fullmatch(r"(\w+)\(([\w,]*)\)", text)
        assert match is not None, text
        by_name = {obj.name: obj for obj in KitchenSymbols.OBJECTS}
        names = [name for name in match.group(2).split(",") if name]
        return GroundAtom(
            predicate=P.get(name=match.group(1)), objects=tuple(by_name[n] for n in names)
        )

    @staticmethod
    def atoms(*, texts: list[str]) -> frozenset[GroundAtom]:
        return frozenset(KitchenSymbols.atom(text=text) for text in texts)

    @staticmethod
    def skill(*, name: str) -> Skill:
        return next(s for s in KitchenSkills.build() if s.name == name)

    @staticmethod
    def pour(*, count: int) -> GroundSkill:
        return GroundSkill(
            skill=KitchenSymbols.skill(name=f"Pour{count}"),
            objects=(O.robot, O.pitcher, *KitchenSymbols.DUMPLINGS),
        )

    @staticmethod
    def ground_skills() -> tuple[GroundSkill, ...]:
        """Every grounding the constant `PayloadOrder` chain admits."""
        pools = {
            O.robot_type: (O.robot,),
            O.pitcher_type: (O.pitcher,),
            O.tray_type: (O.tray,),
            O.shrimp_type: KitchenSymbols.DUMPLINGS,
        }
        chain = KitchenSymbols.chain()
        grounds = []
        for skill in KitchenSkills.build():
            for objects in itertools.product(*(pools[v.type] for v in skill.parameters)):
                ground = GroundSkill(skill=skill, objects=objects)
                ordered = {a for a in ground.preconditions if a.predicate.name == "PayloadOrder"}
                if ordered <= chain:
                    grounds.append(ground)
        return tuple(grounds)

    @staticmethod
    def chain() -> frozenset[GroundAtom]:
        names = [d.name for d in KitchenSymbols.DUMPLINGS]
        return KitchenSymbols.atoms(
            texts=[f"PayloadOrder({a},{b})" for a, b in itertools.pairwise(names)]
        )

    @staticmethod
    def holding_a_full_jug() -> frozenset[GroundAtom]:
        """The state every logged pour started from: a carried dumpling is not `Settled`."""
        names = [d.name for d in KitchenSymbols.DUMPLINGS]
        return KitchenSymbols.chain() | KitchenSymbols.atoms(
            texts=[
                "HoldingPitcher(robot,pitcher)",
                *(f"InPitcher({name},pitcher)" for name in names),
                *(f"PickPlannable(robot,{name})" for name in names),
            ]
        )
