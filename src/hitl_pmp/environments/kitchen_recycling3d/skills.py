"""Object/context-separated skills using the shared continuous-parameter learner.

Held OpenGripper models a successful safe kitchen-floor release: the observed
hand must be empty and the released item on the floor and pickable. Opening alone
is not success. A failed release retains its actual observed state for competence
and failure-effect learning, rather than installing these hypothetical effects.
Closed-empty OpenGripper only promises to open the unoccupied hand.
"""

from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable

from .domain import KitchenConfig
from .predicates import (
    BIN_AT_TARGET,
    BIN_AWAY,
    BIN_FAR,
    BIN_NEAR,
    CLOSED_EMPTY,
    HAND_EMPTY,
    HAND_OCCUPIED,
    HOLDING,
    IN_BIN,
    NEEDS_RECOVERY,
    ON_KITCHEN_FLOOR,
    PICKABLE,
    TRAINING,
    bin_position_retrievable,
    bin_retrievable_predicate,
    item_identity,
)
from .types import NEAR_BIN_POSITION, KitchenObjects

ROBOT = Variable(name="robot", type=KitchenObjects.robot_type)
ITEM = Variable(name="item", type=KitchenObjects.item_type)
BIN = Variable(name="bin", type=KitchenObjects.bin_type)
EMPTY = LiftedAtom(predicate=HAND_EMPTY, variables=(ROBOT,))
OCCUPIED = LiftedAtom(predicate=HAND_OCCUPIED, variables=(ROBOT,))
HELD = LiftedAtom(predicate=HOLDING, variables=(ROBOT, ITEM))
SETTLED = LiftedAtom(predicate=IN_BIN, variables=(ITEM, BIN))
PICK = LiftedAtom(predicate=PICKABLE, variables=(ITEM,))
FLOOR = LiftedAtom(predicate=ON_KITCHEN_FLOOR, variables=(ITEM,))
RECOVER = LiftedAtom(predicate=NEEDS_RECOVERY, variables=(ITEM,))
TRAIN = LiftedAtom(predicate=TRAINING, variables=(ROBOT,))
NEAR = LiftedAtom(predicate=BIN_NEAR, variables=(BIN,))
FAR = LiftedAtom(predicate=BIN_FAR, variables=(BIN,))
TARGET = LiftedAtom(predicate=BIN_AT_TARGET, variables=(BIN,))
AWAY = LiftedAtom(predicate=BIN_AWAY, variables=(BIN,))


class KitchenRecyclingSkills:
    """Names, identities and context guards remain distinct at the parent level."""

    @staticmethod
    def robot_skills(*, config: KitchenConfig) -> tuple[Skill, ...]:
        skills: list[Skill] = []
        retrievable = LiftedAtom(
            predicate=bin_retrievable_predicate(config=config), variables=(BIN,)
        )
        for spec in config.objects:
            identity = LiftedAtom(predicate=item_identity(name=spec.name), variables=(ITEM,))
            for context, context_atom in (("near", NEAR), ("far", FAR)):
                common = frozenset({identity, context_atom})
                skills.extend((
                    Skill(
                        name=f"Pick_{spec.name}_{context}",
                        parameters=(ROBOT, ITEM, BIN),
                        preconditions=common | {EMPTY, PICK},
                        add_effects=frozenset({HELD, OCCUPIED}),
                        delete_effects=frozenset({EMPTY, PICK, FLOOR, SETTLED, RECOVER}),
                        param_dim=0,
                        practice_cost=1.0,
                    ),
                    Skill(
                        name=f"Toss_{spec.name}_{context}",
                        parameters=(ROBOT, ITEM, BIN),
                        preconditions=common | {HELD},
                        add_effects=frozenset({EMPTY, SETTLED, RECOVER}),
                        delete_effects=frozenset({HELD, OCCUPIED, FLOOR, PICK}),
                        conditional_add_effects=frozenset({
                            ConditionalAddEffect(
                                conditions=frozenset({retrievable}), add_effects=frozenset({PICK})
                            )
                        }),
                        param_dim=3,
                        practice_cost=1.0,
                    ),
                    Skill(
                        name=f"OpenGripper_{spec.name}_{context}",
                        parameters=(ROBOT, ITEM, BIN),
                        preconditions=common | {HELD},
                        add_effects=frozenset({EMPTY, FLOOR, PICK}),
                        delete_effects=frozenset({HELD, OCCUPIED, SETTLED, RECOVER}),
                        param_dim=0,
                        practice_cost=1.0,
                    ),
                ))
        for context, context_atom in (("near", NEAR), ("far", FAR)):
            skills.append(
                Skill(
                    name=f"OpenGripperEmpty_{context}",
                    parameters=(ROBOT, BIN),
                    preconditions=frozenset({
                        LiftedAtom(predicate=CLOSED_EMPTY, variables=(ROBOT,)),
                        context_atom,
                    }),
                    add_effects=frozenset({EMPTY}),
                    delete_effects=frozenset({
                        OCCUPIED,
                        LiftedAtom(predicate=CLOSED_EMPTY, variables=(ROBOT,)),
                    }),
                    ignore_effects=frozenset({HOLDING}),
                    param_dim=0,
                    practice_cost=1.0,
                )
            )
        return tuple(skills)

    @staticmethod
    def human_skills(*, config: KitchenConfig) -> tuple[Skill, ...]:
        skills: list[Skill] = []
        retrievable = LiftedAtom(
            predicate=bin_retrievable_predicate(config=config), variables=(BIN,)
        )
        near_reach = (
            frozenset({retrievable})
            if bin_position_retrievable(config=config, position=NEAR_BIN_POSITION, tolerance=0.025)
            else frozenset()
        )
        target_reach = (
            frozenset({retrievable})
            if bin_position_retrievable(
                config=config, position=config.bin_position, tolerance=0.025
            )
            else frozenset()
        )
        for spec in config.objects:
            identity = LiftedAtom(predicate=item_identity(name=spec.name), variables=(ITEM,))
            for context, context_atom in (("near", NEAR), ("far", FAR)):
                skills.append(
                    Skill(
                        name=f"Recover_{spec.name}_{context}",
                        parameters=(ROBOT, ITEM, BIN),
                        preconditions=frozenset({TRAIN, EMPTY, identity, context_atom, RECOVER}),
                        add_effects=frozenset({FLOOR, PICK}),
                        delete_effects=frozenset({SETTLED, RECOVER}),
                        param_dim=0,
                        practice_cost=config.interruption_cost + config.relocation_cost,
                    )
                )
        skills.extend((
            Skill(
                name="RelocateBinNear",
                parameters=(ROBOT, BIN),
                preconditions=frozenset({TRAIN, EMPTY, FAR}),
                add_effects=frozenset({NEAR, AWAY}) | near_reach,
                delete_effects=frozenset({FAR, TARGET, retrievable}) - near_reach,
                ignore_effects=frozenset({IN_BIN}),
                param_dim=0,
                practice_cost=config.interruption_cost + config.relocation_cost,
            ),
            Skill(
                name="RestoreBinTarget",
                parameters=(ROBOT, BIN),
                preconditions=frozenset({TRAIN, EMPTY, AWAY}),
                add_effects=frozenset({FAR, TARGET}) | target_reach,
                delete_effects=frozenset({NEAR, AWAY, retrievable}) - target_reach,
                ignore_effects=frozenset({IN_BIN}),
                param_dim=0,
                practice_cost=config.interruption_cost + config.restoration_cost,
            ),
        ))
        return tuple(skills)
