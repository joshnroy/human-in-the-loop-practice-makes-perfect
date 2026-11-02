"""Pure predicates over a captured KitchenState, never the live environment.

OnKitchenFloor excludes held and settled-in-bin items even if their geometry is
close to floor height. Together with HandEmpty and Pickable, it certifies the
held OpenGripper success contract, including the absence of NeedsRecovery.
"""

from functools import cache

import numpy as np

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

from .domain import KitchenConfig
from .types import KitchenObjects


def _flag(*, state: State, obj: Object, feature: str) -> bool:
    return bool(state.get(obj=obj, feature_name=feature) > 0.5)


def _feature_predicate(
    *, name: str, object_type: Type, feature: str, positive: bool = True
) -> Predicate:
    return Predicate(
        name=name,
        types=(object_type,),
        holds=lambda state, objects: (
            _flag(state=state, obj=objects[0], feature=feature) == positive
        ),
    )


HAND_EMPTY = _feature_predicate(
    name="HandEmpty", object_type=KitchenObjects.robot_type, feature="hand_empty"
)
HAND_OCCUPIED = _feature_predicate(
    name="HandOccupied", object_type=KitchenObjects.robot_type, feature="hand_empty", positive=False
)
TRAINING = _feature_predicate(
    name="Training", object_type=KitchenObjects.robot_type, feature="training"
)
BIN_NEAR = _feature_predicate(name="BinNear", object_type=KitchenObjects.bin_type, feature="near")
BIN_FAR = _feature_predicate(
    name="BinFar", object_type=KitchenObjects.bin_type, feature="near", positive=False
)
BIN_AT_TARGET = _feature_predicate(
    name="BinAtTarget", object_type=KitchenObjects.bin_type, feature="at_target"
)
BIN_AWAY = _feature_predicate(
    name="BinAway", object_type=KitchenObjects.bin_type, feature="at_target", positive=False
)
ON_KITCHEN_FLOOR = Predicate(
    name="OnKitchenFloor",
    types=(KitchenObjects.item_type,),
    holds=lambda state, objects: (
        _flag(state=state, obj=objects[0], feature="on_kitchen_floor")
        and not _flag(state=state, obj=objects[0], feature="held")
        and not _flag(state=state, obj=objects[0], feature="settled")
    ),
)
HOLDING = Predicate(
    name="Holding",
    types=(KitchenObjects.robot_type, KitchenObjects.item_type),
    holds=lambda state, objects: _flag(state=state, obj=objects[1], feature="held"),
)
IN_BIN = Predicate(
    name="InBin",
    types=(KitchenObjects.item_type, KitchenObjects.bin_type),
    holds=lambda state, objects: _flag(state=state, obj=objects[0], feature="settled"),
)
PICKABLE = Predicate(
    name="Pickable",
    types=(KitchenObjects.item_type,),
    holds=lambda state, objects: (
        _flag(state=state, obj=objects[0], feature="reachable")
        and not _flag(state=state, obj=objects[0], feature="held")
        and not _flag(state=state, obj=objects[0], feature="blocked")
    ),
)
NEEDS_RECOVERY = Predicate(
    name="NeedsRecovery",
    types=(KitchenObjects.item_type,),
    holds=lambda state, objects: (
        not _flag(state=state, obj=objects[0], feature="held")
        and (
            _flag(state=state, obj=objects[0], feature="settled")
            or not _flag(state=state, obj=objects[0], feature="on_kitchen_floor")
            or _flag(state=state, obj=objects[0], feature="blocked")
            or not _flag(state=state, obj=objects[0], feature="reachable")
        )
    ),
)
CLOSED_EMPTY = Predicate(
    name="ClosedEmpty",
    types=(KitchenObjects.robot_type,),
    holds=lambda state, objects: (
        not _flag(state=state, obj=objects[0], feature="hand_empty")
        and not any(
            _flag(state=state, obj=obj, feature="held")
            for obj in state.data
            if obj.type == KitchenObjects.item_type
        )
    ),
)


@cache
def item_identity(*, name: str) -> Predicate:
    """Pin each parent skill to one item without pooling learned statistics."""
    return Predicate(
        name=f"Is_{name}",
        types=(KitchenObjects.item_type,),
        holds=lambda state, objects: objects[0].name == name,
    )


def _bin_position_retrievable(
    *,
    position: tuple[float, ...],
    bounds: tuple[float, ...],
    robot_radius: float,
    reach: float,
    bin_outside: tuple[float, ...],
    tolerance: float = 0.0,
) -> bool:
    """A conservative reach certificate for every possible contained item center.

    The bin-origin sphere encloses its entire volume at any orientation. Distance
    to the allowed robot-center rectangle plus that radius bounds the observed
    environment reach test, independently of the near/far context label.
    """
    lower = np.array(bounds[:2]) + robot_radius
    upper = np.array(bounds[2:]) - robot_radius
    if np.any(lower > upper):
        return False
    center = np.array(position[:2])
    nearest = np.clip(center, lower, upper)
    radius = np.linalg.norm(np.array(bin_outside) * [0.5, 0.5, 1.0])
    return bool(np.linalg.norm(center - nearest) + radius + tolerance <= reach)


def bin_position_retrievable(
    *, config: KitchenConfig, position: tuple[float, ...], tolerance: float = 0.0
) -> bool:
    return _bin_position_retrievable(
        position=position,
        bounds=config.kitchen_bounds,
        robot_radius=config.robot_radius,
        reach=config.direct_placement_reach,
        bin_outside=config.bin_outside,
        tolerance=tolerance,
    )


@cache
def _bin_retrievable_predicate(
    *,
    bounds: tuple[float, ...],
    robot_radius: float,
    reach: float,
    bin_outside: tuple[float, ...],
) -> Predicate:
    return Predicate(
        name="BinRetrievable",
        types=(KitchenObjects.bin_type,),
        holds=lambda state, objects: _bin_position_retrievable(
            position=tuple(state.get(obj=objects[0], feature_name=axis) for axis in ("x", "y")),
            bounds=bounds,
            robot_radius=robot_radius,
            reach=reach,
            bin_outside=bin_outside,
        ),
    )


def bin_retrievable_predicate(*, config: KitchenConfig) -> Predicate:
    return _bin_retrievable_predicate(
        bounds=config.kitchen_bounds,
        robot_radius=config.robot_radius,
        reach=config.direct_placement_reach,
        bin_outside=config.bin_outside,
    )


class KitchenRecyclingPredicates:
    @staticmethod
    def all(*, config: KitchenConfig) -> tuple[Predicate, ...]:
        return (
            HAND_EMPTY,
            HAND_OCCUPIED,
            TRAINING,
            BIN_NEAR,
            BIN_FAR,
            BIN_AT_TARGET,
            BIN_AWAY,
            ON_KITCHEN_FLOOR,
            HOLDING,
            IN_BIN,
            PICKABLE,
            NEEDS_RECOVERY,
            CLOSED_EMPTY,
            bin_retrievable_predicate(config=config),
            *(item_identity(name=spec.name) for spec in config.objects),
        )

    @staticmethod
    def goal_atoms(*, config: KitchenConfig) -> frozenset[GroundAtom]:
        objects = KitchenObjects(config=config)
        return frozenset({
            GroundAtom(predicate=BIN_AT_TARGET, objects=(objects.bin,)),
            *(
                GroundAtom(predicate=IN_BIN, objects=(objects.by_name[name], objects.bin))
                for name in config.required_objects
            ),
        })
