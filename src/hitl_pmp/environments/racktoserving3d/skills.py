"""Core Skill templates, independent of environment/config implementations.

Pick/place delivery and pick/return recovery remain available without a scraper.
Food is ready from the start (Josh, 2026-11-03): there is no cooling and no Wait.
AnyFoodOnRack is conservatively retained when PickFood takes the last food off the
rack; ReturnFoodToRack declares AnyFoodOffRack exactly (`_STILL_OFF_RACK`). The
backend refreshes both summaries from measured locations after each action.
"""

from typing import ClassVar, Literal

from hitl_pmp.core.method.types import ConditionalAddEffect, LiftedAtom, Skill, Variable
from hitl_pmp.core.problem.tasks.types import Predicate

from .objects import NUM_FOOD
from .objects import RackServing3DObjects as Objects
from .predicates import (
    ANY_FOOD_OFF_RACK,
    ANY_FOOD_ON_RACK,
    FOOD_AVAILABLE,
    FOOD_ORDER,
    HAND_EMPTY,
    HOLDING_FOOD,
    HOLDING_SCRAPER,
    ON_RACK,
    ON_TRAY,
    PICK_PLANNABLE,
    SCRAPER_AWAY,
    SCRAPER_HOME,
    SCRAPER_ON_FLOOR,
    SERVED,
    SERVED_FIVE,
    SERVED_THREE,
    SPILLED_COUNTER,
    SPILLED_FLOOR,
)
from .semantics import HUMAN_BATCH_RETURN, HUMAN_TOOL_RECOVERY

SKILL_IDS = {
    "PickScraper": 0,
    "GatherAndSweep": 1,
    "ReturnScraper": 2,
    "PickFood": 3,
    "PlaceFoodOnTray": 4,
    "ReturnFoodToRack": 5,
    "Sweep": 6,
}
# GatherAndSweep(s, delta, d): s (m) is how far behind the rack's front edge the two
# x-squeeze passes stop; delta (m) is how far from the line y = c (the plate centre's
# y) each y gather stops, an absolute distance whatever the food; d (m) is how far
# before the plate's outer near edge the final sweep's blade edge ends. Starting
# bounds from the driven-blade grid (2026-11-03), not calibrated values. Speed is a
# constant, not a parameter.
# delta floor 0.005 (Josh, 2026-11-03): from the +-0.85 side stances the deepest
# stop that lifts cleanly on the arm is 0.5 cm from c; 0 cm failed. The range is the
# 2026-11-03 calibration's stops (then 0.045 m minus delta, delta in [-0.03, 0.04]),
# made absolute (Josh, 2026-11-04) so a food's size no longer moves them.
GATHER_AND_SWEEP_BOUNDS = ((0.03, 0.14), (0.005, 0.075), (0.04, 0.12))
# Sweep(d): GatherAndSweep's final stroke alone, over the same d bounds; offered unless
# `RackServingConfig.sweep_skill` is False.
SWEEP_BOUNDS = (GATHER_AND_SWEEP_BOUNDS[2],)
SWEEP_SPEED_M_S = 0.06
_ROBOT = Variable(name="robot", type=Objects.robot_type)
_FOOD = Variable(name="food", type=Objects.food_type)
_SCRAPER = Variable(name="scraper", type=Objects.scraper_type)
_RACK = Variable(name="rack", type=Objects.rack_type)
_TRAY = Variable(name="tray", type=Objects.tray_type)
_SLOTS = tuple(Variable(name=f"food{index}", type=Objects.food_type) for index in range(NUM_FOOD))


def _atom(*, predicate: Predicate, variables: tuple[Variable, ...] = ()) -> LiftedAtom:
    return LiftedAtom(predicate=predicate, variables=variables)


def _locations(*, food: Variable) -> frozenset[LiftedAtom]:
    return frozenset({
        _atom(predicate=ON_RACK, variables=(food, _RACK)),
        _atom(predicate=ON_TRAY, variables=(food, _TRAY)),
        _atom(predicate=SPILLED_COUNTER, variables=(food,)),
        _atom(predicate=SPILLED_FLOOR, variables=(food,)),
    })


_EMPTY = _atom(predicate=HAND_EMPTY, variables=(_ROBOT,))
_HELD_FOOD = _atom(predicate=HOLDING_FOOD, variables=(_ROBOT, _FOOD))
# PickFood needs it; skills that move the food make it true again (`pick_plannable.py`).
_PLANNABLE = _atom(predicate=PICK_PLANNABLE, variables=(_ROBOT, _FOOD))
_HELD_SCRAPER = _atom(predicate=HOLDING_SCRAPER, variables=(_ROBOT, _SCRAPER))
_HOME = _atom(predicate=SCRAPER_HOME, variables=(_SCRAPER,))
_AWAY = _atom(predicate=SCRAPER_AWAY, variables=(_SCRAPER,))
_ANY_ON = _atom(predicate=ANY_FOOD_ON_RACK)
_ANY_OFF = _atom(predicate=ANY_FOOD_OFF_RACK)
_AVAILABLE = _atom(predicate=FOOD_AVAILABLE, variables=(_FOOD,))
_CHAIN = frozenset(
    _atom(predicate=FOOD_ORDER, variables=(first, second))
    for first, second in zip(_SLOTS, _SLOTS[1:], strict=False)
)
# ReturnFoodToRack turns AnyFoodOffRack off only when it returns the last food off the
# rack. No guard can say "no other food is off the rack" with positive conditions, so the
# skill deletes the atom and these put it back whenever some food rests off the rack. The
# returned food is held, so it has no resting place and never matches. Food with no
# measured location at all (in flight, `classify_food`) is not seen here.
_OTHER = Variable(name="other_food", type=Objects.food_type)
_THE_TRAY = Variable(name="the_tray", type=Objects.tray_type)
_STILL_OFF_RACK = frozenset(
    ConditionalAddEffect(
        conditions=frozenset({resting}),
        add_effects=frozenset({_ANY_OFF}),
        constant_bindings=((_OTHER, food), (_THE_TRAY, Objects.tray)),
    )
    for food in Objects.food
    for resting in (
        _atom(predicate=ON_TRAY, variables=(_OTHER, _THE_TRAY)),
        _atom(predicate=SPILLED_COUNTER, variables=(_OTHER,)),
        _atom(predicate=SPILLED_FLOOR, variables=(_OTHER,)),
    )
)


class RackServing3DSkills:
    GATHER_AND_SWEEP_BOUNDS: ClassVar[tuple[tuple[float, float], ...]] = GATHER_AND_SWEEP_BOUNDS

    PICK_SCRAPER: ClassVar[Skill] = Skill(
        name="PickScraper",
        parameters=(_ROBOT, _SCRAPER),
        preconditions=frozenset({_EMPTY, _HOME}),
        add_effects=frozenset({_HELD_SCRAPER, _AWAY}),
        delete_effects=frozenset({_EMPTY, _HOME}),
        param_dim=0,
    )
    GATHER_AND_SWEEP: ClassVar[Skill] = Skill(
        name="GatherAndSweep",
        parameters=(_ROBOT, _SCRAPER, _RACK, _TRAY, *_SLOTS),
        # Every food on the rack (Josh, 2026-11-06): a sweep is a full-rack stroke, so food
        # left after a partial one is picked rather than swept past the served food.
        preconditions=frozenset({
            _HELD_SCRAPER,
            _ANY_ON,
            *_CHAIN,
            *(_atom(predicate=ON_RACK, variables=(food, _RACK)) for food in _SLOTS),
        }),
        add_effects=frozenset({_ANY_OFF}),
        delete_effects=frozenset({
            _ANY_ON,
            *(_atom(predicate=ON_RACK, variables=(food, _RACK)) for food in _SLOTS),
        }),
        conditional_add_effects=frozenset(
            ConditionalAddEffect(
                conditions=frozenset({_atom(predicate=ON_RACK, variables=(food, _RACK))}),
                add_effects=frozenset({_atom(predicate=ON_TRAY, variables=(food, _TRAY))}),
            )
            for food in _SLOTS
        ),
        param_dim=3,
    )
    SWEEP: ClassVar[Skill] = GATHER_AND_SWEEP.model_copy(update={"name": "Sweep", "param_dim": 1})
    RETURN_SCRAPER: ClassVar[Skill] = Skill(
        name="ReturnScraper",
        parameters=(_ROBOT, _SCRAPER),
        preconditions=frozenset({_HELD_SCRAPER}),
        add_effects=frozenset({_EMPTY, _HOME}),
        delete_effects=frozenset({_HELD_SCRAPER, _AWAY}),
        param_dim=0,
    )
    PICK_FOOD: ClassVar[Skill] = Skill(
        name="PickFood",
        parameters=(_ROBOT, _FOOD, _RACK, _TRAY),
        preconditions=frozenset({_EMPTY, _AVAILABLE, _PLANNABLE}),
        add_effects=frozenset({_HELD_FOOD, _ANY_OFF}),
        delete_effects=frozenset({_EMPTY, _AVAILABLE, *_locations(food=_FOOD)}),
        param_dim=0,
    )
    PLACE_FOOD_ON_TRAY: ClassVar[Skill] = Skill(
        name="PlaceFoodOnTray",
        parameters=(_ROBOT, _FOOD, _TRAY),
        preconditions=frozenset({_HELD_FOOD}),
        add_effects=frozenset({
            _EMPTY,
            _AVAILABLE,
            _PLANNABLE,
            _ANY_OFF,
            _atom(predicate=ON_TRAY, variables=(_FOOD, _TRAY)),
        }),
        delete_effects=frozenset({_HELD_FOOD}),
        param_dim=0,
    )
    RETURN_FOOD_TO_RACK: ClassVar[Skill] = Skill(
        name="ReturnFoodToRack",
        parameters=(_ROBOT, _FOOD, _RACK),
        preconditions=frozenset({_HELD_FOOD}),
        add_effects=frozenset({
            _EMPTY,
            _AVAILABLE,
            _PLANNABLE,
            _ANY_ON,
            _atom(predicate=ON_RACK, variables=(_FOOD, _RACK)),
        }),
        delete_effects=frozenset({_HELD_FOOD, _ANY_OFF}),
        conditional_add_effects=_STILL_OFF_RACK,
        param_dim=0,
    )
    HUMAN_BATCH_RETURN: ClassVar[Skill] = Skill(
        name=HUMAN_BATCH_RETURN,
        parameters=(_ROBOT, _RACK, _TRAY, *_SLOTS),
        preconditions=frozenset({_ANY_OFF, *_CHAIN}),
        add_effects=frozenset({
            _ANY_ON,
            *(_atom(predicate=ON_RACK, variables=(food, _RACK)) for food in _SLOTS),
            *(_atom(predicate=FOOD_AVAILABLE, variables=(food,)) for food in _SLOTS),
            *(_atom(predicate=PICK_PLANNABLE, variables=(_ROBOT, food)) for food in _SLOTS),
        }),
        delete_effects=frozenset({
            _ANY_OFF,
            *(
                atom
                for food in _SLOTS
                for atom in (
                    _atom(predicate=ON_TRAY, variables=(food, _TRAY)),
                    _atom(predicate=SPILLED_COUNTER, variables=(food,)),
                    _atom(predicate=SPILLED_FLOOR, variables=(food,)),
                    _atom(predicate=HOLDING_FOOD, variables=(_ROBOT, food)),
                )
            ),
        }),
        conditional_add_effects=frozenset(
            ConditionalAddEffect(
                conditions=frozenset({_atom(predicate=HOLDING_FOOD, variables=(_ROBOT, food))}),
                add_effects=frozenset({_EMPTY}),
            )
            for food in _SLOTS
        ),
        param_dim=0,
    )
    HUMAN_TOOL_RECOVERY: ClassVar[Skill] = Skill(
        name=HUMAN_TOOL_RECOVERY,
        parameters=(_ROBOT, _SCRAPER),
        preconditions=frozenset({_AWAY}),
        add_effects=frozenset({_HOME}),
        delete_effects=frozenset({
            _HELD_SCRAPER,
            _AWAY,
            _atom(predicate=SCRAPER_ON_FLOOR, variables=(_SCRAPER,)),
        }),
        conditional_add_effects=frozenset({
            ConditionalAddEffect(
                conditions=frozenset({_HELD_SCRAPER}), add_effects=frozenset({_EMPTY})
            ),
        }),
        param_dim=0,
    )

    # `RackServingConfig.two_route_rewards`: the same two bulk skills and batch return,
    # with success restated. GatherAndSweep promises all five as before, and so Served
    # and ServedFive. Sweep promises Served, ServedThree and the middle three slots on
    # the plate, and nothing about the outer two (Josh, 2026-11-12): spilled or still on
    # the rack, either is the declared outcome. So OnRack and AnyFoodOnRack are forgotten
    # (`ignore_effects`) rather than deleted: the model takes the rack as cleared, as it
    # did, and a measured outer piece still on it contradicts nothing. An ignore effect
    # is predicate-wide; that names exactly the outer two here because every OnRack atom
    # is one of the five the skill requires and the middle three are deleted anyway.
    # Either bulk try is still followed by the person's batch return, which withdraws
    # the outcome.
    # Known limit: the middle three slots are the middle three pieces only where the
    # start lines the pieces up in slot order (`start_layout` spaced, middle_three). On
    # a scattered start, or once a PickFood/ReturnFoodToRack has moved a piece, the model
    # still names slots 1-3, so a Sweep that lands a different three is Served for the
    # environment and a failed try for a method scoring by these effects.
    TWO_ROUTE_GATHER_AND_SWEEP: ClassVar[Skill] = GATHER_AND_SWEEP.model_copy(
        update={
            "add_effects": GATHER_AND_SWEEP.add_effects
            | {_atom(predicate=p) for p in (SERVED, SERVED_FIVE)}
        }
    )
    TWO_ROUTE_SWEEP: ClassVar[Skill] = SWEEP.model_copy(
        update={
            "add_effects": SWEEP.add_effects
            | {_atom(predicate=p) for p in (SERVED, SERVED_THREE)}
            | {_atom(predicate=ON_TRAY, variables=(food, _TRAY)) for food in _SLOTS[1:4]},
            "delete_effects": frozenset(
                _atom(predicate=ON_RACK, variables=(food, _RACK)) for food in _SLOTS[1:4]
            ),
            "ignore_effects": frozenset({ON_RACK, ANY_FOOD_ON_RACK}),
            "conditional_add_effects": frozenset(),
        }
    )
    TWO_ROUTE_HUMAN_BATCH_RETURN: ClassVar[Skill] = HUMAN_BATCH_RETURN.model_copy(
        update={
            "delete_effects": HUMAN_BATCH_RETURN.delete_effects
            | {_atom(predicate=p) for p in (SERVED, SERVED_THREE, SERVED_FIVE)}
        }
    )

    @staticmethod
    def robot_skills(
        *, include_sweep: bool = True, include_placement: bool = True, two_route: bool = False
    ) -> tuple[Skill, ...]:
        skills = RackServing3DSkills
        return (
            skills.PICK_SCRAPER,
            skills.TWO_ROUTE_GATHER_AND_SWEEP if two_route else skills.GATHER_AND_SWEEP,
            skills.RETURN_SCRAPER,
            skills.PICK_FOOD,
            *((skills.PLACE_FOOD_ON_TRAY,) if include_placement else ()),
            skills.RETURN_FOOD_TO_RACK,
        ) + (((skills.TWO_ROUTE_SWEEP if two_route else skills.SWEEP),) if include_sweep else ())

    @staticmethod
    def human_skills(
        *, mode: Literal["training", "deployment"] = "training", two_route: bool = False
    ) -> tuple[Skill, ...]:
        if mode == "deployment":
            return ()
        skills = RackServing3DSkills
        return (
            skills.TWO_ROUTE_HUMAN_BATCH_RETURN if two_route else skills.HUMAN_BATCH_RETURN,
            skills.HUMAN_TOOL_RECOVERY,
        )
