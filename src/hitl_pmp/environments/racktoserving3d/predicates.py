"""State-attached measured facts; readiness is derived only from the clock."""

import itertools

from hitl_pmp.core.problem.environment.types import Object, State, Type
from hitl_pmp.core.problem.tasks.types import GroundAtom, Predicate

from .objects import RackServing3DObjects as Objects

AbstractAtom = tuple[str, tuple[str, ...]]


class RackServing3DAtoms:
    @staticmethod
    def holds(*, state: State, name: str, objects: tuple[Object, ...]) -> bool:
        atoms = getattr(state, "abstract_atoms", None)
        if atoms is None:
            raise ValueError("RackServing3D requires measured abstract_atoms on the state")
        return (name, tuple(obj.name for obj in objects)) in atoms

    @staticmethod
    def predicate(*, name: str, types: tuple[Type, ...]) -> Predicate:
        return Predicate(
            name=name,
            types=types,
            holds=lambda state, objects: RackServing3DAtoms.holds(
                state=state, name=name, objects=objects
            ),
        )

    @staticmethod
    def with_summaries(
        *,
        atoms: frozenset[AbstractAtom],
        available_food: frozenset[str] | None = None,
    ) -> frozenset[AbstractAtom]:
        """Refresh derived facts, including removing stale summary facts.

        The backend supplies exclusive measured locations via ``classify_food``.
        FoodAvailable allows autonomous picking from any observed support surface,
        including the tray and spill locations, without forcing a sweep solution.
        """
        derived_names = {
            "AnyFoodOnRack",
            "AnyFoodOffRack",
            "FoodAvailable",
            "ScraperAway",
        }
        result = {atom for atom in atoms if atom[0] not in derived_names}
        on_rack = {
            food.name
            for food in Objects.food
            if ("OnRack", (food.name, Objects.rack.name)) in result
        }
        if on_rack:
            result.add(("AnyFoodOnRack", ()))
        if len(on_rack) < len(Objects.food):
            result.add(("AnyFoodOffRack", ()))
        for food in Objects.food:
            supported = any(
                atom in result
                for atom in (
                    ("OnRack", (food.name, Objects.rack.name)),
                    ("OnTray", (food.name, Objects.tray.name)),
                    ("SpilledCounter", (food.name,)),
                    ("SpilledFloor", (food.name,)),
                )
            )
            if available_food is not None:
                supported = food.name in available_food
            if supported and ("HoldingFood", (Objects.robot.name, food.name)) not in result:
                result.add(("FoodAvailable", (food.name,)))
        if ("ScraperHome", (Objects.scraper.name,)) not in result:
            result.add(("ScraperAway", (Objects.scraper.name,)))
        return frozenset(result)


ON_RACK = RackServing3DAtoms.predicate(name="OnRack", types=(Objects.food_type, Objects.rack_type))
ON_TRAY = RackServing3DAtoms.predicate(name="OnTray", types=(Objects.food_type, Objects.tray_type))
SPILLED_COUNTER = RackServing3DAtoms.predicate(name="SpilledCounter", types=(Objects.food_type,))
SPILLED_FLOOR = RackServing3DAtoms.predicate(name="SpilledFloor", types=(Objects.food_type,))
HOLDING_FOOD = RackServing3DAtoms.predicate(
    name="HoldingFood", types=(Objects.robot_type, Objects.food_type)
)
HOLDING_SCRAPER = RackServing3DAtoms.predicate(
    name="HoldingScraper", types=(Objects.robot_type, Objects.scraper_type)
)
HAND_EMPTY = RackServing3DAtoms.predicate(name="HandEmpty", types=(Objects.robot_type,))
SCRAPER_HOME = RackServing3DAtoms.predicate(name="ScraperHome", types=(Objects.scraper_type,))
SCRAPER_AWAY = RackServing3DAtoms.predicate(name="ScraperAway", types=(Objects.scraper_type,))
SCRAPER_ON_FLOOR = RackServing3DAtoms.predicate(
    name="ScraperOnFloor", types=(Objects.scraper_type,)
)
FOOD_AVAILABLE = RackServing3DAtoms.predicate(name="FoodAvailable", types=(Objects.food_type,))
ANY_FOOD_ON_RACK = RackServing3DAtoms.predicate(name="AnyFoodOnRack", types=())
ANY_FOOD_OFF_RACK = RackServing3DAtoms.predicate(name="AnyFoodOffRack", types=())
_FOOD_CHAIN = frozenset(
    (first.name, second.name) for first, second in itertools.pairwise(Objects.food)
)
FOOD_ORDER = Predicate(
    name="FoodOrder",
    types=(Objects.food_type, Objects.food_type),
    holds=lambda state, objects: (objects[0].name, objects[1].name) in _FOOD_CHAIN,
)
DYNAMIC_PREDICATES = (
    ON_RACK,
    ON_TRAY,
    SPILLED_COUNTER,
    SPILLED_FLOOR,
    HOLDING_FOOD,
    HOLDING_SCRAPER,
    HAND_EMPTY,
    SCRAPER_HOME,
    SCRAPER_AWAY,
    SCRAPER_ON_FLOOR,
    FOOD_AVAILABLE,
    ANY_FOOD_ON_RACK,
    ANY_FOOD_OFF_RACK,
)


class RackServing3DPredicates:
    @staticmethod
    def all() -> tuple[Predicate, ...]:
        return (*DYNAMIC_PREDICATES, FOOD_ORDER)

    @staticmethod
    def static_atoms() -> frozenset[GroundAtom]:
        return frozenset(
            GroundAtom(predicate=FOOD_ORDER, objects=pair)
            for pair in itertools.pairwise(Objects.food)
        )

    @staticmethod
    def goal_atoms() -> frozenset[GroundAtom]:
        """Delivery is the goal, independently of how the robot accomplished it."""
        return frozenset(
            GroundAtom(predicate=ON_TRAY, objects=(food, Objects.tray)) for food in Objects.food
        )
