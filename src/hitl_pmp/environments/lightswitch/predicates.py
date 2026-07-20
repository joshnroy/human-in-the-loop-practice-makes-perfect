from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import LightSwitchEnvironment


class LightOnClassifier:
    """Whether a light's level is within tolerance of its target. A static-method
    container, never instantiated, same as every other business-logic class in this
    project."""

    @staticmethod
    def holds(*, state: State, light: Object) -> bool:
        level = state.get(obj=light, feature_name="level")
        target = state.get(obj=light, feature_name="target")
        return bool(abs(level - target) < LightSwitchEnvironment.light_on_tolerance)


class AdjacentClassifier:
    """Whether two cells are next to each other (exactly one grid unit apart).
    Predicators' actual implementation precomputes a neighbor set for planner-loop
    speed ("Adjacent is a bottleneck for speed"); irrelevant here since nothing
    grounds/searches over these predicates yet -- a direct distance check is the
    simpler, equally-correct choice for the current scope."""

    @staticmethod
    def holds(*, state: State, cell1: Object, cell2: Object) -> bool:
        x1 = state.get(obj=cell1, feature_name="x")
        x2 = state.get(obj=cell2, feature_name="x")
        return bool(abs(abs(x1 - x2) - 1.0) < LightSwitchEnvironment.same_position_tolerance)


# Predicate.holds is a positional (state, objects) callable per its interface
# contract (Goal.is_satisfied calls it that way) -- each lambda below just adapts
# that into a call to the relevant class's keyword-only holds.
LIGHT_ON = Predicate(
    name="LightOn",
    types=(LightSwitchEnvironment.light_type,),
    holds=lambda state, objects: LightOnClassifier.holds(state=state, light=objects[0]),
)

# RobotInCell and LightInCell are both literally the same relation (an object and a
# cell sharing a position), applied to different object pairs -- predicators itself
# uses one shared _In_holds for both, so both adapt directly to
# LightSwitchEnvironment.same_position rather than each getting their own classifier.
ROBOT_IN_CELL = Predicate(
    name="RobotInCell",
    types=(LightSwitchEnvironment.robot_type, LightSwitchEnvironment.cell_type),
    holds=lambda state, objects: LightSwitchEnvironment.same_position(
        state=state, obj1=objects[0], obj2=objects[1]
    ),
)

LIGHT_IN_CELL = Predicate(
    name="LightInCell",
    types=(LightSwitchEnvironment.light_type, LightSwitchEnvironment.cell_type),
    holds=lambda state, objects: LightSwitchEnvironment.same_position(
        state=state, obj1=objects[0], obj2=objects[1]
    ),
)

ADJACENT = Predicate(
    name="Adjacent",
    types=(LightSwitchEnvironment.cell_type, LightSwitchEnvironment.cell_type),
    holds=lambda state, objects: AdjacentClassifier.holds(
        state=state, cell1=objects[0], cell2=objects[1]
    ),
)
