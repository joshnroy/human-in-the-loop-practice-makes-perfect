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


# Predicate.holds is a positional (state, objects) callable per its interface
# contract (Goal.is_satisfied calls it that way) -- this lambda just adapts that
# into a call to LightOnClassifier's keyword-only holds.
LIGHT_ON = Predicate(
    name="LightOn",
    types=(LightSwitchEnvironment.light_type,),
    holds=lambda state, objects: LightOnClassifier.holds(state=state, light=objects[0]),
)
