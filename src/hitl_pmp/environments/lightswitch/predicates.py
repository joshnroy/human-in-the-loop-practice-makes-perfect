from hitl_pmp.core.problem.environment.types import Object, State
from hitl_pmp.core.problem.tasks.types import Predicate

from .environment import LightSwitchEnvironment


def _is_light_on(*, state: State, light: Object) -> bool:
    level = state.get(obj=light, feature_name="level")
    target = state.get(obj=light, feature_name="target")
    return bool(abs(level - target) < LightSwitchEnvironment.light_on_tolerance)


# Predicate.holds is a positional (state, objects) callable per its interface
# contract (Goal.is_satisfied calls it that way) -- this lambda just adapts that
# into a call to the keyword-only helper above.
LIGHT_ON = Predicate(
    name="LightOn",
    types=(LightSwitchEnvironment.light_type,),
    holds=lambda state, objects: _is_light_on(state=state, light=objects[0]),
)
