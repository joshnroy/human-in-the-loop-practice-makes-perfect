"""Two Routes' state-dependent practice cost: only the robot's own reset is priced."""

from typing import Any


class TwoRoutesCosts:
    @staticmethod
    def robot_step(*, context: dict[str, Any]) -> float:
        """A practice cost file's `robot_step` function: the environment's robot-reset
        price for a step taken from an end state, nothing for any other step."""
        observation = context.get("observation") or {}
        return (
            float(observation.get("robot_reset_price", 0.0))
            if observation.get("resetting")
            else 0.0
        )


# The cost file names a module attribute (`module:name`).
robot_step = TwoRoutesCosts.robot_step
