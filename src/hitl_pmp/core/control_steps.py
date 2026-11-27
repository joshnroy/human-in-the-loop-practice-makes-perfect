"""External control-step boundaries must survive controller failure handling."""

import functools
from typing import Any


class ControlStepLimitReached(BaseException):
    """The external physical interaction budget has ended."""


class ControlStepObserverFailed(BaseException):
    """An instrumentation failure is not a failed robot controller."""


class SimulatorDiverged(BaseException):
    """The physics simulator's own instability guard reset the scene mid-run.

    MuJoCo answers a NaN, Inf or huge `qpos`/`qvel`/`qacc` with `mj_resetData` and
    carries on, so whatever follows is a wiped scene, not the robot's doing. Not an
    `Exception`, so no skill's failure handling can score it as a failed skill.
    `time` is the simulated time at the start of the step the guard fired in."""

    def __init__(
        self,
        *,
        time: float,
        time_after_reset: float,
        warning: str,
        index: int,
        body: str | None,
        scene_step: int,
    ) -> None:
        self.details: dict[str, Any] = {
            "time": time,
            "time_after_reset": time_after_reset,
            "warning": warning,
            "index": index,
            "body": body,
            "scene_step": scene_step,
        }
        super().__init__(
            f"simulator diverged: {warning} at index {index} (body {body}) in the step "
            f"from t = {time:.4f} s, scene step {scene_step}; MuJoCo reset the scene, "
            f"now at t = {time_after_reset:.4f} s"
        )

    # Keyword-only arguments do not survive `BaseException`'s default pickling.
    def __reduce__(self) -> tuple[Any, ...]:
        return (functools.partial(SimulatorDiverged, **self.details), ())
