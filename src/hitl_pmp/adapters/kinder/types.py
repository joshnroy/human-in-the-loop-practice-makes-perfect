"""Plain data produced by this package. See `controllers.py` for the producer."""

from pydantic import BaseModel, ConfigDict


class ControllerRun(BaseModel):
    """What one KINDER controller execution did.

    `steps` is how many simulator steps it took to terminate -- a real quantity in this
    project's records (`docs/kinder-environment-validation.md` reports the oracle's
    per-controller counts), so it is returned rather than discarded.

    `error` is non-`None` when the controller raised. That is an ordinary outcome, not an
    exceptional one: KINDER's motion planners `assert plan is not None`, so an unreachable
    grasp or an unplannable arm trajectory surfaces as an `AssertionError` out of
    `reset()`. `core.Environment.take_action` must be total over its action space, so the
    failure has to cross this boundary as data rather than as an exception.
    """

    model_config = ConfigDict(frozen=True)

    steps: int
    terminated: bool
    error: str | None = None
