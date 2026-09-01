"""Practice-termination action."""

from pydantic import BaseModel, ConfigDict


class StopAction(BaseModel):
    """Sentinel selected when further practice has no value."""

    model_config = ConfigDict(frozen=True)


STOP_ACTION = StopAction()
NUM_SAMPLES = 1
